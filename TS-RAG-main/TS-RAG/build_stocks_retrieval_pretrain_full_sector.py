import argparse
import math
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from sector_metadata import PRETRAIN_TICKERS, metadata_for_ticker


def configure_device(device: str) -> None:
    if device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print("[sector] Configured CPU threading: torch.set_num_threads(1), torch.set_num_interop_threads(1)")


def resolve_pretrain_tickers(all_tickers: list[str]) -> list[str]:
    selected = [t for t in PRETRAIN_TICKERS if t in all_tickers]
    missing = [t for t in PRETRAIN_TICKERS if t not in all_tickers]
    if missing:
        print(f"[sector] Warning: missing configured pretrain tickers: {missing}")
    if not selected:
        raise RuntimeError("[sector] No configured pretrain tickers found in stocks.csv")
    return selected


def rerank_with_sector(
    query_ticker: str,
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    candidate_tickers: np.ndarray,
    top_k: int,
    sector_bonus: float,
    naics_bonus: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_meta = metadata_for_ticker(query_ticker)
    scores = candidate_distances.astype(np.float32).copy()

    for i, idx in enumerate(candidate_indices):
        c_ticker = str(candidate_tickers[idx])
        c_meta = metadata_for_ticker(c_ticker)
        if q_meta["sector"] != "unknown" and q_meta["sector"] == c_meta["sector"]:
            scores[i] -= sector_bonus
        if q_meta["naics2"] != "unknown" and q_meta["naics2"] == c_meta["naics2"]:
            scores[i] -= naics_bonus

    order = np.argsort(scores)[:top_k]
    return candidate_indices[order], scores[order]


def build_retrieval_database_for_stocks(
    stocks_csv: Path,
    output_parquet: Path,
    lookback_length: int,
    prediction_length: int,
    chronos_model_id: str,
    device: str,
) -> int:
    df = pd.read_csv(stocks_csv)
    all_tickers = list(df.columns[1:])
    tickers = resolve_pretrain_tickers(all_tickers)

    configure_device(device)
    pipeline = ChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )

    rows = []
    emb_dim = -1
    total_len = lookback_length + prediction_length

    for ticker in tickers:
        series = df[ticker].astype(float).to_numpy()
        if len(series) < total_len:
            continue
        num = len(series) - total_len + 1
        batch_size = 256 if device != "cpu" else 32

        for start in range(0, num, batch_size):
            end = min(start + batch_size, num)
            xs = []
            ys = []
            for i in range(start, end):
                w = series[i : i + total_len]
                xs.append(w[:lookback_length].astype(np.float32))
                ys.append(w[lookback_length:].astype(np.float32))
            x_arr = np.stack(xs, axis=0)
            y_arr = np.stack(ys, axis=0)
            x_tensor = torch.tensor(x_arr, dtype=torch.float32)
            with torch.no_grad():
                embeddings, _ = pipeline.embed(x_tensor)
                eos_embeddings = embeddings[:, -1, :].float().cpu().numpy()
            if emb_dim < 0:
                emb_dim = eos_embeddings.shape[-1]

            meta = metadata_for_ticker(ticker)
            for i in range(x_arr.shape[0]):
                rows.append(
                    {
                        "ticker": ticker,
                        "sector": meta["sector"],
                        "naics2": meta["naics2"],
                        "x": x_arr[i],
                        "y": y_arr[i],
                        "embedding": eos_embeddings[i],
                    }
                )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_parquet, index=False)
    print(f"[sector] Saved retrieval DB to {output_parquet}")
    return emb_dim


def build_pretrain_parquet_for_stocks(
    stocks_csv: Path,
    retrieval_parquet: Path,
    output_dir: Path,
    lookback_length: int,
    prediction_length: int,
    top_k: int,
    chronos_model_id: str,
    device: str,
    candidate_multiplier: int,
    sector_bonus: float,
    naics_bonus: float,
) -> None:
    df = pd.read_csv(stocks_csv)
    tickers = resolve_pretrain_tickers(list(df.columns[1:]))

    db = pd.read_parquet(retrieval_parquet)
    embeddings = np.vstack(db["embedding"].to_numpy()).astype("float32")
    db_tickers = db["ticker"].astype(str).to_numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    configure_device(device)
    pipeline = ChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )

    total_len = lookback_length + prediction_length
    candidate_k = max(top_k * candidate_multiplier, top_k)
    rows = []

    for ticker in tickers:
        series = df[ticker].astype(float).to_numpy()
        if len(series) < total_len:
            continue
        num = len(series) - total_len + 1
        batch_size = 256 if device != "cpu" else 32

        for start in range(0, num, batch_size):
            end = min(start + batch_size, num)
            windows = []
            contexts = []
            for i in range(start, end):
                w = series[i : i + total_len].astype(np.float32)
                windows.append(w)
                contexts.append(w[:lookback_length])

            window_arr = np.stack(windows, axis=0)
            context_arr = np.stack(contexts, axis=0)
            with torch.no_grad():
                q_emb, _ = pipeline.embed(torch.tensor(context_arr, dtype=torch.float32))
                q_vec = q_emb[:, -1, :].float().cpu().numpy().astype("float32")
            distances, indices = index.search(q_vec, candidate_k)

            for i in range(window_arr.shape[0]):
                rr_idx, rr_dist = rerank_with_sector(
                    query_ticker=ticker,
                    candidate_indices=indices[i].astype(np.int64),
                    candidate_distances=distances[i].astype(np.float32),
                    candidate_tickers=db_tickers,
                    top_k=top_k,
                    sector_bonus=sector_bonus,
                    naics_bonus=naics_bonus,
                )
                rows.append(
                    {
                        "target": window_arr[i],
                        "indices": rr_idx.astype(np.int64),
                        "distances": rr_dist.astype(np.float32),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stocks_pretrain_full_sector.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[sector] Saved pretrain parquet to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build sector-aware full TS-RAG artifacts.")
    parser.add_argument("--lookback_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    parser.add_argument("--sector_bonus", type=float, default=0.10)
    parser.add_argument("--naics_bonus", type=float, default=0.15)
    parser.add_argument("--embedding_model_id", type=str, default="amazon/chronos-t5-base")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    stocks_csv = repo_root / "datasets" / "stocks" / "stocks.csv"
    retrieval_parquet = repo_root.parent / "database" / "pretrain" / "stocks_retrieval_database_full_sector_512.parquet"
    pretrain_dir = repo_root / "datasets" / "pretrain" / "stocks-with-retrieval_full_sector_512"
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sector] Using embedding device: {device}")

    emb_dim = build_retrieval_database_for_stocks(
        stocks_csv=stocks_csv,
        output_parquet=retrieval_parquet,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        chronos_model_id=args.embedding_model_id,
        device=device,
    )
    print(f"[sector] Retrieval DB built with embedding dim {emb_dim}")

    build_pretrain_parquet_for_stocks(
        stocks_csv=stocks_csv,
        retrieval_parquet=retrieval_parquet,
        output_dir=pretrain_dir,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        top_k=args.top_k,
        chronos_model_id=args.embedding_model_id,
        device=device,
        candidate_multiplier=args.candidate_multiplier,
        sector_bonus=args.sector_bonus,
        naics_bonus=args.naics_bonus,
    )


if __name__ == "__main__":
    main()

