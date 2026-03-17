import math
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline


def sliding_windows(series: np.ndarray, window: int) -> np.ndarray:
    n = len(series)
    if n < window:
        return np.empty((0, window), dtype=series.dtype)
    num = n - window + 1
    out = np.lib.stride_tricks.as_strided(
        series,
        shape=(num, window),
        strides=(series.strides[0], series.strides[0]),
    )
    return out.copy()


def build_retrieval_database_for_stocks(
    stocks_csv: Path,
    output_parquet: Path,
    lookback_length: int,
    prediction_length: int,
    chronos_model_id: str = "amazon/chronos-t5-base",
    device: str = "cpu",
) -> int:
    """
    Build a retrieval database parquet for stocks.csv compatible with Retriever_for_pretrain:
    columns: ['ticker', 'x', 'y', 'embedding'] where:
      - x: past window of length lookback_length
      - y: future window of length prediction_length
      - embedding: Chronos embedding of x (EOS representation)
    Returns the embedding dimension (d).
    """
    df = pd.read_csv(stocks_csv)
    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {stocks_csv}")

    tickers = df.columns[1:]
    if len(tickers) == 0:
        raise RuntimeError(f"No ticker columns found in {stocks_csv}")

    print(f"[full] Building retrieval database from {stocks_csv}")
    print(f"[full] Found {len(tickers)} tickers: {list(tickers)[:5]}{' ...' if len(tickers) > 5 else ''}")

    pipeline = ChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )

    all_rows = []
    emb_dim: int = -1

    for ticker in tickers:
        series = df[ticker].astype(float).to_numpy()
        total_len = lookback_length + prediction_length
        if len(series) < total_len:
            print(f"[full] Skipping {ticker}: length {len(series)} < {total_len}")
            continue

        num = len(series) - total_len + 1
        print(f"[full] Ticker {ticker}: {len(series)} points, {num} windows")

        batch_size = 256
        num_batches = math.ceil(num / batch_size)

        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, num)

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
                # embeddings: [B, L_embed, d]; take EOS
                eos_embeddings = embeddings[:, -1, :].float().cpu().numpy()

            if emb_dim < 0:
                emb_dim = eos_embeddings.shape[-1]

            for i in range(x_arr.shape[0]):
                all_rows.append(
                    {
                        "ticker": ticker,
                        "x": x_arr[i],
                        "y": y_arr[i],
                        "embedding": eos_embeddings[i],
                    }
                )

    if not all_rows:
        raise RuntimeError("[full] No windows generated; check lengths and data.")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    db_df = pd.DataFrame(all_rows)
    db_df.to_parquet(output_parquet, index=False)
    print(f"[full] Saved stocks retrieval database to {output_parquet} (emb_dim={emb_dim})")
    return emb_dim


def build_pretrain_parquet_for_stocks(
    stocks_csv: Path,
    retrieval_parquet: Path,
    output_dir: Path,
    lookback_length: int,
    prediction_length: int,
    top_k: int,
    chronos_model_id: str = "amazon/chronos-t5-base",
    device: str = "cpu",
) -> None:
    """
    Build a pretrain parquet dataset compatible with CustomPretrainDataset:
      - 'target': full window [context, future] of length lookback_length+prediction_length
      - 'indices': nearest-neighbor indices in retrieval DB
      - 'distances': corresponding FAISS distances
    Uses the same ChronosPipeline as the retrieval DB for query embeddings.
    """
    df = pd.read_csv(stocks_csv)
    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {stocks_csv}")

    tickers = df.columns[1:]
    if len(tickers) == 0:
        raise RuntimeError(f"No ticker columns found in {stocks_csv}")

    print(f"[full] Building pretrain parquet from {stocks_csv}")

    # Load retrieval DB
    db = pd.read_parquet(retrieval_parquet)
    if "embedding" not in db.columns:
        raise ValueError(f"[full] 'embedding' column not found in {retrieval_parquet}")

    embeddings = np.vstack(db["embedding"].to_numpy()).astype("float32")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    pipeline = ChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
    )

    total_len = lookback_length + prediction_length
    all_rows = []

    for ticker in tickers:
        series = df[ticker].astype(float).to_numpy()
        if len(series) < total_len:
            print(f"[full] Skipping {ticker} for pretrain parquet: length {len(series)} < {total_len}")
            continue

        num = len(series) - total_len + 1
        print(f"[full] Ticker {ticker}: {num} windows for pretraining")

        batch_size = 256
        num_batches = math.ceil(num / batch_size)

        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, num)

            windows = []
            contexts = []
            for i in range(start, end):
                w = series[i : i + total_len].astype(np.float32)
                windows.append(w)
                contexts.append(w[:lookback_length])

            window_arr = np.stack(windows, axis=0)
            context_arr = np.stack(contexts, axis=0)

            context_tensor = torch.tensor(context_arr, dtype=torch.float32)
            with torch.no_grad():
                q_emb, _ = pipeline.embed(context_tensor)
                q_vec = q_emb[:, -1, :].float().cpu().numpy().astype("float32")

            distances, indices = index.search(q_vec, top_k)

            for i in range(window_arr.shape[0]):
                all_rows.append(
                    {
                        "target": window_arr[i],
                        "indices": indices[i].astype(np.int64),
                        "distances": distances[i].astype(np.float32),
                    }
                )

    if not all_rows:
        raise RuntimeError("[full] No pretrain windows generated; check lengths and data.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stocks_pretrain_full.parquet"
    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(out_path, index=False)
    print(f"[full] Saved full TS-RAG stocks pretrain parquet to {out_path}")


def main():
    repo_root = Path(__file__).resolve().parent
    stocks_csv = repo_root / "datasets" / "stocks" / "stocks.csv"

    retrieval_parquet = repo_root.parent / "database" / "pretrain" / "stocks_retrieval_database_full_512.parquet"
    pretrain_dir = repo_root / "datasets" / "pretrain" / "stocks-with-retrieval_full_512"

    lookback_length = 512
    prediction_length = 64
    top_k = 10

    if not stocks_csv.exists():
        raise FileNotFoundError(
            f"{stocks_csv} not found. Run build_stocks_csv.py first to create the aggregated stocks.csv."
        )

    emb_dim = build_retrieval_database_for_stocks(
        stocks_csv=stocks_csv,
        output_parquet=retrieval_parquet,
        lookback_length=lookback_length,
        prediction_length=prediction_length,
        chronos_model_id="amazon/chronos-t5-base",
        device="cpu",
    )
    print(f"[full] Retrieval DB built with embedding dim {emb_dim}")

    build_pretrain_parquet_for_stocks(
        stocks_csv=stocks_csv,
        retrieval_parquet=retrieval_parquet,
        output_dir=pretrain_dir,
        lookback_length=lookback_length,
        prediction_length=prediction_length,
        top_k=top_k,
        chronos_model_id="amazon/chronos-t5-base",
        device="cpu",
    )


if __name__ == "__main__":
    main()

