import argparse
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from transformers import AutoConfig

from models.ChronosBolt import (
    ChronosBoltModelForForecastingWithRetrieval,
    ChronosBoltPipelineWithRetrieval,
)
from sector_metadata import EVAL_TICKERS, metadata_for_ticker
from utils.metrics import metric as metric_fn


def mase(pred: np.ndarray, true: np.ndarray, insample: np.ndarray, seasonality: int = 1) -> float:
    if len(insample) <= seasonality:
        return float("nan")
    scale = np.mean(np.abs(insample[seasonality:] - insample[:-seasonality]))
    if scale <= 1e-12:
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / scale)


def weighted_quantile_loss(quantile_forecast: np.ndarray, true: np.ndarray, quantiles: list[float]) -> float:
    denom = np.sum(np.abs(true)) + 1e-8
    total = 0.0
    for qi, q in enumerate(quantiles):
        diff = true - quantile_forecast[qi]
        total += np.sum(np.maximum(q * diff, (q - 1.0) * diff))
    return float((2.0 * total) / (len(quantiles) * denom))


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
        c_meta = metadata_for_ticker(str(candidate_tickers[idx]))
        if q_meta["sector"] != "unknown" and q_meta["sector"] == c_meta["sector"]:
            scores[i] -= sector_bonus
        if q_meta["naics2"] != "unknown" and q_meta["naics2"] == c_meta["naics2"]:
            scores[i] -= naics_bonus
    order = np.argsort(scores)[:top_k]
    return candidate_indices[order], scores[order]


def load_close_series(jsonl_path: str, time_col: str = "timestamp", price_col: str = "close") -> np.ndarray:
    df = pd.read_json(jsonl_path, lines=True).sort_values(time_col)
    return df[price_col].astype(float).to_numpy()


def main():
    parser = argparse.ArgumentParser(description="Sector-aware full TS-RAG inference with metadata reranking.")
    parser.add_argument("--stocks_dir", type=str, default="../sampled_stocks/new_directory")
    parser.add_argument("--tickers", type=str, nargs="*", default=None)
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--model_id", type=str, default="./checkpoints/ChronosBoltRetrieve_Stocks_TSRAG_full_sector")
    parser.add_argument("--retrieval_parquet", type=str, default="../database/pretrain/stocks_retrieval_database_full_sector_512.parquet")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compute_metrics", action="store_true")
    parser.add_argument("--lookback_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    parser.add_argument("--sector_bonus", type=float, default=0.10)
    parser.add_argument("--naics_bonus", type=float, default=0.15)
    parser.add_argument("--embedding_model_id", type=str, default="amazon/chronos-t5-base")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else EVAL_TICKERS
    print(f"Tickers to run (sector-aware TS-RAG full): {tickers}")

    db = pd.read_parquet(Path(args.retrieval_parquet))
    embeddings = np.vstack(db["embedding"].to_numpy()).astype("float32")
    x_seq = db["x"].values
    y_seq = db["y"].values
    db_tickers = db["ticker"].astype(str).to_numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    embed_pipeline = ChronosPipeline.from_pretrained(
        args.embedding_model_id,
        device_map=args.device,
        torch_dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
    )
    config = AutoConfig.from_pretrained(args.model_id)
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(args.model_id, config=config, augment="moe2")
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device).eval()
    pipe = ChronosBoltPipelineWithRetrieval(model=model)

    results = []
    candidate_k = max(args.top_k * args.candidate_multiplier, args.top_k)

    for ticker in tickers:
        jsonl_path = os.path.join(args.stocks_dir, f"{ticker}.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"[WARN] Missing {ticker}: {jsonl_path}")
            continue
        series = load_close_series(jsonl_path)
        context_arr = series[:-args.prediction_length] if args.compute_metrics and len(series) > args.prediction_length else series
        true_future = series[-args.prediction_length:] if args.compute_metrics and len(series) > args.prediction_length else None
        ctx_win = context_arr[-args.lookback_length:].astype(np.float32)
        with torch.no_grad():
            q_emb, _ = embed_pipeline.embed(torch.tensor(ctx_win[None, :], dtype=torch.float32))
            q_vec = q_emb[:, -1, :].float().cpu().numpy().astype("float32")
        distances, indices = index.search(q_vec, candidate_k)
        rr_idx, rr_dist = rerank_with_sector(
            query_ticker=ticker,
            candidate_indices=indices[0].astype(np.int64),
            candidate_distances=distances[0].astype(np.float32),
            candidate_tickers=db_tickers,
            top_k=args.top_k,
            sector_bonus=args.sector_bonus,
            naics_bonus=args.naics_bonus,
        )
        retrieved = np.stack([np.concatenate([x_seq[i].astype(np.float32), y_seq[i].astype(np.float32)]) for i in rr_idx], axis=0)
        with torch.no_grad():
            forecast = pipe.predict(
                context=torch.tensor(context_arr, dtype=torch.float32, device=device).unsqueeze(0),
                prediction_length=args.prediction_length,
                retrieved_seq=torch.tensor(retrieved, dtype=torch.float32, device=device).unsqueeze(0),
                distances=torch.tensor(rr_dist, dtype=torch.float32, device=device).unsqueeze(0),
            )
        quantiles = pipe.quantiles
        q_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        median_forecast = forecast[0, q_idx].detach().cpu().numpy()
        print(f"\n=== {ticker} (sector-aware TS-RAG full) ===")
        print(median_forecast)

        if args.compute_metrics and true_future is not None:
            pred_arr = median_forecast.reshape(-1)
            true_arr = true_future.reshape(-1)
            mae, mse, rmse, mape, mspe, smape, nd = metric_fn(pred_arr, true_arr)
            mase_val = mase(pred_arr, true_arr, context_arr.reshape(-1), seasonality=1)
            wql_val = weighted_quantile_loss(forecast[0].detach().cpu().numpy(), true_arr, quantiles)
            print(f"MASE={mase_val:.6f} WQL={wql_val:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")
            results.append({"MAE": mae, "MSE": mse, "RMSE": rmse, "MASE": mase_val, "WQL": wql_val, "MAPE": mape, "MSPE": mspe, "SMAPE": smape, "ND": nd})

    if args.compute_metrics and results:
        print("\n=== Aggregate metrics across evaluated tickers ===")
        mase_vals = np.array([r["MASE"] for r in results], dtype=np.float64)
        wql_vals = np.array([r["WQL"] for r in results], dtype=np.float64)
        mase_vals = mase_vals[~np.isnan(mase_vals)]
        if mase_vals.size > 0:
            print(f"  MASE: {float(np.mean(mase_vals)):.6f}")
        else:
            print("  MASE: nan")
        print(f"  WQL:  {float(np.mean(wql_vals)):.6f}")


if __name__ == "__main__":
    main()

