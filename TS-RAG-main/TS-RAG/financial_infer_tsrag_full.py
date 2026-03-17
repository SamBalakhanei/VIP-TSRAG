import os
import argparse
import random
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from transformers import AutoConfig

from utils.metrics import metric as metric_fn
from models.ChronosBolt import (
    ChronosBoltModelForForecastingWithRetrieval,
    ChronosBoltPipelineWithRetrieval,
)


def load_close_series(jsonl_path: str, time_col: str = "timestamp", price_col: str = "close") -> np.ndarray:
    df = pd.read_json(jsonl_path, lines=True)
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in {jsonl_path}")
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in {jsonl_path}")
    df = df.sort_values(time_col)
    return df[price_col].astype(float).to_numpy()


def load_retrieval_db(
    retrieval_parquet: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load retrieval DB parquet built by build_stocks_retrieval_pretrain_full.py.
    Returns:
      - embeddings: [N, d]
      - x_seq: array of object, each entry np.ndarray (lookback_length,)
      - y_seq: array of object, each entry np.ndarray (prediction_length,)
    """
    db = pd.read_parquet(retrieval_parquet)
    if "embedding" not in db.columns or "x" not in db.columns or "y" not in db.columns:
        raise ValueError(f"'embedding', 'x', or 'y' missing in {retrieval_parquet}")

    embeddings = np.vstack(db["embedding"].to_numpy()).astype("float32")
    x_seq = db["x"].values
    y_seq = db["y"].values
    return embeddings, x_seq, y_seq


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index


def retrieve_for_context(
    context: np.ndarray,
    pipeline: ChronosPipeline,
    index: faiss.IndexFlatL2,
    x_seq,
    y_seq,
    lookback_length: int,
    prediction_length: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a 1D numpy context array, perform retrieval:
      - embed last lookback_length points with ChronosPipeline
      - search FAISS index
      - construct retrieved_seq [1, top_k, lookback_length+prediction_length]
      - distances [1, top_k]
    """
    if context.shape[0] < lookback_length:
        raise ValueError(
            f"context length {context.shape[0]} < lookback_length {lookback_length} "
            f"(cannot perform full retrieval)."
        )

    ctx_win = context[-lookback_length:].astype(np.float32)
    ctx_tensor = torch.tensor(ctx_win[None, :], dtype=torch.float32)
    with torch.no_grad():
        q_emb, _ = pipeline.embed(ctx_tensor)
        q_vec = q_emb[:, -1, :].float().cpu().numpy().astype("float32")

    distances, indices = index.search(q_vec, top_k)
    distances = distances[0].astype(np.float32)
    indices = indices[0].astype(np.int64)

    retrieved_list = []
    for idx in indices:
        x = x_seq[idx].astype(np.float32)
        y = y_seq[idx].astype(np.float32)
        seq = np.concatenate([x, y], axis=-1)
        retrieved_list.append(seq)

    retrieved_arr = np.stack(retrieved_list, axis=0)  # [top_k, L_total]
    retrieved_tensor = torch.tensor(retrieved_arr, dtype=torch.float32).unsqueeze(0)  # [1, top_k, L_total]
    dist_tensor = torch.tensor(distances, dtype=torch.float32).unsqueeze(0)  # [1, top_k]
    return retrieved_tensor, dist_tensor


def run_forecast_for_ticker_full(
    pipe: ChronosBoltPipelineWithRetrieval,
    retrieval_index: faiss.IndexFlatL2,
    embeddings: np.ndarray,
    x_seq,
    y_seq,
    stocks_dir: str,
    ticker: str,
    time_col: str,
    price_col: str,
    prediction_length: int,
    compute_metrics: bool,
    device: torch.device,
    embed_pipeline: ChronosPipeline,
    lookback_length: int,
) -> None:
    jsonl_path = os.path.join(stocks_dir, f"{ticker}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[WARN] File not found for ticker {ticker}: {jsonl_path}")
        return

    series = load_close_series(jsonl_path, time_col=time_col, price_col=price_col)

    print(f"\n=== {ticker} (TS-RAG full) ===")
    print(f"Loaded {len(series)} points from {jsonl_path}")

    if compute_metrics and len(series) > prediction_length:
        context_arr = series[:-prediction_length]
        true_future = series[-prediction_length:]
    else:
        context_arr = series
        true_future = None

    # Retrieval based on last lookback_length of context
    retrieved_seq, distances = retrieve_for_context(
        context=context_arr,
        pipeline=embed_pipeline,
        index=retrieval_index,
        x_seq=x_seq,
        y_seq=y_seq,
        lookback_length=lookback_length,
        prediction_length=prediction_length,
        top_k=10,
    )

    context_tensor = torch.tensor(context_arr, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        forecast = pipe.predict(
            context=context_tensor,
            prediction_length=prediction_length,
            retrieved_seq=retrieved_seq.to(device),
            distances=distances.to(device),
        )

    quantiles = pipe.quantiles
    if 0.5 in quantiles:
        q_idx = quantiles.index(0.5)
    else:
        q_idx = len(quantiles) // 2

    median_forecast = forecast[0, q_idx].detach().cpu().numpy()

    print(f"Prediction length: {prediction_length}")
    print(f"Quantiles: {quantiles}")
    print("Median forecast (TS-RAG full, with retrieval):")
    print(median_forecast)

    if compute_metrics and true_future is not None:
        pred_arr = median_forecast.reshape(-1)
        true_arr = true_future.reshape(-1)
        mae, mse, rmse, mape, mspe, smape, nd = metric_fn(pred_arr, true_arr)
        print("Metrics (TS-RAG full, using last prediction_length points as ground truth):")
        print(f"  MAE:   {mae:.6f}")
        print(f"  MSE:   {mse:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAPE:  {mape:.6f}")
        print(f"  MSPE:  {mspe:.6f}")
        print(f"  SMAPE: {smape:.6f}")
        print(f"  ND:    {nd:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full TS-RAG (ChronosBoltRetrieve with retrieval at inference) on financial JSONL stock data."
    )
    parser.add_argument("--stocks_dir", type=str, default="../sampled_stocks/new_directory")
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="*",
        default=None,
        help="List of tickers (without .jsonl). If omitted, sample random tickers.",
    )
    parser.add_argument("--num_stocks", type=int, default=5, help="Number of random stocks if tickers not provided.")
    parser.add_argument("--time_col", type=str, default="timestamp")
    parser.add_argument("--price_col", type=str, default="close")
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument(
        "--model_id",
        type=str,
        default="./checkpoints/ChronosBoltRetrieve_Stocks_TSRAG_full",
        help="Directory containing the full TS-RAG fine-tuned model (ChronosBoltRetrieve).",
    )
    parser.add_argument(
        "--retrieval_parquet",
        type=str,
        default="../database/pretrain/stocks_retrieval_database_full_512.parquet",
        help="Parquet retrieval DB built by build_stocks_retrieval_pretrain_full.py.",
    )
    parser.add_argument("--device", type=str, default="cpu", help='Use "cpu" or "cuda".')
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="If set, use the last prediction_length points as ground truth to compute error metrics.",
    )
    parser.add_argument("--lookback_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--embedding_model_id",
        type=str,
        default="amazon/chronos-t5-base",
        help="Chronos embedding model used to build the retrieval DB.",
    )

    args = parser.parse_args()

    if args.tickers is None or len(args.tickers) == 0:
        all_files = [f for f in os.listdir(args.stocks_dir) if f.endswith(".jsonl")]
        if not all_files:
            raise RuntimeError(f"No *.jsonl files found in {args.stocks_dir}")
        random.shuffle(all_files)
        selected = all_files[: args.num_stocks]
        tickers = [os.path.splitext(f)[0] for f in selected]
    else:
        tickers = args.tickers

    print(f"Tickers to run (TS-RAG full): {tickers}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Load retrieval DB & index
    retrieval_parquet_path = Path(args.retrieval_parquet)
    embeddings, x_seq, y_seq = load_retrieval_db(retrieval_parquet_path)
    index = build_faiss_index(embeddings)

    # Embedding model for queries
    embed_pipeline = ChronosPipeline.from_pretrained(
        args.embedding_model_id,
        device_map=args.device,
        torch_dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
    )

    # Load full TS-RAG model
    config = AutoConfig.from_pretrained(args.model_id)
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        args.model_id,
        config=config,
        augment="moe2",
    )
    model.to(device)
    model.eval()
    pipe = ChronosBoltPipelineWithRetrieval(model=model)

    for t in tickers:
        run_forecast_for_ticker_full(
            pipe=pipe,
            retrieval_index=index,
            embeddings=embeddings,
            x_seq=x_seq,
            y_seq=y_seq,
            stocks_dir=args.stocks_dir,
            ticker=t,
            time_col=args.time_col,
            price_col=args.price_col,
            prediction_length=args.prediction_length,
            compute_metrics=args.compute_metrics,
            device=device,
            embed_pipeline=embed_pipeline,
            lookback_length=args.lookback_length,
        )


if __name__ == "__main__":
    main()

