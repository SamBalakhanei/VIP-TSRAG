import os
import math
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch


def sliding_windows(
    series: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Simple sliding-window view over a 1D numpy array.

    Returns an array of shape (num_windows, window).
    """
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


def build_stocks_retrieval_database(
    stocks_csv: Path,
    output_parquet: Path,
    lookback_length: int,
) -> None:
    """
    Build a retrieval database for stocks.csv, storing for each window:
    - x: lookback_length past values
    - embedding: fixed-size embedding of x (used for FAISS retrieval)
    """
    df = pd.read_csv(stocks_csv)
    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {stocks_csv}")

    tickers = df.columns[1:]
    if len(tickers) == 0:
        raise RuntimeError(f"No ticker columns found in {stocks_csv}")

    print(f"Building retrieval database from {stocks_csv}")
    print(f"Found {len(tickers)} tickers: {list(tickers)[:5]}{' ...' if len(tickers) > 5 else ''}")

    # We choose a fixed embedding dimension compatible with pretrain.py (d=768).
    embedding_dim = 768

    all_rows = []

    for ticker in tickers:
        series = df[ticker].astype(float).to_numpy()
        windows = sliding_windows(series, lookback_length)
        if windows.size == 0:
            print(f"Skipping {ticker}: not enough points for lookback_length={lookback_length}")
            continue

        print(f"Ticker {ticker}: {len(series)} points, {windows.shape[0]} windows")

        batch_size = 512
        num_batches = math.ceil(windows.shape[0] / batch_size)

        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, windows.shape[0])
            batch_x = windows[start:end]

            # Simple deterministic embedding:
            # - normalize each window
            # - tile / pad / truncate to embedding_dim
            x_norm = (batch_x - batch_x.mean(axis=1, keepdims=True)) / (
                batch_x.std(axis=1, keepdims=True) + 1e-8
            )
            flat = x_norm.astype(np.float32)

            # expand to embedding_dim
            if flat.shape[1] >= embedding_dim:
                emb = flat[:, -embedding_dim:]
            else:
                reps = (embedding_dim + flat.shape[1] - 1) // flat.shape[1]
                tiled = np.tile(flat, (1, reps))
                emb = tiled[:, -embedding_dim:]

            for i in range(batch_x.shape[0]):
                all_rows.append(
                    {
                        "ticker": ticker,
                        "x": batch_x[i].astype(np.float32),
                        "embedding": emb[i].astype(np.float32),
                    }
                )

    if not all_rows:
        raise RuntimeError("No windows were generated for any ticker; check lookback_length and data.")

    out_dir = output_parquet.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Store as a DataFrame with object columns for x + embedding; downstream code will np.vstack embeddings
    db_df = pd.DataFrame(all_rows)
    db_df.to_parquet(output_parquet, index=False)
    print(f"Saved stocks retrieval database to {output_parquet}")


def build_stocks_pretrain_parquet(
    stocks_csv: Path,
    retrieval_parquet: Path,
    output_dir: Path,
    context_length: int,
    prediction_length: int,
    top_k: int,
) -> None:
    """
    Build a pretrain parquet dataset compatible with CustomPretrainDataset from stocks.csv,
    using the provided retrieval database parquet.

    Each row in the output will contain:
    - target: concatenated [context, future] of length context_length + prediction_length
    - indices: top_k indices into the retrieval DB
    - distances: corresponding FAISS distances
    """
    df = pd.read_csv(stocks_csv)
    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {stocks_csv}")

    tickers = df.columns[1:]
    if len(tickers) == 0:
        raise RuntimeError(f"No ticker columns found in {stocks_csv}")

    print(f"Building pretrain parquet from {stocks_csv}")

    # Load retrieval DB
    db = pd.read_parquet(retrieval_parquet)
    if "embedding" not in db.columns:
        raise ValueError(f"'embedding' column not found in retrieval parquet {retrieval_parquet}")

    embeddings = np.vstack(db["embedding"].to_numpy()).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    total_seq_len = context_length + prediction_length
    all_rows = []

    for ticker in tickers:
        series = df[ticker].astype(float).to_numpy()
        if len(series) < total_seq_len:
            print(f"Skipping {ticker} in pretrain parquet: length {len(series)} < {total_seq_len}")
            continue

        print(f"Ticker {ticker}: building windows for pretraining")

        num = len(series) - total_seq_len + 1
        for start in range(num):
            end = start + total_seq_len
            window = series[start:end].astype(np.float32)
            context = window[:context_length]

            # Query retrieval index with context
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            # Here we simply reuse the raw context as embedding input by zero-padding/truncating;
            # in practice, you'd want to use the same Chronos embedding as in the DB builder.
            query_vec = context_tensor.numpy().astype("float32")
            if query_vec.shape[1] != embeddings.shape[1]:
                # simple pad/truncate along length dimension to match embedding dim
                if query_vec.shape[1] > embeddings.shape[1]:
                    query_vec = query_vec[:, -embeddings.shape[1] :]
                else:
                    pad_width = embeddings.shape[1] - query_vec.shape[1]
                    query_vec = np.pad(query_vec, ((0, 0), (pad_width, 0)), mode="constant")

            distances, indices = index.search(query_vec, top_k)
            distances = distances[0].astype(np.float32)
            indices = indices[0].astype(np.int64)

            all_rows.append(
                {
                    "target": window,
                    "indices": indices,
                    "distances": distances,
                }
            )

    if not all_rows:
        raise RuntimeError("No pretrain windows were generated; check context/prediction lengths and data.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stocks_pretrain.parquet"
    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved stocks pretrain parquet to {out_path}")


def main():
    repo_root = Path(__file__).resolve().parent
    stocks_csv = repo_root / "datasets" / "stocks" / "stocks.csv"

    retrieval_parquet = repo_root.parent / "database" / "pretrain" / "stocks_retrieval_database_512.parquet"
    pretrain_dir = repo_root / "datasets" / "pretrain" / "stocks-with-retrieval_512"

    lookback_length = 512
    context_length = 512
    prediction_length = 64
    top_k = 10

    if not stocks_csv.exists():
        raise FileNotFoundError(
            f"{stocks_csv} not found. Run build_stocks_csv.py first to create the aggregated stocks.csv."
        )

    # 1) Build retrieval database parquet
    build_stocks_retrieval_database(
        stocks_csv=stocks_csv,
        output_parquet=retrieval_parquet,
        lookback_length=lookback_length,
    )

    # 2) Build pretrain parquet dataset
    build_stocks_pretrain_parquet(
        stocks_csv=stocks_csv,
        retrieval_parquet=retrieval_parquet,
        output_dir=pretrain_dir,
        context_length=context_length,
        prediction_length=prediction_length,
        top_k=top_k,
    )


if __name__ == "__main__":
    main()

