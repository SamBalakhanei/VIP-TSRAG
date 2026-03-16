import os
from pathlib import Path

import pandas as pd


def main():
    """
    Aggregate all JSONL stock files from ../sampled_stocks/new_directory
    into a single multivariate CSV that TS-RAG can consume, with:

    - first column: date (datetime, sorted ascending)
    - one column per ticker: its close price
    """
    repo_root = Path(__file__).resolve().parent.parent
    stocks_jsonl_dir = repo_root.parent / "sampled_stocks" / "new_directory"
    output_dir = repo_root / "datasets" / "stocks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "stocks.csv"

    if not stocks_jsonl_dir.is_dir():
        raise FileNotFoundError(f"Stocks JSONL directory not found: {stocks_jsonl_dir}")

    jsonl_files = sorted([p for p in stocks_jsonl_dir.iterdir() if p.suffix == ".jsonl"])
    if not jsonl_files:
        raise RuntimeError(f"No *.jsonl files found in {stocks_jsonl_dir}")

    print(f"Found {len(jsonl_files)} JSONL files in {stocks_jsonl_dir}")

    merged_df = None

    for path in jsonl_files:
        ticker = path.stem  # filename without .jsonl
        print(f"Processing {ticker} from {path}")

        df = pd.read_json(path, lines=True)
        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' column not found in {path}")
        if "close" not in df.columns:
            raise ValueError(f"'close' column not found in {path}")

        # convert ms timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["date", "close"]].copy()
        df = df.sort_values("date")
        df = df.rename(columns={"close": ticker})

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="date", how="outer")

    if merged_df is None or merged_df.empty:
        raise RuntimeError("Merged DataFrame is empty after processing all JSONL files.")

    merged_df = merged_df.sort_values("date")
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved aggregated stocks CSV to {output_csv}")


if __name__ == "__main__":
    main()

