import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from utils.metrics import metric as metric_fn


def load_close_series(jsonl_path: str, time_col: str = "timestamp", price_col: str = "close") -> np.ndarray:
    df = pd.read_json(jsonl_path, lines=True)
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in {jsonl_path}")
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in {jsonl_path}")
    df = df.sort_values(time_col)
    return df[price_col].astype(float).to_numpy()


def run_forecast_for_ticker(pipeline, stocks_dir, ticker, time_col, price_col, prediction_length, compute_metrics: bool):
    jsonl_path = os.path.join(stocks_dir, f"{ticker}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[WARN] File not found for ticker {ticker}: {jsonl_path}")
        return

    series = load_close_series(jsonl_path, time_col=time_col, price_col=price_col)

    print(f"\n=== {ticker} ===")
    print(f"Loaded {len(series)} points from {jsonl_path}")

    # use all but last prediction_length points as context if computing metrics
    if compute_metrics and len(series) > prediction_length:
        context_arr = series[:-prediction_length]
        true_future = series[-prediction_length:]
    else:
        context_arr = series
        true_future = None

    context = torch.tensor(context_arr, dtype=torch.float32)

    forecast = pipeline.predict(context=context, prediction_length=prediction_length)

    quantiles = pipeline.quantiles
    if 0.5 in quantiles:
        q_idx = quantiles.index(0.5)
    else:
        q_idx = len(quantiles) // 2

    median_forecast = forecast[0, q_idx].cpu().numpy()

    print(f"Prediction length: {prediction_length}")
    print(f"Quantiles: {quantiles}")
    print(f"Median forecast:")
    print(median_forecast)

    if compute_metrics and true_future is not None:
        pred_arr = median_forecast.reshape(-1)
        true_arr = true_future.reshape(-1)
        mae, mse, rmse, mape, mspe, smape, nd = metric_fn(pred_arr, true_arr)
        print("Metrics (using last prediction_length points as ground truth):")
        print(f"  MAE:   {mae:.6f}")
        print(f"  MSE:   {mse:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAPE:  {mape:.6f}")
        print(f"  MSPE:  {mspe:.6f}")
        print(f"  SMAPE: {smape:.6f}")
        print(f"  ND:    {nd:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Run Chronos-Bolt inference on financial JSONL stock data.")
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
    parser.add_argument("--model_id", type=str, default="./checkpoints/financial_finetuned")
    parser.add_argument("--device", type=str, default="cpu", help='Use "cpu", "cuda", or "mps".')
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="If set, use the last prediction_length points as ground truth to compute error metrics.",
    )

    args = parser.parse_args()

    # determine tickers
    if args.tickers is None or len(args.tickers) == 0:
        all_files = [f for f in os.listdir(args.stocks_dir) if f.endswith(".jsonl")]
        if not all_files:
            raise RuntimeError(f"No *.jsonl files found in {args.stocks_dir}")
        random.shuffle(all_files)
        selected = all_files[: args.num_stocks]
        tickers = [os.path.splitext(f)[0] for f in selected]
    else:
        tickers = args.tickers

    print(f"Tickers to run: {tickers}")

    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id,
        device_map=args.device,
        torch_dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
    )

    for t in tickers:
        run_forecast_for_ticker(
            pipeline=pipeline,
            stocks_dir=args.stocks_dir,
            ticker=t,
            time_col=args.time_col,
            price_col=args.price_col,
            prediction_length=args.prediction_length,
            compute_metrics=args.compute_metrics,
        )


if __name__ == "__main__":
    main()


