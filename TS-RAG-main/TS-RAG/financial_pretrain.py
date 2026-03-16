import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig

from models.ChronosBolt import ChronosBoltModelForForecasting
from utils.metrics import metric as metric_fn


class StockWindowDataset(Dataset):
    """
    Simple sliding-window dataset for univariate stock close prices.

    Expects a 1D numpy array of close prices for a single ticker,
    already sorted by time and (optionally) scaled.
    """

    def __init__(
        self,
        series: np.ndarray,
        seq_len: int,
        pred_len: int,
    ) -> None:
        super().__init__()
        assert series.ndim == 1, "series must be 1D array of close prices"
        self.series = series.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self._length = max(0, len(self.series) - self.seq_len - self.pred_len + 1)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        mid = start + self.seq_len
        end = mid + self.pred_len
        x = self.series[start:mid]
        y = self.series[mid:end]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_stock_series_from_jsonl(
    directory: str,
    tickers: List[str],
    time_col: str = "date",
    price_col: str = "close",
) -> Dict[str, np.ndarray]:
    """
    Load close price time series for the given tickers from JSONL files.

    Each file is expected to be named <TICKER>.jsonl under `directory`
    and to contain at least a timestamp column (`time_col`) and a
    close-price column (`price_col`).
    """
    series_dict: Dict[str, np.ndarray] = {}

    for ticker in tickers:
        path = os.path.join(directory, f"{ticker}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL file for ticker {ticker} not found at {path}")

        df = pd.read_json(path, lines=True)
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in {path}")
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in {path}")

        df = df.sort_values(time_col)
        series = df[price_col].astype(float).to_numpy()
        series_dict[ticker] = series

    return series_dict


def build_datasets(
    series_dict: Dict[str, np.ndarray],
    seq_len: int,
    pred_len: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[Dataset, Dataset, Dict[str, Dataset]]:
    """
    Split each ticker's series into train/val/test and return:
    - a combined train dataset over all tickers
    - a combined val dataset over all tickers
    - a dict of per-ticker test datasets for metric reporting
    """
    train_datasets: List[Dataset] = []
    val_datasets: List[Dataset] = []
    test_datasets_by_ticker: Dict[str, Dataset] = {}

    for ticker, series in series_dict.items():
        n = len(series)
        if n < seq_len + pred_len + 1:
            # skip very short series
            print(
                f"Skipping {ticker}: length {n} < required {seq_len + pred_len + 1} "
                f"(seq_len={seq_len}, pred_len={pred_len})"
            )
            continue

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = max(1, n - n_train - n_val)

        train_series = series[:n_train]
        val_series = series[n_train : n_train + n_val]
        test_series = series[n_train + n_val :]

        train_datasets.append(StockWindowDataset(train_series, seq_len, pred_len))
        if len(val_series) >= seq_len + pred_len + 1:
            val_datasets.append(StockWindowDataset(val_series, seq_len, pred_len))
        if len(test_series) >= seq_len + pred_len + 1:
            test_datasets_by_ticker[ticker] = StockWindowDataset(test_series, seq_len, pred_len)

    # simple concatenation via torch.utils.data.ConcatDataset to keep code minimal
    from torch.utils.data import ConcatDataset

    if not train_datasets:
        raise RuntimeError(
            f"No usable stock series for training. Each series must have length >= seq_len + pred_len + 1 "
            f"(seq_len={seq_len}, pred_len={pred_len}). Try reducing seq_len/pred_len or selecting tickers "
            f"with longer histories."
        )

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None

    return train_dataset, val_dataset, test_datasets_by_ticker


def train_one_epoch(
    model: ChronosBoltModelForForecasting,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(context=batch_x, target=batch_y)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_loss(
    model: ChronosBoltModelForForecasting,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(context=batch_x, target=batch_y)
        loss = outputs.loss
        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_test_metrics(
    model: ChronosBoltModelForForecasting,
    datasets_by_ticker: Dict[str, Dataset],
    seq_len: int,
    pred_len: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per ticker using central (0.5) quantile forecast.
    """
    model.eval()
    all_results: Dict[str, Dict[str, float]] = {}

    # central quantile index (assumes 0.5 is present)
    quantiles = model.chronos_config.quantiles
    if 0.5 in quantiles:
        q_idx = quantiles.index(0.5)
    else:
        q_idx = len(quantiles) // 2

    for ticker, dataset in datasets_by_ticker.items():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(context=batch_x)
            # outputs.quantile_preds: [B, num_q, pred_len]
            q_preds = outputs.quantile_preds[:, q_idx, :].cpu().numpy()
            preds.append(q_preds.reshape(-1))
            trues.append(batch_y.numpy().reshape(-1))

        if preds:
            pred_arr = np.concatenate(preds, axis=0)
            true_arr = np.concatenate(trues, axis=0)
            mae, mse, rmse, mape, mspe, smape, nd = metric_fn(pred_arr, true_arr)
            all_results[ticker] = {
                "MAE": float(mae),
                "MSE": float(mse),
                "RMSE": float(rmse),
                "MAPE": float(mape),
                "MSPE": float(mspe),
                "SMAPE": float(smape),
                "ND": float(nd),
            }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Pretrain Chronos/TS-RAG on financial stock JSONL data")
    parser.add_argument("--stocks_dir", type=str, default="../sampled_stocks/new_directory")
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="*",
        default=None,
        help="List of tickers to use (without .jsonl). If not provided, the script will pick the first 5 *.jsonl files.",
    )
    parser.add_argument("--num_stocks", type=int, default=5, help="Minimum number of stocks to use")
    # default to Polygon-style millisecond epoch column name
    parser.add_argument("--time_col", type=str, default="timestamp")
    parser.add_argument("--price_col", type=str, default="close")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=64)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pretrained_model_path", type=str, default="amazon/chronos-bolt-base")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints/financial_finetuned",
    )
    parser.add_argument("--gpu_loc", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")

    # determine tickers
    if args.tickers is None or len(args.tickers) == 0:
        all_files = [
            f for f in os.listdir(args.stocks_dir) if f.endswith(".jsonl")
        ]
        all_files.sort()
        if len(all_files) < args.num_stocks:
            raise RuntimeError(
                f"Requested at least {args.num_stocks} stocks, but only found {len(all_files)} JSONL files in {args.stocks_dir}"
            )
        tickers = [os.path.splitext(f)[0] for f in all_files[: args.num_stocks]]
    else:
        tickers = args.tickers

    print(f"Using tickers: {tickers}")

    # load series
    series_dict = load_stock_series_from_jsonl(
        directory=args.stocks_dir,
        tickers=tickers,
        time_col=args.time_col,
        price_col=args.price_col,
    )

    # simple per-ticker standardization to zero-mean unit-std
    for t in list(series_dict.keys()):
        s = series_dict[t]
        mean = s.mean()
        std = s.std()
        if std == 0:
            std = 1.0
        series_dict[t] = (s - mean) / std

    train_dataset, val_dataset, test_datasets_by_ticker = build_datasets(
        series_dict,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    # load Chronos-Bolt base model (can be HF model id or local dir)
    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    model = ChronosBoltModelForForecasting.from_pretrained(args.pretrained_model_path, config=config)

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Starting pretraining on financial time series...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer)
        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, device)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}")

    # evaluate on per-ticker test splits
    print("Evaluating per-ticker metrics on held-out data...")
    results = evaluate_test_metrics(
        model=model,
        datasets_by_ticker=test_datasets_by_ticker,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        device=device,
    )

    for ticker, metrics in results.items():
        print(f"Ticker: {ticker}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

    # save fine-tuned model and config as a HF-style directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving fine-tuned model to {args.save_dir}")
    model.save_pretrained(args.save_dir)
    config.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()

