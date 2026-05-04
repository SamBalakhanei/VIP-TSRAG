from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd

from sector_metadata import PRETRAIN_TICKERS

EVAL_TICKERS = PRETRAIN_TICKERS


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n[tsrag-full] {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def is_pretrain_data_compatible(data_dir: Path, expected_len: int) -> bool:
    if not data_dir.is_dir():
        return False
    parquet_files = sorted([f for f in data_dir.iterdir() if f.suffix == ".parquet"])
    if not parquet_files:
        return False
    try:
        df = pd.read_parquet(parquet_files[0])
    except Exception:
        return False
    if "target" not in df.columns:
        return False
    first_target = df.iloc[0]["target"]
    try:
        return len(first_target) == expected_len
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-call full TS-RAG pipeline: build artifacts + pretrain + evaluate."
    )
    parser.add_argument("--stocks_dir", type=str, default="../../sampled_stocks/new_directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--evaluation_steps", type=int, default=10000)
    parser.add_argument("--pretrained_model_path", type=str, default="amazon/chronos-bolt-base")
    parser.add_argument("--model_id", type=str, default="ChronosBoltRetrieve_Stocks_TSRAG_full_fixed10")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--data_path", type=str, default="/content/drive/MyDrive/VIP-TSRAG/VIP-TSRAG-main/TS-RAG-main/TS-RAG/datasets/pretrain/stocks-with-retrieval_full_512")#../datasets/pretrain/stocks-with-retrieval_full_512")
    parser.add_argument("--skip_build_csv", action="store_true")
    parser.add_argument("--skip_artifact_build", action="store_true")
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--artifact_device", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    stocks_csv = repo_root.parent / "datasets" / "stocks" / "stocks.csv"
    pretrain_data_dir = Path(args.data_path)    
    expected_target_len = args.context_length + args.prediction_length
    if args.skip_build_csv and not stocks_csv.exists():
        print(f"\n[tsrag-full] stocks.csv not found at {stocks_csv}; enabling build_stocks_csv.py.")
        args.skip_build_csv = False

    if args.skip_artifact_build:
        if not pretrain_data_dir.is_dir():
            print(f"\n[tsrag-full] Pretraining data directory not found at {pretrain_data_dir}; enabling artifact build.")
            args.skip_artifact_build = False
        elif not is_pretrain_data_compatible(pretrain_data_dir, expected_target_len):
            print(
                f"\n[tsrag-full] Existing pretrain data at {pretrain_data_dir} does not match "
                f"expected target length {expected_target_len}. Enabling artifact build."
            )
            args.skip_artifact_build = False

    if not args.skip_build_csv:
        run_step([sys.executable, "build_stocks_csv.py"], "Building aggregated stocks.csv")
    else:
        print("\n[tsrag-full] Skipping build_stocks_csv.py")

    if not args.skip_artifact_build:
        artifact_device = args.artifact_device if args.artifact_device else args.device
        run_step(
            [
                sys.executable,
                "build_stocks_retrieval_pretrain_full.py",
                "--device",
                artifact_device,
                "--lookback_length",
                str(args.context_length),
                "--prediction_length",
                str(args.prediction_length),
                "--retrieval_database_path",
                "../database/pretrain/stocks_retrieval_database_full_512.parquet",
                "--output_dir",
                args.data_path,
            ],
            "Building full TS-RAG retrieval/pretrain artifacts (fixed 10-stock subset)",
        )
    else:
        print("\n[tsrag-full] Skipping build_stocks_retrieval_pretrain_full.py")

    pretrain_cmd = [
        sys.executable,
        "pretrain.py",
        "--model",
        "ChronosBoltRetrieve",
        "--augment_mode",
        "moe2",
        "--pretrained_model_path",
        args.pretrained_model_path,
        "--retrieval_database_path",
        "../database/pretrain/stocks_retrieval_database_full_512.parquet",
        "--data_path",
        args.data_path,
        "--context_length",
        str(args.context_length),
        "--prediction_length",
        str(args.prediction_length),
        "--top_k",
        str(args.top_k),
        "--batch_size",
        str(args.batch_size),
        "--train_steps",
        str(args.train_steps),
        "--evaluation_steps",
        str(args.evaluation_steps),
        "--checkpoints",
        args.checkpoints,
        "--model_id",
        args.model_id,
    ]
    if not args.skip_pretrain:
        run_step(pretrain_cmd, "Pretraining full TS-RAG")
    else:
        print("\n[tsrag-full] Skipping pretrain.py")

    infer_cmd = [
        sys.executable,
        "financial_infer_tsrag_full.py",
        "--stocks_dir",
        args.stocks_dir,
        "--tickers",
        *EVAL_TICKERS,
        "--model_id",
        f"./checkpoints/{args.model_id}",
        "--retrieval_parquet",
        "../database/pretrain/stocks_retrieval_database_full_512.parquet",
        "--prediction_length",
        str(args.prediction_length),
        "--compute_metrics",
        "--device",
        args.device,
    ]
    run_step(infer_cmd, "Evaluating full TS-RAG on fixed 5-stock subset")

    print("\n[tsrag-full] Pipeline completed successfully.")


if __name__ == "__main__":
    main()

