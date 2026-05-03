import argparse
import subprocess
import sys


PRETRAIN_TICKERS = [
    "OTEX",
    "PATH",
    "PUBM",
    "GOOGL",
    "ADP",
    "CBRE",
    "BLK",
    "FRT",
    "PINS",
    "MKTX",
]

EVAL_TICKERS = [
    "INTA",
    "FFIV",
    "MLM",
    "PARR",
    "SEIC",
]


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n[baseline] {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-call baseline pipeline: pretrain + evaluate.")
    parser.add_argument("--stocks_dir", type=str, default="../../sampled_stocks/new_directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrained_model_path", type=str, default="amazon/chronos-bolt-base")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/financial_finetuned_fixed10")
    args = parser.parse_args()

    pretrain_cmd = [
        sys.executable,
        "financial_pretrain.py",
        "--stocks_dir",
        args.stocks_dir,
        "--tickers",
        *PRETRAIN_TICKERS,
        "--pretrained_model_path",
        args.pretrained_model_path,
        "--save_dir",
        args.save_dir,
        "--seq_len",
        str(args.seq_len),
        "--pred_len",
        str(args.prediction_length),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
    ]
    run_step(pretrain_cmd, "Pretraining baseline on fixed 10-stock subset")

    infer_cmd = [
        sys.executable,
        "financial_infer.py",
        "--stocks_dir",
        args.stocks_dir,
        "--tickers",
        *EVAL_TICKERS,
        "--model_id",
        args.save_dir,
        "--prediction_length",
        str(args.prediction_length),
        "--compute_metrics",
        "--device",
        args.device,
    ]
    run_step(infer_cmd, "Evaluating baseline on fixed 5-stock subset")

    print("\n[baseline] Pipeline completed successfully.")


if __name__ == "__main__":
    main()

