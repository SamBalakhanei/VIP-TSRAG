import argparse
import subprocess
import sys

from sector_metadata import PRETRAIN_TICKERS

EVAL_TICKERS = PRETRAIN_TICKERS


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n[baseline] {label}")
    print(" ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-call baseline pipeline: pretrain + evaluate.")
    parser.add_argument("--stocks_dir", type=str, default="../../sampled_stocks/new_directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrained_model_path", type=str, default="amazon/chronos-bolt-base")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/financial_finetuned_fixed5")
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
    run_step(pretrain_cmd, "Pretraining baseline on fixed 5-stock subset")

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

