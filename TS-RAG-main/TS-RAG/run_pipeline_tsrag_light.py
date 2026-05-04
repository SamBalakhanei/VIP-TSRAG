import argparse
import subprocess
import sys

from sector_metadata import PRETRAIN_TICKERS

EVAL_TICKERS = PRETRAIN_TICKERS


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n[tsrag-light] {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-call light TS-RAG pipeline: build artifacts + pretrain + evaluate."
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
    parser.add_argument("--model_id", type=str, default="ChronosBoltRetrieve_Stocks_TSRAG_fixed10")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--skip_build_csv", action="store_true")
    parser.add_argument("--skip_artifact_build", action="store_true")
    parser.add_argument("--skip_pretrain", action="store_true")
    args = parser.parse_args()

    if not args.skip_build_csv:
        run_step([sys.executable, "build_stocks_csv.py"], "Building aggregated stocks.csv")
    else:
        print("\n[tsrag-light] Skipping build_stocks_csv.py")

    if not args.skip_artifact_build:
        run_step(
            [sys.executable, "build_stocks_retrieval_pretrain.py"],
            "Building light TS-RAG retrieval/pretrain artifacts (fixed 10-stock subset)",
        )
    else:
        print("\n[tsrag-light] Skipping build_stocks_retrieval_pretrain.py")

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
        "../database/pretrain/stocks_retrieval_database_512.parquet",
        "--data_path",
        "../datasets/pretrain/stocks-with-retrieval_512",
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
        run_step(pretrain_cmd, "Pretraining light TS-RAG")
    else:
        print("\n[tsrag-light] Skipping pretrain.py")

    infer_cmd = [
        sys.executable,
        "financial_infer_tsrag.py",
        "--stocks_dir",
        args.stocks_dir,
        "--tickers",
        *EVAL_TICKERS,
        "--model_id",
        f"./checkpoints/{args.model_id}",
        "--prediction_length",
        str(args.prediction_length),
        "--compute_metrics",
        "--device",
        args.device,
    ]
    run_step(infer_cmd, "Evaluating light TS-RAG on fixed 5-stock subset")

    print("\n[tsrag-light] Pipeline completed successfully.")


if __name__ == "__main__":
    main()

