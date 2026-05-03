import argparse
import subprocess
import sys


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n[preprocess] {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all data preprocessing steps once.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    parser.add_argument("--sector_bonus", type=float, default=0.10)
    parser.add_argument("--naics_bonus", type=float, default=0.15)
    args = parser.parse_args()

    # Build aggregated stocks.csv
    run_step([sys.executable, "build_stocks_csv.py"], "Building aggregated stocks.csv")

    # Build light TS-RAG retrieval/pretrain artifacts
    run_step([sys.executable, "build_stocks_retrieval_pretrain.py"], "Building light TS-RAG retrieval/pretrain artifacts")

    # Build full TS-RAG retrieval/pretrain artifacts
    run_step(
        [sys.executable, "build_stocks_retrieval_pretrain_full.py", "--device", args.device],
        "Building full TS-RAG retrieval/pretrain artifacts"
    )

    # Build sector-aware full TS-RAG artifacts
    run_step(
        [
            sys.executable,
            "build_stocks_retrieval_pretrain_full_sector.py",
            "--device",
            args.device,
            "--top_k",
            str(args.top_k),
            "--candidate_multiplier",
            str(args.candidate_multiplier),
            "--sector_bonus",
            str(args.sector_bonus),
            "--naics_bonus",
            str(args.naics_bonus),
        ],
        "Building sector-aware full TS-RAG artifacts"
    )

    print("\n[preprocess] All preprocessing completed successfully.")


if __name__ == "__main__":
    main()