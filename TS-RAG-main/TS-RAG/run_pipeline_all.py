import argparse
import subprocess
import sys


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n[all] {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline + light TS-RAG + full TS-RAG with reduced redundancy."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stocks_dir", type=str, default="../../sampled_stocks/new_directory")
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--run_light", action="store_true")
    parser.add_argument("--run_full", action="store_true")
    args = parser.parse_args()

    run_baseline = args.run_baseline or (not args.run_baseline and not args.run_light and not args.run_full)
    run_light = args.run_light or (not args.run_baseline and not args.run_light and not args.run_full)
    run_full = args.run_full or (not args.run_baseline and not args.run_light and not args.run_full)

    # Shared once for TS-RAG variants
    if run_light or run_full:
        run_step([sys.executable, "build_stocks_csv.py"], "Building aggregated stocks.csv once")

    if run_baseline:
        run_step(
            [
                sys.executable,
                "run_pipeline_baseline.py",
                "--stocks_dir",
                args.stocks_dir,
                "--device",
                args.device,
                "--prediction_length",
                str(args.prediction_length),
            ],
            "Running baseline pipeline",
        )

    if run_light:
        run_step(
            [
                sys.executable,
                "run_pipeline_tsrag_light.py",
                "--stocks_dir",
                args.stocks_dir,
                "--device",
                args.device,
                "--prediction_length",
                str(args.prediction_length),
                "--skip_build_csv",
            ],
            "Running light TS-RAG pipeline without redundant CSV build",
        )

    if run_full:
        run_step(
            [
                sys.executable,
                "run_pipeline_tsrag_full.py",
                "--stocks_dir",
                args.stocks_dir,
                "--device",
                args.device,
                "--prediction_length",
                str(args.prediction_length),
                "--skip_build_csv",
            ],
            "Running full TS-RAG pipeline without redundant CSV build",
        )

    print("\n[all] Completed selected pipelines.")


if __name__ == "__main__":
    main()

