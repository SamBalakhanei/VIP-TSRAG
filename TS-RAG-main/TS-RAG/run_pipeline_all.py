import argparse
import subprocess
import sys


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n[all] {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run data preprocessing once, then all model pipelines: baseline + light TS-RAG + full TS-RAG + sector TS-RAG."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stocks_dir", type=str, default="../../sampled_stocks/new_directory")
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--run_preprocess", action="store_true")
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--run_light", action="store_true")
    parser.add_argument("--run_full", action="store_true")
    parser.add_argument("--run_sector", action="store_true")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    parser.add_argument("--sector_bonus", type=float, default=0.10)
    parser.add_argument("--naics_bonus", type=float, default=0.15)
    args = parser.parse_args()

    run_preprocess = args.run_preprocess or (not args.run_preprocess and not args.run_baseline and not args.run_light and not args.run_full and not args.run_sector)
    run_baseline = args.run_baseline or (not args.run_preprocess and not args.run_baseline and not args.run_light and not args.run_full and not args.run_sector)
    run_light = args.run_light or (not args.run_preprocess and not args.run_baseline and not args.run_light and not args.run_full and not args.run_sector)
    run_full = args.run_full or (not args.run_preprocess and not args.run_baseline and not args.run_light and not args.run_full and not args.run_sector)
    run_sector = args.run_sector or (not args.run_preprocess and not args.run_baseline and not args.run_light and not args.run_full and not args.run_sector)

    if run_preprocess:
        run_step(
            [
                sys.executable,
                "preprocess_data.py",
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
            "Running data preprocessing once",
        )

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
                "--skip_artifact_build",
            ],
            "Running light TS-RAG pipeline",
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
                "--skip_artifact_build",
            ],
            "Running full TS-RAG pipeline",
        )

    if run_sector:
        run_step(
            [
                sys.executable,
                "run_pipeline_tsrag_full_sector.py",
                "--stocks_dir",
                args.stocks_dir,
                "--device",
                args.device,
                "--prediction_length",
                str(args.prediction_length),
                "--skip_build_csv",
                "--skip_artifact_build",
            ],
            "Running sector TS-RAG pipeline",
        )

    print("\n[all] Completed selected pipelines.")


if __name__ == "__main__":
    main()

