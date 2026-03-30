import subprocess
import os
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GTR_DIR = REPO_ROOT / "GTR"

# Replace these with a real command that already works in your repo
cmd = [
    "python",
    "run.py",
    "--is_training", "1",
    "--model", "GTR",
    "--model_id", "gtr_test_run",
    "--data", "custom",
    "--root_path", "../sampled_stocks/new_directory/",
    "--data_path", "MSFT.csv",
    "--features", "S",
    "--target", "OT",
    "--seq_len", "64",
    "--label_len", "32",
    "--pred_len", "16",
    "--enc_in", "1",
    "--dec_in", "1",
    "--c_out", "1",
]

result = subprocess.run(
    cmd,
    cwd=GTR_DIR,
    capture_output=True,
    text=True
)

print("RETURN CODE:", result.returncode)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

results_dir = GTR_DIR / "results"
metrics_files = list(results_dir.rglob("metrics.json"))

if not metrics_files:
    print("No metrics.json found.")
else:
    latest = max(metrics_files, key=lambda p: p.stat().st_mtime)
    print("Found metrics file:", latest)
    with open(latest, "r") as f:
        metrics = json.load(f)
    print("Metrics:", metrics)