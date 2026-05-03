import subprocess
import re
import statistics

SEEDS = [2025, 2026, 2027, 2028, 2029]
PRED_LENS = [96, 192, 336, 720]

MSE_RE = re.compile(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)")

def build_cmd(seed: int, residual_gtr: int, pred_len: int):
    return [
        "python", "-u", "run.py",
        "--is_training", "1",
        "--root_path", "./dataset/",
        "--data_path", "weather.csv",
        "--model_id", f"weather_96_{pred_len}",
        "--model", "GTR",
        "--data", "custom",
        "--features", "M",
        "--seq_len", "96",
        "--pred_len", str(pred_len),
        "--enc_in", "21",
        "--dec_in", "21",
        "--c_out", "21",
        "--cycle", "144",
        "--train_epochs", "30",
        "--patience", "5",
        "--dropout", "0.5",
        "--itr", "1",
        "--batch_size", "64",
        "--learning_rate", "0.001",
        "--num_workers", "0",
        "--random_seed", str(seed),
        "--residual_gtr", str(residual_gtr),
    ]

def run_one(seed: int, residual_gtr: int, pred_len: int):
    cmd = build_cmd(seed, residual_gtr, pred_len)

    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + "\n" + result.stderr

    print(output)

    matches = MSE_RE.findall(output)
    if not matches:
        raise RuntimeError(
            f"Could not find mse/mae in output for seed={seed}, residual_gtr={residual_gtr}, pred_len={pred_len}"
        )

    mse, mae = matches[-1]
    return float(mse), float(mae)

def summarize(name, rows):
    mses = [x["mse"] for x in rows]
    maes = [x["mae"] for x in rows]

    print(f"\n{name}")
    print("-" * len(name))
    for row in rows:
        print(f"seed={row['seed']}  mse={row['mse']:.6f}  mae={row['mae']:.6f}")

    print(f"avg mse = {statistics.mean(mses):.6f}")
    print(f"avg mae = {statistics.mean(maes):.6f}")

def main():
    for pred_len in PRED_LENS:
        print("\n" + "#" * 100)
        print(f"Evaluating Weather, pred_len = {pred_len}")
        print("#" * 100)

        baseline_rows = []
        modified_rows = []

        for seed in SEEDS:
            mse, mae = run_one(seed, residual_gtr=0, pred_len=pred_len)
            baseline_rows.append({"seed": seed, "mse": mse, "mae": mae})

            mse, mae = run_one(seed, residual_gtr=1, pred_len=pred_len)
            modified_rows.append({"seed": seed, "mse": mse, "mae": mae})

        summarize(f"Baseline (residual_gtr=0, pred_len={pred_len})", baseline_rows)
        summarize(f"Modified (residual_gtr=1, pred_len={pred_len})", modified_rows)

        base_mse_avg = statistics.mean(x["mse"] for x in baseline_rows)
        base_mae_avg = statistics.mean(x["mae"] for x in baseline_rows)
        mod_mse_avg = statistics.mean(x["mse"] for x in modified_rows)
        mod_mae_avg = statistics.mean(x["mae"] for x in modified_rows)

        print("\nFinal comparison")
        print("----------------")
        print(f"pred_len: {pred_len}")
        print(f"Baseline avg mse: {base_mse_avg:.6f}")
        print(f"Modified avg mse: {mod_mse_avg:.6f}")
        print(f"MSE improvement: {(base_mse_avg - mod_mse_avg) / base_mse_avg * 100:.3f}%")
        print(f"Baseline avg mae: {base_mae_avg:.6f}")
        print(f"Modified avg mae: {mod_mae_avg:.6f}")
        print(f"MAE improvement: {(base_mae_avg - mod_mae_avg) / base_mae_avg * 100:.3f}%")

if __name__ == "__main__":
    main()