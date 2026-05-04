import subprocess
import re
import statistics

SEEDS = [2025, 2026, 2027, 2028, 2029]
PRED_LENS = [96, 192, 336, 720]

PERIOD_SETS = [
    "24,168",
    "12,24,168",
    "12,24,48,168",
]

MSE_RE = re.compile(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)")

def build_cmd(seed: int, pred_len: int, multi_period_resgtr: int, periods: str):
    cmd = [
        "python", "-u", "run.py",
        "--is_training", "1",
        "--root_path", "./dataset/",
        "--data_path", "ETTh1.csv",
        "--model_id", f"ETTh1_96_{pred_len}",
        "--model", "GTR",
        "--data", "ETTh1",
        "--features", "M",
        "--seq_len", "96",
        "--pred_len", str(pred_len),
        "--enc_in", "7",
        "--cycle", "24",
        "--train_epochs", "30",
        "--patience", "5",
        "--dropout", "0.5",
        "--itr", "1",
        "--batch_size", "256",
        "--learning_rate", "0.001",
        "--num_workers", "0",
        "--random_seed", str(seed),
        "--multi_period_resgtr", str(multi_period_resgtr),
    ]

    if multi_period_resgtr == 1:
        cmd += ["--periods", periods]

    return cmd

def run_one(seed: int, pred_len: int, multi_period_resgtr: int, periods: str):
    cmd = build_cmd(seed, pred_len, multi_period_resgtr, periods)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + "\n" + result.stderr

    matches = MSE_RE.findall(output)
    if not matches:
        raise RuntimeError(
            f"Could not find mse/mae in output for seed={seed}, pred_len={pred_len}, "
            f"multi_period_resgtr={multi_period_resgtr}, periods={periods}\n\n"
            f"Full output:\n{output}"
        )

    mse, mae = matches[-1]
    return float(mse), float(mae)

def summarize_rows(name, rows):
    mses = [r["mse"] for r in rows]
    maes = [r["mae"] for r in rows]

    lines = []
    lines.append(f"\n{name}")
    lines.append("-" * len(name))
    for r in rows:
        lines.append(f"seed={r['seed']}  mse={r['mse']:.6f}  mae={r['mae']:.6f}")

    lines.append(f"avg mse = {statistics.mean(mses):.6f}")
    lines.append(f"avg mae = {statistics.mean(maes):.6f}")
    return "\n".join(lines)

def main():
    all_results = {}

    total_runs = len(PRED_LENS) * (1 + len(PERIOD_SETS)) * len(SEEDS)
    done_runs = 0

    for pred_len in PRED_LENS:
        all_results[pred_len] = {}

        baseline_rows = []
        for seed in SEEDS:
            mse, mae = run_one(
                seed=seed,
                pred_len=pred_len,
                multi_period_resgtr=0,
                periods=""
            )
            baseline_rows.append({"seed": seed, "mse": mse, "mae": mae})

            done_runs += 1
            print(
                f"[{done_runs}/{total_runs}] Done baseline | pred_len={pred_len} | seed={seed} | "
                f"mse={mse:.6f} | mae={mae:.6f}",
                flush=True
            )

        all_results[pred_len]["baseline"] = baseline_rows

        for period_set in PERIOD_SETS:
            rows = []
            for seed in SEEDS:
                mse, mae = run_one(
                    seed=seed,
                    pred_len=pred_len,
                    multi_period_resgtr=1,
                    periods=period_set
                )
                rows.append({"seed": seed, "mse": mse, "mae": mae})

                done_runs += 1
                print(
                    f"[{done_runs}/{total_runs}] Done periods={period_set} | pred_len={pred_len} | "
                    f"seed={seed} | mse={mse:.6f} | mae={mae:.6f}",
                    flush=True
                )

            all_results[pred_len][period_set] = rows

    # print everything only at the end
    for pred_len in PRED_LENS:
        print("\n" + "#" * 100)
        print(f"Evaluating pred_len = {pred_len}")
        print("#" * 100)

        baseline_rows = all_results[pred_len]["baseline"]
        print(summarize_rows(f"Baseline (pred_len={pred_len})", baseline_rows))

        baseline_mse_avg = statistics.mean(r["mse"] for r in baseline_rows)
        baseline_mae_avg = statistics.mean(r["mae"] for r in baseline_rows)

        for period_set in PERIOD_SETS:
            rows = all_results[pred_len][period_set]
            print(summarize_rows(
                f"Multi-period residualized (periods={period_set}, pred_len={pred_len})",
                rows
            ))

            avg_mse = statistics.mean(r["mse"] for r in rows)
            avg_mae = statistics.mean(r["mae"] for r in rows)

            print("\nComparison vs baseline")
            print("----------------------")
            print(f"periods: {period_set}")
            print(f"pred_len: {pred_len}")
            print(f"Baseline avg mse: {baseline_mse_avg:.6f}")
            print(f"Modified avg mse: {avg_mse:.6f}")
            print(f"MSE improvement: {(baseline_mse_avg - avg_mse) / baseline_mse_avg * 100:.3f}%")
            print(f"Baseline avg mae: {baseline_mae_avg:.6f}")
            print(f"Modified avg mae: {avg_mae:.6f}")
            print(f"MAE improvement: {(baseline_mae_avg - avg_mae) / baseline_mae_avg * 100:.3f}%")

if __name__ == "__main__":
    main()