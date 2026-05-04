import sqlite3
from pathlib import Path
from datetime import datetime
import json

DB_PATH = Path("data/benchmark.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.executescript("""
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS tasks;
DROP TABLE IF EXISTS runs;
DROP TABLE IF EXISTS leaderboard_entries;
DROP TABLE IF EXISTS run_metrics;

CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    model_name TEXT,
    family TEXT,
    description TEXT
);

CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    task_name TEXT,
    horizon INTEGER,
    lookback INTEGER,
    sector TEXT
);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    model_id TEXT,
    task_id TEXT,
    status TEXT,
    created_at TEXT,
    config_json TEXT
);

CREATE TABLE run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    ticker TEXT,
    metric_name TEXT,
    metric_value REAL
);

CREATE TABLE leaderboard_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT,
    task_id TEXT,
    metric_name TEXT,
    mean_score REAL,
    num_runs INTEGER,
    num_series INTEGER
);
""")

models = [
    ("gtr_original", "GTR Original / haware0", "GTR", "Original GTR run on 500 stocks"),
    ("gtr_modified", "GTR Modified / haware1", "GTR", "Modified GTR run on 500 stocks"),
    ("raf_baseline", "RAF Baseline", "RAF", "RAF baseline run on 500 stocks"),
    ("raf_modified", "RAF Modified", "RAF", "RAF retrieval-augmented run on 500 stocks"),
    ("tsrag_placeholder", "TS-RAG", "TS-RAG", "Pending results"),
]

cursor.executemany("INSERT INTO models VALUES (?, ?, ?, ?)", models)

tasks = [
    ("stocks_500", "500 Stocks Benchmark", None, None, "All"),
]

cursor.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?)", tasks)

results = [
    {
        "run_id": "run_gtr_original",
        "model_id": "gtr_original",
        "metrics": {
            "MSE": 0.3689919,
            "MAE": 0.3917737,
        },
        "config": {"variant": "haware0", "num_stocks": 500},
    },
    {
        "run_id": "run_gtr_modified",
        "model_id": "gtr_modified",
        "metrics": {
            "MSE": 0.3701031,
            "MAE": 0.3913500,
        },
        "config": {"variant": "haware1", "num_stocks": 500},
    },
    {
        "run_id": "run_raf_baseline",
        "model_id": "raf_baseline",
        "metrics": {
            "MASE": 3.773072,
            "WQL": 0.083917,
        },
        "config": {"variant": "baseline", "num_stocks": 500},
    },
    {
        "run_id": "run_raf_modified",
        "model_id": "raf_modified",
        "metrics": {
            "MASE": 3.539574,
            "WQL": 0.077464,
        },
        "config": {"variant": "retrieval_augmented", "num_stocks": 500},
    },
]

for result in results:
    cursor.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
        (
            result["run_id"],
            result["model_id"],
            "stocks_500",
            "completed",
            datetime.utcnow().isoformat(),
            json.dumps(result["config"]),
        ),
    )

    for metric_name, metric_value in result["metrics"].items():
        cursor.execute(
            """
            INSERT INTO leaderboard_entries
            (model_id, task_id, metric_name, mean_score, num_runs, num_series)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result["model_id"],
                "stocks_500",
                metric_name,
                metric_value,
                1,
                500,
            ),
        )

conn.commit()
conn.close()

print("Real benchmark results inserted. TS-RAG left blank.")
