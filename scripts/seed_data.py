import sqlite3
from pathlib import Path
from datetime import datetime
import random
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
    ("tsrag_1", "TS-RAG", "TS-RAG", "Retrieval model"),
    ("gtr_1", "GTR", "GTR", "Temporal retrieval"),
    ("raf_1", "RAF", "RAF", "Augmented forecasting"),
]

cursor.executemany("INSERT INTO models VALUES (?, ?, ?, ?)", models)


tasks = [
    ("task_1", "Tech H16 L64", 16, 64, "Tech"),
    ("task_2", "Tech H16 L128", 16, 128, "Tech"),
    ("task_3", "Tech H32 L64", 32, 64, "Tech"),
    ("task_4", "Tech H32 L128", 32, 128, "Tech"),
    ("task_5", "Finance H16 L64", 16, 64, "Finance"),
    ("task_6", "Finance H16 L128", 16, 128, "Finance"),
    ("task_7", "Healthcare H32 L64", 32, 64, "Healthcare"),
    ("task_8", "Healthcare H32 L128", 32, 128, "Healthcare"),
]

cursor.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?)", tasks)


tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]


run_id_counter = 1

for model_id, model_name, _, _ in models:
    for task_id, task_name, horizon, lookback, sector in tasks:

        base = random.uniform(0.8, 1.5)

        for i in range(3): 
            run_id = f"run_{run_id_counter}"
            run_id_counter += 1

            config = {
                "horizon": horizon,
                "lookback": lookback,
                "sector": sector
            }

            cursor.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    model_id,
                    task_id,
                    "completed",
                    datetime.utcnow().isoformat(),
                    json.dumps(config)
                )
            )

            for ticker in tickers:
                mae = base + random.uniform(-0.1, 0.1)
                mase = base + random.uniform(-0.1, 0.1)
                rmse = base + random.uniform(-0.2, 0.2)

                cursor.execute(
                    "INSERT INTO run_metrics (run_id, ticker, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                    (run_id, ticker, "MAE", mae)
                )
                cursor.execute(
                    "INSERT INTO run_metrics VALUES (NULL, ?, ?, ?, ?)",
                    (run_id, ticker, "MASE", mase)
                )
                cursor.execute(
                    "INSERT INTO run_metrics VALUES (NULL, ?, ?, ?, ?)",
                    (run_id, ticker, "RMSE", rmse)
                )

        for metric in ["MAE", "MASE", "RMSE"]:
            avg_score = base + random.uniform(-0.05, 0.05)

            cursor.execute(
                "INSERT INTO leaderboard_entries (model_id, task_id, metric_name, mean_score, num_runs, num_series) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    model_id,
                    task_id,
                    metric,
                    avg_score,
                    3,
                    len(tickers)
                )
            )

rows = conn.execute("PRAGMA table_info(tasks)").fetchall()
print(rows)

conn.commit()
    
conn.close()



print("dummy data created!")