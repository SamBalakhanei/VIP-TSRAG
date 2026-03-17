Full TS‑RAG Pipeline (Stocks, Retrieval at Inference)
=====================================================

This is a **third approach** that mirrors the original TS‑RAG idea more closely:

- Uses a dedicated Chronos embedding model (`amazon/chronos-t5-base`) to build a retrieval database.
- Pretrains `ChronosBoltRetrieve` (retrieval‑augmented Chronos‑Bolt) on stocks with retrieval information.
- **Performs retrieval at inference time** and feeds retrieved sequences into the model.

1) Build aggregated CSV and full TS‑RAG artifacts
-------------------------------------------------

```bash
cd TS-RAG-main/TS-RAG

# Build multivariate stocks CSV from JSONL files
python build_stocks_csv.py

# Build retrieval DB + pretrain parquet using Chronos embeddings
python build_stocks_retrieval_pretrain_full.py
```

This will create:

- `../database/pretrain/stocks_retrieval_database_full_512.parquet`
- `datasets/pretrain/stocks-with-retrieval_full_512/stocks_pretrain_full.parquet`

2) Pretrain full TS‑RAG on stocks
---------------------------------

Use the existing `pretrain.py`, pointing it to the *full* artifacts:

```bash
python pretrain.py \
  --model ChronosBoltRetrieve \
  --augment_mode moe2 \
  --pretrained_model_path "amazon/chronos-bolt-base" \
  --retrieval_database_path "../database/pretrain/stocks_retrieval_database_full_512.parquet" \
  --data_path "../datasets/pretrain/stocks-with-retrieval_full_512" \
  --context_length 512 \
  --prediction_length 64 \
  --top_k 10 \
  --batch_size 256 \
  --train_steps 200000 \
  --evaluation_steps 10000 \
  --checkpoints "./checkpoints/" \
  --model_id "ChronosBoltRetrieve_Stocks_TSRAG_full"
```

This trains a TS‑RAG model that uses retrieval during training, with embeddings from `amazon/chronos-t5-base`.

3) Evaluate full TS‑RAG (with retrieval at inference)
-----------------------------------------------------

```bash
# Random 5 stocks:
python financial_infer_tsrag_full.py \
  --stocks_dir "../../sampled_stocks/new_directory" \
  --num_stocks 5 \
  --model_id "./checkpoints/ChronosBoltRetrieve_Stocks_TSRAG_full" \
  --retrieval_parquet "../database/pretrain/stocks_retrieval_database_full_512.parquet" \
  --prediction_length 64 \
  --compute_metrics \
  --device "cpu"

# Same 5 stocks as baseline (replace tickers with any 5 that exist in sampled_stocks/new_directory):
python financial_infer_tsrag_full.py \
  --stocks_dir "../../sampled_stocks/new_directory" \
  --tickers AAT PUBM QQQA MLM GOOGL \
  --model_id "./checkpoints/ChronosBoltRetrieve_Stocks_TSRAG_full" \
  --retrieval_parquet "../database/pretrain/stocks_retrieval_database_full_512.parquet" \
  --prediction_length 64 \
  --compute_metrics \
  --device "cpu"
```

This script:

- Loads the **same retrieval DB** used for pretraining.
- Uses `amazon/chronos-t5-base` to embed the **test context** and query FAISS.
- Constructs `retrieved_seq` and `distances` for each test example.
- Runs `ChronosBoltRetrieve` via `ChronosBoltPipelineWithRetrieval` to get quantile forecasts.
- Computes metrics (MAE, MSE, RMSE, MAPE, MSPE, SMAPE, ND) on the last 64 points.

You can now compare three approaches on exactly the same set of stocks:

1. **Baseline**: Fine‑tuned Chronos‑Bolt (`README_baseline.md`).
2. **Light TS‑RAG**: Stocks‑only retrieval, no retrieval at inference (`README_tsrag.md`).
3. **Full TS‑RAG**: Chronos embedding model + retrieval at inference (this README). 

