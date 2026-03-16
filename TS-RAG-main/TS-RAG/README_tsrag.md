TS-RAG Pipeline
1) Build aggregated CSV and TS‑RAG pretrain artifacts:
```
python build_stocks_csv.py
python build_stocks_retrieval_pretrain.py
```

2) Pretrain TS‑RAG:
```
python pretrain.py --model ChronosBoltRetrieve --augment_mode moe2 --pretrained_model_path "amazon/chronos-bolt-base" --retrieval_database_path "../database/pretrain/stocks_retrieval_database_512.parquet" --data_path "../datasets/pretrain/stocks-with-retrieval_512" --context_length 512 --prediction_length 64 --top_k 10 --batch_size 256 --train_steps 200000 --evaluation_steps 10000 --checkpoints "./checkpoints/" --model_id "ChronosBoltRetrieve_Stocks_TSRAG"
```

3) Evaluate TS‑RAG:
```
python financial_infer_tsrag.py --stocks_dir "../../sampled_stocks/new_directory" --num_stocks 5 --model_id "./checkpoints/ChronosBoltRetrieve_Stocks_TSRAG" --prediction_length 64 --compute_metrics --device "cpu"
```