Fine-Tuned Chronos-Bolt + Evaluation Pipeline

1) Fine-tune Chronos-Bolt on stocks from directory TS-RAG-main/TS-RAG:
```
python financial_pretrain.py  --stocks_dir "../../sampled_stocks/new_directory"  --pretrained_model_path "amazon/chronos-bolt-base"  --save_dir "./checkpoints/financial_finetuned"  --seq_len 512  --pred_len 64  --batch_size 64  --epochs 3
```

2) Run inference with metrics using the fine‑tuned model. Either pick 5 stocks at random (`--num_stocks 5`) or specify tickers explicitly (`--tickers`):
```
# Random 5 stocks:
python financial_infer.py --stocks_dir "../../sampled_stocks/new_directory" --num_stocks 5 --model_id "./checkpoints/financial_finetuned" --prediction_length 64 --compute_metrics --device "cpu"

# Same 5 stocks every time (use the same list for TS-RAG for fair comparison):
python financial_infer.py --stocks_dir "../../sampled_stocks/new_directory" --tickers AAT PUBM QQQA MLM GOOGL --model_id "./checkpoints/financial_finetuned" --prediction_length 64 --compute_metrics --device "cpu"
```