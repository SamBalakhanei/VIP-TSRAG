# VIP-TSRAG Repository Overview

This repository contains several different time series forecasting pipelines and experiments.

## Overall repository structure

- `app/` — leaderboard application-related code
- `GTR/` — code and experiments for Global Temporal Retrieval (GTR)
- `raf/` — code and experiments for Retrieval Augmented Forecasting (RAF)
- `sampled_stocks/` — stock dataset / sampled stock data used in experiments
- `scripts/` — helper scripts
- `TS-RAG-main/` — code and experiments for TS-RAG models

## Leaderboard

For the leaderboard:
1. Run python scripts/data_fill.py inside the VIP-TSRAG folder (metric results from three pipelines)
2. uvicorn app.api.main:app --reload
3. streamlit run app/ui/Home.py



## GTR

The main GTR work for this project is inside the `GTR/` folder.

Different GTR modifications are stored in different branches:

- `main` — main branch
- `horizon-aware-eval` — horizon-aware GTR modification
- `residual-gtr` — residualized GTR modification
- `past-future-backbone-gtr` — past-and-future retrieval inside the backbone
- `multi-period-resgtr` — multi-period residualized GTR

This branch structure was used to keep the different GTR ideas separate and make comparisons easier.

### Files changed for the modifications

All GTR modifications were implemented with changes only in these files:

- `GTR/models/GTR.py`
- `GTR/run.py`
- `GTR/run_5seed_compare.py`

The main architecture changes were made in `GTR.py`, the experiment flags and run settings were added in `run.py`, and `run_5seed_compare.py` was used to evaluate the modifications across multiple random seeds.

### Stock benchmarking notebook

The notebook `GTR/vip_gtr.ipynb` was used for stock benchmarking experiments.

It was used to run and evaluate GTR on the stock dataset separately from the ETTh1 modification experiments.

### Notes

The stock benchmarking experiments and the ETTh1 modification experiments were used for different purposes:

- **Stock benchmarking** was used to evaluate base GTR on the financial forecasting task in this project.
- **ETTh1 experiments** were used to test whether different GTR architecture modifications improve performance in a standard long-horizon forecasting setting.

This makes it easier to separate:
1. how GTR performs on the stock prediction task, and
2. how different GTR modifications compare under a controlled benchmark.

### Sample command: ETTh1 benchmarking with horizon-aware

Run this from inside the `GTR/` folder on the `horizon-aware-eval` branch.

```bash
python -u run.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model GTR --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --cycle 24 --train_epochs 30 --patience 5 --dropout 0.5 --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed 2026 --num_workers 0 --horizon_aware 1

