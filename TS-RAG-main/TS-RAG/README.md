Note to avoid dependency conflicts: Python version 3.9.25
Requirements file: ____
Use Conda

Commands used to run each of the pipelines:

python run_pipeline_baseline.py --device cpu --seq_len 256 --prediction_length 32 --epochs 1 --batch_size 32

python run_pipeline_tsrag_light.py --device cpu --context_length 256 --prediction_length 32 --batch_size 64 --train_steps 1200 --evaluation_steps 300 --top_k 5

python run_pipeline_tsrag_full.py --device cpu --artifact_device cpu --context_length 256 --prediction_length 32 --batch_size 64 --train_steps 800 --evaluation_steps 200 --top_k 5

python run_pipeline_tsrag_full_sector.py --device cpu --artifact_device cpu --context_length 256 --prediction_length 32 --batch_size 64 --train_steps 800 --evaluation_steps 200 --top_k 5 --candidate_multiplier 2 --sector_bonus 0.10 --naics_bonus 0.15