import os
from pathlib import Path

import torch
from chronos import ChronosPipeline

from retrieve import do_retrieve
from zeroshot import parser as zeroshot_parser


def main():
    """
    End-to-end helper for stocks:

    1) Assumes build_stocks_csv.py has already produced datasets/stocks/stocks.csv
    2) Runs do_retrieve on the 'stocks' dataset to create a retrieved CSV
    3) Calls zeroshot.py's argument parser programmatically to run ChronosBoltRetrieve

    Usage (from TS-RAG-main/TS-RAG):

        python stocks_full_rag.py --target AAT --pretrained_model_path ./checkpoints/financial_finetuned

    You can override most zeroshot arguments; sensible defaults are provided for stocks.
    """
    repo_root = Path(__file__).resolve().parent
    stocks_root = repo_root / "datasets" / "stocks"
    stocks_csv = stocks_root / "stocks.csv"

    if not stocks_csv.exists():
        raise FileNotFoundError(
            f"stocks.csv not found at {stocks_csv}. "
            f"Run build_stocks_csv.py first."
        )

    # ---- Step 1: extend zeroshot parser with a few convenience defaults for stocks ----
    parser = zeroshot_parser
    # We don't redefine arguments; we just rely on zeroshot's existing ones.
    args = parser.parse_args()

    # Provide stocks-friendly defaults if user didn't override them
    if not args.model_id:
        args.model_id = "stocks_retrieve"
    if args.root_path == "./dataset/traffic/":
        args.root_path = str(stocks_root)
    if args.data_path == "traffic.csv":
        args.data_path = "stocks.csv"
    if args.data == "custom":
        args.data = "custom_retrieve"
    if args.metadata_database_name == "ETTh2":
        args.metadata_database_name = "stocks"
    if args.metadata_frequency == "hour":
        args.metadata_frequency = "day"
    if args.lookback_length == 512:
        args.lookback_length = 512
    if args.model == "model":
        args.model = "ChronosBoltRetrieve"
    if args.pretrained_model_path == "./checkpoints/base":
        # let user override via CLI; otherwise default to financial_finetuned if present
        default_ft = repo_root / "checkpoints" / "financial_finetuned"
        if default_ft.exists():
            args.pretrained_model_path = str(default_ft)

    # ---- Step 2: build retrieval CSV for stocks if needed ----
    original_data_name = "stocks"
    retrieval_database_dir = repo_root.parent / "retrieval_database"
    retrieval_database_dir.mkdir(parents=True, exist_ok=True)

    args.metadata["lookback_length"] = args.lookback_length
    args.metadata["frequency"] = args.metadata_frequency
    args.metadata["database_name"] = args.metadata_database_name.split(" ")

    retrieval_database_names = "_".join(args.metadata["database_name"])
    retrieved_csv_name = (
        f"{original_data_name}_retrieve_"
        f"{retrieval_database_names}_"
        f"{args.metadata['lookback_length']}_"
        f"{args.mode}_"
        f"{args.embedding_tuning}.csv"
    )
    retrieved_csv_path = stocks_root / retrieved_csv_name

    if retrieved_csv_path.exists():
        print(f"Retrieved CSV already exists: {retrieved_csv_path}")
    else:
        print("Building retrieval database and retrieved CSV for stocks...")
        if "chronos" not in args.embedding_model_type:
            raise ValueError("stocks_full_rag.py currently only supports chronos embedding_model_type.")

        if args.embedding_tuning is None:
            model_path = "amazon/chronos-t5-base"
        else:
            model_path = str(
                repo_root.parent
                / "tuning_results"
                / f"{args.metadata_database_name}_{str(args.seq_len)}_chronos_{args.embedding_tuning}"
            )
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Embedding model path does not exist: {model_path}")

        device_address = f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu"
        embedding_model = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device_address,
            torch_dtype=torch.bfloat16,
        )

        top_k = args.top_k if args.top_k > 20 else 20
        do_retrieve(
            original_data_name,
            str(retrieval_database_dir),
            str(stocks_root),
            args.metadata,
            args.mode,
            top_k,
            args.seq_len,
            args.pred_len,
            seed=2021,
            dimension=args.dimension,
            embedding_model=embedding_model,
            save=True,
            embedding_tuning=args.embedding_tuning,
        )
        if not retrieved_csv_path.exists():
            raise FileNotFoundError(f"Expected retrieved CSV not found at {retrieved_csv_path}")

    # Point zeroshot at the retrieved CSV and run its main logic by re-invoking the module.
    args.data_path = retrieved_csv_name

    # Persist effective args so a direct zeroshot call would reproduce this run
    print("\nEffective zeroshot args for stocks:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Re-import zeroshot main logic by executing it as a module
    # (we assume this script is run as: python stocks_full_rag.py --...zeroshot-args...)
    from zeroshot import main as zeroshot_main  # type: ignore

    zeroshot_main(args)


if __name__ == "__main__":
    main()

