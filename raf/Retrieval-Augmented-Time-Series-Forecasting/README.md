# Usage

- To perform inference for RAF and the baseline, install the necessary packages by running:

    ```
    pip install -r requirements.txt
    ```

## Selecting Tickers
Go to `run_chronos.py` and at the top, modify the array of tickers to whatever you want

Tickers must be in this hf repo to work: https://huggingface.co/datasets/paperswithbacktest/Stocks-Daily-Price

YAML file does not need to be modified

## Running Chronos: Baseline vs. Naive RAF Performance Comparison
This codebase is built on top of the main [Chronos Forecasting GitHub repository](https://github.com/amazon-science/chronos-forecasting). For more details and updates, please refer to the official repo.

- For the baseline approach, run `run_chronos.py`:
    ```
    python run_chronos.py configs/multi.yaml path/to/result/file.csv --no-augment
    ```

- For RAF, run `run_chronos.py`:
    ```
    python run_chronos.py configs/multi.yaml path/to/result/file.csv --augment
    ```
