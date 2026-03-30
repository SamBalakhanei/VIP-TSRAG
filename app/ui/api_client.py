import requests

BASE_URL = "http://localhost:8000"


def get_leaderboard(
    metric_name: str = "MASE",
    family: str | None = None,
    sector: str | None = None,
    horizon: int | None = None,
    lookback: int | None = None,
):
    params = {"metric_name": metric_name}

    if family:
        params["family"] = family
    if sector:
        params["sector"] = sector
    if horizon is not None:
        params["horizon"] = horizon
    if lookback is not None:
        params["lookback"] = lookback

    response = requests.get(f"{BASE_URL}/leaderboard", params=params, timeout=30)
    response.raise_for_status()
    return response.json()["rows"]


def list_models():
    response = requests.get(f"{BASE_URL}/models", timeout=30)
    response.raise_for_status()
    return response.json()["rows"]


def list_runs(model_id: str | None = None):
    params = {}
    if model_id:
        params["model_id"] = model_id
    response = requests.get(f"{BASE_URL}/runs", params=params, timeout=30)
    response.raise_for_status()
    return response.json()["rows"]


def get_run_detail(run_id: str):
    response = requests.get(f"{BASE_URL}/runs/{run_id}", timeout=30)
    response.raise_for_status()
    return response.json()