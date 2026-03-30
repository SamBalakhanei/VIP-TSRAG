from pydantic import BaseModel


class LeaderboardRow(BaseModel):
    rank: int
    model_id: str
    model_name: str
    family: str
    task_name: str
    horizon: int
    lookback: int
    sector: str
    metric_name: str
    mean_score: float
    num_runs: int
    num_series: int


class LeaderboardResponse(BaseModel):
    rows: list[LeaderboardRow]