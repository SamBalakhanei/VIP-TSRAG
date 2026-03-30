from pydantic import BaseModel


class RunSummary(BaseModel):
    run_id: str
    model_id: str
    model_name: str
    task_name: str
    status: str
    metric_name: str
    mean_score: float | None = None
    created_at: str | None = None


class RunDetail(BaseModel):
    run_id: str
    model_id: str
    model_name: str
    task_name: str
    status: str
    config_json: str | None = None
    artifact_path: str | None = None
    created_at: str | None = None