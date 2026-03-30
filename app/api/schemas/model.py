from pydantic import BaseModel


class ModelSummary(BaseModel):
    model_id: str
    model_name: str
    family: str
    description: str | None = None