from fastapi import APIRouter, HTTPException, Query

from app.api.services.run_service import RunService

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
def list_runs(model_id: str | None = Query(default=None)):
    return {"rows": RunService.list_runs(model_id=model_id)}


@router.get("/{run_id}")
def get_run_detail(run_id: str):
    row = RunService.get_run_detail(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return row