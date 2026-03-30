from fastapi import APIRouter, Query

from app.api.schemas.leaderboard import LeaderboardResponse
from app.api.services.leaderboard_service import LeaderboardService

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get("", response_model=LeaderboardResponse)
def get_leaderboard(
    metric_name: str = Query(default="MASE"),
    family: str | None = Query(default=None),
    sector: str | None = Query(default=None),
    horizon: int | None = Query(default=None),
    lookback: int | None = Query(default=None),
):
    rows = LeaderboardService.get_leaderboard(
        metric_name=metric_name,
        family=family,
        sector=sector,
        horizon=horizon,
        lookback=lookback,
    )
    return {"rows": rows}