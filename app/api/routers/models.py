from fastapi import APIRouter

from app.api.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models():
    return {"rows": ModelService.list_models()}