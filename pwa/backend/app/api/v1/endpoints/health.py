"""
Health check endpoint - used by Member 3 for deployment monitoring and
by the frontend PWA to verify backend availability before scanning.
"""
from datetime import datetime

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.response import HealthResponse

router = APIRouter()
settings = get_settings()

APP_VERSION = "1.0.0"


@router.get("", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=APP_VERSION,
        environment=settings.APP_ENV,
        timestamp=datetime.utcnow(),
    )
