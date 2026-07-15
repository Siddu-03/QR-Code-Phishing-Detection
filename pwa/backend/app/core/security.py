"""
Security utilities: API key verification dependency used to protect
write-sensitive endpoints (scan submission, report generation).
"""
from fastapi import Header, HTTPException, status

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


async def verify_api_key(x_api_key: str = Header(default=None, alias="X-API-Key")) -> str:
    """
    FastAPI dependency that validates the X-API-Key header against the
    configured API_KEY. Raise 401 if missing or invalid.

    In development mode with the default placeholder key, this check is
    relaxed so the team can develop locally without extra setup, but a
    warning is logged.
    """
    if settings.APP_ENV == "development" and settings.API_KEY == "change-me-in-production":
        logger.warning("API key check bypassed: running in development with default API_KEY")
        return x_api_key or "dev-mode"

    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key
