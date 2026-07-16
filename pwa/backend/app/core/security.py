"""
Security utilities: API key verification dependency used to protect
write-sensitive endpoints (scan submission, report generation/download).
"""
import secrets

from fastapi import Header, HTTPException, status

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


async def verify_api_key(x_api_key: str = Header(default=None, alias="X-API-Key")) -> str:
    """
    FastAPI dependency that validates the X-API-Key header against the
    configured API_KEY.

    Authentication is ALWAYS enforced unless AUTH_DISABLED is explicitly
    set to true in configuration. It is never silently relaxed just
    because APP_ENV=="development" or because API_KEY still has its
    placeholder value - that combination previously bypassed auth
    automatically, which is exactly the failure mode this guards against.
    """
    if settings.AUTH_DISABLED:
        logger.warning(
            "Authentication is DISABLED via explicit AUTH_DISABLED=true config flag. "
            "This must never be set in a reachable/production environment."
        )
        return x_api_key or "auth-disabled"

    if settings.API_KEY == "change-me-in-production":
        # Fail closed: a default placeholder key must never authenticate
        # real requests, in dev or prod.
        logger.error(
            "API_KEY is still the default placeholder value. Refusing all "
            "requests until a real API_KEY is configured (or AUTH_DISABLED=true "
            "is explicitly set for local development)."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Server API key is not configured",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key
