"""
QR Shield Backend - FastAPI Application Entry Point
Member 1 - feature/backend-fastapi

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.core.config import APP_VERSION, get_settings
from app.core.logger import get_logger
from app.middleware.cors import add_cors_middleware
from app.services.qr_service import QRShieldEngineUnavailable

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    logger.info("%s starting up in '%s' mode", settings.APP_NAME, settings.APP_ENV)
    yield
    logger.info("%s shutting down", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for QR Shield - computer-vision-based QR tamper detection",
    version=APP_VERSION,
    lifespan=lifespan,
)

add_cors_middleware(app)
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "error_code": "VALIDATION_ERROR"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": f"HTTP_{exc.status_code}"},
    )


@app.exception_handler(QRShieldEngineUnavailable)
async def qr_shield_engine_unavailable_handler(request: Request, exc: QRShieldEngineUnavailable):
    """
    The backend is only a wrapper around the QR Shield engine (Change 1).
    When that engine can't be reached, callers get a clear 503 instead
    of a fabricated result from a second implementation.
    """
    logger.error("QR Shield engine unavailable on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "error_code": "ENGINE_UNAVAILABLE"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Global safety net: any exception not already handled above (a bug,
    an unexpected error from the QR Shield engine, etc.) is logged with
    full detail server-side and returned to the client as a single
    consistent JSON shape instead of leaking a raw traceback or an
    empty/non-JSON 500 response.
    """
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred. Please try again or contact support.",
            "error_code": "INTERNAL_SERVER_ERROR",
        },
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": settings.APP_NAME,
        "status": "running",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
