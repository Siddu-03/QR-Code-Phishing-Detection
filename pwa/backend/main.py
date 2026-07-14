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
from app.core.config import get_settings
from app.core.logger import get_logger
from app.middleware.cors import add_cors_middleware

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
    version="1.0.0",
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
