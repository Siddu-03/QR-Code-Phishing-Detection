"""
Aggregates all v1 endpoint routers into a single APIRouter mounted by main.py.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import health, history, report, scan

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(scan.router, prefix="/scan", tags=["Scan"])
api_router.include_router(report.router, prefix="/report", tags=["Report"])
api_router.include_router(history.router, prefix="/history", tags=["History"])
