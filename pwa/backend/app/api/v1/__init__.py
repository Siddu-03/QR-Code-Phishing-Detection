"""
Aggregated endpoint routers for API v1.

Re-exports `api_router` (built in app.api.v1.endpoints) so that
`from app.api.v1 import api_router`, as used by main.py, resolves
correctly (Change 1 - fixes FastAPI startup failure).
"""
from app.api.v1.endpoints import api_router

__all__ = ["api_router"]
