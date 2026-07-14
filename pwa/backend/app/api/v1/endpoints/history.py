"""
History endpoint: lists past scans with pagination and optional
tampered-only filtering, for the frontend dashboard/history view.
"""
from typing import Optional

from fastapi import APIRouter, Query

from app.models import history_store
from app.schemas.response import HistoryResponse, ScanHistoryItem

router = APIRouter()


@router.get("", response_model=HistoryResponse, summary="List scan history")
async def get_history(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    tampered_only: Optional[bool] = Query(default=None),
) -> HistoryResponse:
    result = history_store.list(limit=limit, offset=offset, tampered_only=tampered_only)

    items = [
        ScanHistoryItem(
            scan_id=r["scan_id"],
            filename=r["filename"],
            scanned_at=r["scanned_at"],
            verdict=r["verdict"],
            confidence=r["tamper"]["confidence"],
        )
        for r in result["items"]
    ]

    return HistoryResponse(total=result["total"], limit=limit, offset=offset, items=items)
