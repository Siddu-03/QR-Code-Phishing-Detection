"""
Report endpoint: generates and serves a downloadable report (PDF/JSON)
for a previously completed scan.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.core.security import verify_api_key
from app.models import history_store
from app.schemas.request import ReportRequest
from app.schemas.response import ReportResponse
from app.services.report_service import report_service

router = APIRouter()


@router.post(
    "",
    response_model=ReportResponse,
    summary="Generate a report for a completed scan",
)
async def generate_report(
    payload: ReportRequest,
    _api_key: str = Depends(verify_api_key),
) -> ReportResponse:
    record = history_store.get(payload.scan_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No scan found with id '{payload.scan_id}'",
        )

    report_path = report_service.generate(record, fmt=payload.format)

    return ReportResponse(
        scan_id=payload.scan_id,
        report_url=f"/api/v1/report/{payload.scan_id}/download?format={payload.format}",
        generated_at=datetime.utcnow(),
    )


@router.get(
    "/{scan_id}/download",
    summary="Download a previously generated report file",
)
async def download_report(scan_id: str, format: str = "pdf"):
    if format not in ("pdf", "json"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="format must be 'pdf' or 'json'")

    record = history_store.get(scan_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No scan found with id '{scan_id}'")

    report_path = report_service.generate(record, fmt=format)
    media_type = "application/pdf" if format == "pdf" else "application/json"
    filename = f"qr_shield_report_{scan_id}.{format}"

    return FileResponse(path=report_path, media_type=media_type, filename=filename)
