"""
Report endpoint: generates and serves a downloadable report (PDF/JSON)
for a previously completed scan.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
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
    record = await run_in_threadpool(history_store.get, payload.scan_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No scan found with id '{payload.scan_id}'",
        )

    # Report generation (PDF rendering / disk I/O) is blocking - keep it
    # off the event loop (Change 4).
    await run_in_threadpool(report_service.generate, record, payload.format)

    return ReportResponse(
        scan_id=payload.scan_id,
        report_url=f"/api/v1/report/{payload.scan_id}/download?format={payload.format}",
        generated_at=datetime.utcnow(),
    )


@router.get(
    "/{scan_id}/download",
    summary="Download a previously generated report file",
)
async def download_report(
    scan_id: str,
    format: str = "pdf",
    _api_key: str = Depends(verify_api_key),
):
    if format not in ("pdf", "json"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="format must be 'pdf' or 'json'")

    record = await run_in_threadpool(history_store.get, scan_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No scan found with id '{scan_id}'")

    report_path = await run_in_threadpool(report_service.generate, record, format)
    media_type = "application/pdf" if format == "pdf" else "application/json"
    filename = f"qr_shield_report_{scan_id}.{format}"

    return FileResponse(path=report_path, media_type=media_type, filename=filename)
