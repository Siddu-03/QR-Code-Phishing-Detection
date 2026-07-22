"""
Scan endpoint: accepts an uploaded QR-code image and runs it through the
full QR Shield engine pipeline - QR Detection, Tamper Analysis, URL
Analysis, and Risk Assessment - storing the result in history and
returning a structured verdict to the frontend.

Every call into the QR Shield engine is CPU-bound / blocking (OpenCV,
file I/O), so each stage is dispatched via `run_in_threadpool` to keep
the FastAPI event loop free for other requests (Change 4).
"""
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.core.logger import get_logger
from app.core.security import verify_api_key
from app.models import history_store
from app.schemas.response import (
    ProcessingTimes,
    RiskAssessmentResult,
    ScanResponse,
    URLAnalysisResult,
)
from app.services.qr_service import qr_service
from app.services.risk_service import risk_service
from app.services.url_service import url_service
from app.utils.helpers import (
    bytes_to_cv2_image,
    generate_id,
    save_upload_copy,
    validate_and_read_upload,
)

router = APIRouter()
logger = get_logger(__name__)


def _determine_verdict(qr_decoded: bool, is_tampered: bool, confidence: float, threshold: float) -> str:
    if not qr_decoded:
        return "no_qr_found"
    if is_tampered:
        return "tampered"
    if confidence >= threshold * 0.7:
        return "suspicious"
    return "safe"


def _run_url_analysis(qr_data: Optional[str]) -> tuple[URLAnalysisResult, object]:
    """Runs URL Analysis on a decoded QR payload, if any. Returns the API
    schema result plus the raw engine URLResult (or None) for RiskEngine."""
    if not qr_data:
        return URLAnalysisResult(), None

    raw_result = url_service.analyze(qr_data)
    api_result = URLAnalysisResult(
        analyzed=True,
        url=qr_data,
        is_suspicious=not raw_result.is_safe(),
        reasons=list(raw_result.flags),
    )
    return api_result, raw_result


def _run_risk_assessment(detection_result: dict, scan_id: str, tamper_raw, url_raw):
    raw_result = risk_service.assess(
        detection_result, image_id=scan_id, tamper_result=tamper_raw, url_result=url_raw
    )
    api_result = RiskAssessmentResult(
        assessed=True,
        risk_level=raw_result.risk_level.value,
        score=round(float(raw_result.score), 4),
        recommendation=raw_result.recommendation,
    )
    return api_result, raw_result


@router.post(
    "",
    response_model=ScanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Scan a QR code image for tampering, phishing, and overall risk",
)
async def scan_qr_image(
    file: UploadFile = File(..., description="QR code image (png/jpg/webp)"),
    edge_weight: float = Form(default=0.35, ge=0, le=1),
    contour_weight: float = Form(default=0.35, ge=0, le=1),
    overlay_weight: float = Form(default=0.30, ge=0, le=1),
    tamper_threshold: float = Form(default=0.5, ge=0, le=1),
    _api_key: str = Depends(verify_api_key),
) -> ScanResponse:
    t_total_start = time.perf_counter()
    times = {}

    contents = await validate_and_read_upload(file)
    scan_id = generate_id("scan")

    saved_path = await run_in_threadpool(
        save_upload_copy, contents, file.filename or "upload.png", scan_id
    )
    image = await run_in_threadpool(bytes_to_cv2_image, contents)

    # -- QR Detection --------------------------------------------------
    t0 = time.perf_counter()
    detection_result = await run_in_threadpool(qr_service.detect, saved_path)
    times["qr_detection_ms"] = (time.perf_counter() - t0) * 1000

    qr_result = qr_service.to_api_qr_result(detection_result)

    # -- Tamper Analysis --------------------------------------------------
    t0 = time.perf_counter()
    tamper_raw = await run_in_threadpool(
        qr_service.analyze_tamper,
        image,
        edge_weight,
        contour_weight,
        overlay_weight,
        tamper_threshold,
    )
    times["tamper_analysis_ms"] = (time.perf_counter() - t0) * 1000
    tamper_result = qr_service.to_api_tamper_result(tamper_raw)

    # -- URL Analysis --------------------------------------------------
    t0 = time.perf_counter()
    url_result, url_raw = await run_in_threadpool(_run_url_analysis, qr_result.data)
    times["url_analysis_ms"] = (time.perf_counter() - t0) * 1000

    # -- Risk Assessment --------------------------------------------------
    t0 = time.perf_counter()
    risk_result, risk_raw = await run_in_threadpool(
        _run_risk_assessment, detection_result, scan_id, tamper_raw, url_raw
    )
    times["risk_assessment_ms"] = (time.perf_counter() - t0) * 1000

    verdict = _determine_verdict(
        qr_result.decoded, tamper_result.is_tampered, tamper_result.confidence, tamper_threshold
    )

    times["total_ms"] = (time.perf_counter() - t_total_start) * 1000

    response = ScanResponse(
        scan_id=scan_id,
        filename=file.filename or "upload.png",
        scanned_at=datetime.utcnow(),
        qr=qr_result,
        tamper=tamper_result,
        url_analysis=url_result,
        risk_assessment=risk_result,
        verdict=verdict,
        recommendation=risk_raw.recommendation,
        confidence=round(float(risk_raw.confidence), 4),
        processing_times=ProcessingTimes(**times),
    )

    await run_in_threadpool(history_store.add, response.model_dump())
    logger.info(
        "Scan %s completed with verdict=%s risk_level=%s confidence=%.3f",
        scan_id,
        verdict,
        risk_result.risk_level,
        tamper_result.confidence,
    )

    return response
