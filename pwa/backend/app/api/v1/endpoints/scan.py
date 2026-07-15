"""
Scan endpoint: accepts an uploaded QR-code image, decodes it, runs the
weighted tamper-detection pipeline, stores the result in history, and
returns a structured verdict to the frontend.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.core.logger import get_logger
from app.core.security import verify_api_key
from app.models import history_store
from app.schemas.response import ScanResponse
from app.services.qr_service import qr_service
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


@router.post(
    "",
    response_model=ScanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Scan a QR code image for tampering",
)
async def scan_qr_image(
    file: UploadFile = File(..., description="QR code image (png/jpg/webp)"),
    edge_weight: float = Form(default=0.35, ge=0, le=1),
    contour_weight: float = Form(default=0.35, ge=0, le=1),
    overlay_weight: float = Form(default=0.30, ge=0, le=1),
    tamper_threshold: float = Form(default=0.5, ge=0, le=1),
    _api_key: str = Depends(verify_api_key),
) -> ScanResponse:
    contents = await validate_and_read_upload(file)
    image = bytes_to_cv2_image(contents)

    scan_id = generate_id("scan")
    save_upload_copy(contents, file.filename or "upload.png", scan_id)

    qr_result = qr_service.decode_qr(image)
    tamper_result = qr_service.analyze_tamper(
        image,
        edge_weight=edge_weight,
        contour_weight=contour_weight,
        overlay_weight=overlay_weight,
        threshold=tamper_threshold,
    )

    verdict = _determine_verdict(
        qr_result.decoded, tamper_result.is_tampered, tamper_result.confidence, tamper_threshold
    )

    response = ScanResponse(
        scan_id=scan_id,
        filename=file.filename or "upload.png",
        scanned_at=datetime.utcnow(),
        qr=qr_result,
        tamper=tamper_result,
        verdict=verdict,
    )

    history_store.add(response.model_dump())
    logger.info("Scan %s completed with verdict=%s confidence=%.3f", scan_id, verdict, tamper_result.confidence)

    return response
