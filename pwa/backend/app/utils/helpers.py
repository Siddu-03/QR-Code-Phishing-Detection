"""
Shared helper functions: upload validation, ID generation, image I/O.
"""
import uuid
from pathlib import Path
from typing import Tuple

import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def generate_id(prefix: str = "scan") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


async def validate_and_read_upload(file: UploadFile) -> bytes:
    """
    Validates content-type and size of an uploaded image, returns raw bytes.
    Raises HTTPException(400/413) on failure.
    """
    if file.content_type not in settings.allowed_image_types_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Allowed: {', '.join(settings.allowed_image_types_list)}"
            ),
        )

    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max size of {settings.MAX_UPLOAD_SIZE_MB}MB",
        )
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        )
    return contents


def bytes_to_cv2_image(data: bytes) -> np.ndarray:
    """Decodes raw image bytes into an OpenCV BGR ndarray."""
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image. File may be corrupted.",
        )
    return image


def save_upload_copy(data: bytes, filename: str, scan_id: str) -> str:
    """Persists a copy of the uploaded image to UPLOAD_DIR, returns saved path."""
    safe_suffix = Path(filename).suffix or ".png"
    save_path = Path(settings.UPLOAD_DIR) / f"{scan_id}{safe_suffix}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(data)
    return str(save_path)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def normalize_weights(*weights: float) -> Tuple[float, ...]:
    """Normalizes a set of weights so they sum to 1.0 (avoids div-by-zero)."""
    total = sum(weights) or 1.0
    return tuple(w / total for w in weights)
