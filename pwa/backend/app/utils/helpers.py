"""
Shared helper functions: upload validation, ID generation, image I/O.
"""
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def _sniff_image_type(header: bytes) -> Optional[str]:
    """
    Identifies an image format from its magic bytes. Deliberately not
    using the stdlib `imghdr` module, which is deprecated (removed in
    Python 3.13+); this is a minimal, dependency-free replacement
    covering the formats this API accepts.
    """
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    return None


# Maps the sniffed image type to the content-types we advertise as
# accepted, so a mislabeled/renamed file can't sneak through just
# because its declared Content-Type header looked fine.
_SNIFF_TO_CONTENT_TYPES = {
    "png": {"image/png"},
    "jpeg": {"image/jpeg", "image/jpg"},
    "webp": {"image/webp"},
}


def generate_id(prefix: str = "scan") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


async def validate_and_read_upload(file: UploadFile) -> bytes:
    """
    Validates content-type and size of an uploaded image, returns raw
    bytes. Raises HTTPException(400/413) on failure.

    Hardened (Change 12) beyond a bare Content-Type header check:
      - the header is checked against the allow-list as before
      - the actual file bytes are sniffed (imghdr) and cross-checked
        against the declared content-type, so a renamed/mislabeled
        non-image file can't pass validation just by spoofing the header
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

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        )
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max size of {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    sniffed = _sniff_image_type(contents[:16])
    allowed_content_types = _SNIFF_TO_CONTENT_TYPES.get(sniffed or "", set())
    if not sniffed or file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content does not match a supported image format (png/jpeg/webp).",
        )

    return contents


def bytes_to_cv2_image(data: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into an OpenCV BGR ndarray.

    Hardened (Change 12): rejects decoded images whose dimensions exceed
    MAX_IMAGE_DIMENSION_PX, which protects downstream CV processing from
    decompression-bomb-style images (a small file that decodes into an
    enormous array and exhausts memory/CPU).
    """
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image. File may be corrupted.",
        )

    height, width = image.shape[:2]
    max_dim = settings.MAX_IMAGE_DIMENSION_PX
    if height > max_dim or width > max_dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image dimensions {width}x{height} exceed the maximum allowed "
                f"{max_dim}x{max_dim} pixels."
            ),
        )

    return image


def save_upload_copy(data: bytes, filename: str, scan_id: str) -> str:
    """Persists a copy of the uploaded image to UPLOAD_DIR, returns saved path."""
    safe_suffix = Path(filename).suffix.lower() or ".png"
    if safe_suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        safe_suffix = ".png"
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
