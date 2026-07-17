"""
QR Service: decodes QR codes from uploaded images and runs tamper
analysis using the existing QR Shield engine's TamperDetector.

This module is ONLY a wrapper around the QR Shield engine. It must
never re-implement the engine's detection logic (see Change 1 of the
backend audit) - if the engine isn't importable, the service fails
loudly with QRShieldEngineUnavailable instead of silently running a
second, divergent implementation.

Integration note for the team:
The existing CV modules (`src/image_loader`, `src/preprocessing`,
`src/qr_detector`, `src/tamper_analysis`, ...) live in a separate
package from this backend. Set QR_SHIELD_CORE_PATH in .env to point at
the directory that CONTAINS those `src/` folders (e.g. the repo root).
This module adds that path to sys.path at import time and imports the
real TamperDetector.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.response import DetectorScore, QRDecodeResult, TamperResult
from app.utils.helpers import normalize_weights

settings = get_settings()
logger = get_logger(__name__)


class QRShieldEngineUnavailable(RuntimeError):
    """
    Raised when the real QR Shield engine (src/tamper_analysis, etc.)
    cannot be imported. Callers (API layer) must translate this into a
    503 Service Unavailable response - never into a fallback
    implementation.
    """


# --- Attempt to wire in the real tamper_analysis package ---------------
_CORE_AVAILABLE = False
_RealTamperDetector = None
try:
    core_path = Path(settings.QR_SHIELD_CORE_PATH).resolve()
    if core_path.exists() and str(core_path) not in sys.path:
        sys.path.insert(0, str(core_path))
    from src.tamper_analysis.tamper_detector import TamperDetector as _RealTamperDetector  # type: ignore
    _CORE_AVAILABLE = True
    logger.info("Loaded real TamperDetector from qr_shield_core at %s", core_path)
except Exception as exc:  # noqa: BLE001 - broad on purpose, this is an optional integration
    logger.warning(
        "qr_shield_core TamperDetector not found (%s). The scan endpoint "
        "will return 503 until QR_SHIELD_CORE_PATH points at a valid "
        "QR Shield engine checkout.",
        exc,
    )

ENGINE_NAME = "qr_shield_core.tamper_analysis.TamperDetector"


class QRService:
    """Decodes QR codes and orchestrates tamper analysis for a scan."""

    def __init__(self):
        self._qr_decoder = cv2.QRCodeDetector()
        self._use_real_core = _CORE_AVAILABLE
        if self._use_real_core:
            self._tamper_detector = _RealTamperDetector()

    @property
    def engine_available(self) -> bool:
        return self._use_real_core

    # -- QR decoding -----------------------------------------------------
    def decode_qr(self, image: np.ndarray) -> QRDecodeResult:
        try:
            data, points, _ = self._qr_decoder.detectAndDecode(image)
        except cv2.error as exc:
            logger.error("OpenCV QR decode failure: %s", exc)
            return QRDecodeResult(decoded=False)

        if not data:
            return QRDecodeResult(decoded=False)

        bbox = None
        if points is not None:
            bbox = points.reshape(-1, 2).tolist()

        return QRDecodeResult(decoded=True, data=data, qr_type="QR_CODE", bounding_box=bbox)

    # -- Tamper analysis ---------------------------------------------------
    def analyze_tamper(
        self,
        image: np.ndarray,
        edge_weight: float = 0.35,
        contour_weight: float = 0.35,
        overlay_weight: float = 0.30,
        threshold: float = 0.5,
    ) -> TamperResult:
        if not self._use_real_core:
            raise QRShieldEngineUnavailable(
                "QR Shield tamper_analysis engine is not available. "
                "Set QR_SHIELD_CORE_PATH to a valid QR Shield engine checkout."
            )

        edge_weight, contour_weight, overlay_weight = normalize_weights(
            edge_weight, contour_weight, overlay_weight
        )

        try:
            result = self._tamper_detector.analyze(
                image,
                weights={"edge": edge_weight, "contour": contour_weight, "overlay": overlay_weight},
            )
        except Exception as exc:  # noqa: BLE001
            # Do NOT fall back to a second implementation - surface the
            # real engine failure so it can be diagnosed and fixed.
            logger.error("QR Shield TamperDetector raised an error: %s", exc)
            raise QRShieldEngineUnavailable(f"Tamper analysis engine failed: {exc}") from exc

        detectors = [
            DetectorScore(name=k, raw_score=v["raw"], weight=v["weight"], weighted_score=v["weighted"])
            for k, v in result.detector_scores.items()
        ]
        return TamperResult(
            is_tampered=result.confidence >= threshold,
            confidence=round(float(result.confidence), 4),
            threshold=threshold,
            detectors=detectors,
            reasons=result.reasons,
            engine=ENGINE_NAME,
        )


qr_service = QRService()
