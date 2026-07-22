"""
QR Service: orchestrates the existing QR Shield engine's QR Detection
module and Tamper Analysis module. This service is ONLY a wrapper - it
must never re-implement detection or tamper-analysis logic itself
(Change 3 of the backend audit: the previous implementation used
`cv2.QRCodeDetector` directly instead of the shared `src.qr_detector`
module, and has been refactored to call the real engine instead).

Integration note for the team:
The existing CV modules (`src/qr_detector`, `src/tamper_analysis`, ...)
live in a separate package from this backend. Set QR_SHIELD_CORE_PATH in
.env to point at the directory that CONTAINS those `src/` folders (e.g.
the repo root). `app.core.qr_shield_engine` adds that path to
`sys.path` at import time; this module imports the real `detect_qr` and
`TamperDetector` from it.
"""
from typing import Any, Dict

import numpy as np

from app.core.logger import get_logger
from app.core.qr_shield_engine import QRShieldEngineUnavailable
from app.schemas.response import DetectorScore, QRDecodeResult, TamperResult
from app.utils.helpers import normalize_weights

logger = get_logger(__name__)

# --- Attempt to wire in the real qr_detector + tamper_analysis modules ---
_CORE_AVAILABLE = False
_detect_qr = None
_TamperDetector = None
_DetectorWeights = None
try:
    from src.qr_detector.qr_detector import detect_qr as _detect_qr  # type: ignore
    from src.tamper_analysis.tamper_detector import TamperDetector as _TamperDetector  # type: ignore
    from src.tamper_analysis.tamper_detector import DetectorWeights as _DetectorWeights  # type: ignore

    _CORE_AVAILABLE = True
    logger.info("Loaded real QR Shield qr_detector + tamper_analysis engine modules")
except Exception as exc:  # noqa: BLE001 - broad on purpose, this is an optional integration
    logger.warning(
        "qr_shield_core qr_detector/tamper_analysis modules not found (%s). "
        "The scan endpoint will return 503 until QR_SHIELD_CORE_PATH points "
        "at a valid QR Shield engine checkout.",
        exc,
    )

QR_DETECTOR_ENGINE_NAME = "qr_shield_core.qr_detector.detect_qr"
TAMPER_ENGINE_NAME = "qr_shield_core.tamper_analysis.TamperDetector"


class QRService:
    """Decodes QR codes and runs tamper analysis using the real engine."""

    @property
    def engine_available(self) -> bool:
        return _CORE_AVAILABLE

    # -- QR detection ------------------------------------------------------
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Runs the real QR Shield detection pipeline (`src.qr_detector.detect_qr`)
        against an image already saved to disk, returning its raw
        `DetectionResult` dict unchanged (this raw dict is also what
        `RiskEngine.assess()` expects as `detection_result`).
        """
        if not _CORE_AVAILABLE:
            raise QRShieldEngineUnavailable(
                "QR Shield qr_detector engine is not available. "
                "Set QR_SHIELD_CORE_PATH to a valid QR Shield engine checkout."
            )
        try:
            return _detect_qr(image_path)
        except (FileNotFoundError, ValueError) as exc:
            # A corrupt/missing file at this stage means our own upload
            # handling has a bug - surface it as an engine failure rather
            # than a fabricated empty result.
            logger.error("QR Shield detect_qr raised an error: %s", exc)
            raise QRShieldEngineUnavailable(f"QR detection engine failed: {exc}") from exc

    @staticmethod
    def to_api_qr_result(detection_result: Dict[str, Any]) -> QRDecodeResult:
        """Converts the engine's raw DetectionResult into our API schema.

        Only the first detection is surfaced through `ScanResponse.qr`,
        preserving the existing single-QR API contract even though the
        underlying engine supports multi-QR images.
        """
        if not detection_result.get("detected"):
            return QRDecodeResult(decoded=False)

        first = detection_result["detections"][0]
        return QRDecodeResult(
            decoded=True,
            data=first.get("data"),
            qr_type="QR_CODE",
            bounding_box=first.get("corner_points"),
        )

    # -- Tamper analysis -----------------------------------------------------
    def analyze_tamper(
        self,
        image: np.ndarray,
        edge_weight: float = 0.35,
        contour_weight: float = 0.35,
        overlay_weight: float = 0.30,
        threshold: float = 0.5,
    ):
        """Runs the real TamperDetector, returning its raw TamperResult
        object (also what `RiskEngine.assess()` expects as `tamper_result`)."""
        if not _CORE_AVAILABLE:
            raise QRShieldEngineUnavailable(
                "QR Shield tamper_analysis engine is not available. "
                "Set QR_SHIELD_CORE_PATH to a valid QR Shield engine checkout."
            )

        edge_weight, contour_weight, overlay_weight = normalize_weights(
            edge_weight, contour_weight, overlay_weight
        )
        # The real engine's pattern_interruption stage isn't exposed as a
        # tunable weight in our API (only edge/contour/overlay are, for
        # backward compatibility with the existing 3-knob contract), so it
        # is disabled (weight 0.0) rather than silently stealing weight
        # from the other three.
        weights = _DetectorWeights(edge=edge_weight, contour=contour_weight, overlay=overlay_weight, pattern=0.0)

        try:
            detector = _TamperDetector(threshold=threshold, weights=weights)
            return detector.analyze(image)
        except Exception as exc:  # noqa: BLE001
            # Do NOT fall back to a second implementation - surface the
            # real engine failure so it can be diagnosed and fixed.
            logger.error("QR Shield TamperDetector raised an error: %s", exc)
            raise QRShieldEngineUnavailable(f"Tamper analysis engine failed: {exc}") from exc

    @staticmethod
    def to_api_tamper_result(result) -> TamperResult:
        """Converts the engine's raw TamperResult into our API schema."""
        detectors = [
            DetectorScore(
                name=d.name,
                raw_score=round(float(d.score), 4),
                weight=round(float(d.weight), 4),
                weighted_score=round(float(d.score) * float(d.weight), 4),
            )
            for d in result.detector_scores
        ]
        return TamperResult(
            is_tampered=result.tampered,
            confidence=round(float(result.confidence), 4),
            threshold=result.metadata.get("threshold", 0.5),
            detectors=detectors,
            reasons=list(result.reasons),
            engine=TAMPER_ENGINE_NAME,
        )


qr_service = QRService()
