"""
QR Service: decodes QR codes from uploaded images and runs the
Weeks 1-4 tamper-detection pipeline (image_loader + tamper_analysis)
built earlier in the project.

Integration note for the team:
The existing CV modules (`src/image_loader`, `src/tamper_analysis`,
`src/dataset_management`) live in a separate package from this backend.
Set QR_SHIELD_CORE_PATH in .env to point at the directory that CONTAINS
those `src/` folders (e.g. the repo root, or wherever `tamper_analysis`
is importable from). This module adds that path to sys.path at import
time and imports the real TamperDetector. If the core package isn't
found yet (e.g. running the backend standalone before the folders are
merged in), a compatible fallback implementation is used instead so the
API keeps working end-to-end during development.
"""
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logger import get_logger
from app.schemas.response import DetectorScore, QRDecodeResult, TamperResult
from app.utils.helpers import normalize_weights

settings = get_settings()
logger = get_logger(__name__)

# --- Attempt to wire in the real Week 3/4 tamper_analysis package -----
_CORE_AVAILABLE = False
try:
    core_path = Path(settings.QR_SHIELD_CORE_PATH).resolve()
    if core_path.exists() and str(core_path) not in sys.path:
        sys.path.insert(0, str(core_path))
    from src.tamper_analysis.tamper_detector import TamperDetector as _RealTamperDetector  # type: ignore
    _CORE_AVAILABLE = True
    logger.info("Loaded real TamperDetector from qr_shield_core at %s", core_path)
except Exception as exc:  # noqa: BLE001 - broad on purpose, this is an optional integration
    logger.warning(
        "qr_shield_core TamperDetector not found (%s). "
        "Falling back to built-in lightweight detectors. "
        "Set QR_SHIELD_CORE_PATH in .env once the CV modules are merged in.",
        exc,
    )


class QRService:
    """Decodes QR codes and orchestrates tamper analysis for a scan."""

    def __init__(self):
        self._qr_decoder = cv2.QRCodeDetector()
        self._use_real_core = _CORE_AVAILABLE
        if self._use_real_core:
            self._tamper_detector = _RealTamperDetector()

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
        edge_weight, contour_weight, overlay_weight = normalize_weights(
            edge_weight, contour_weight, overlay_weight
        )

        if self._use_real_core:
            return self._analyze_with_real_core(image, edge_weight, contour_weight, overlay_weight, threshold)
        return self._analyze_with_fallback(image, edge_weight, contour_weight, overlay_weight, threshold)

    def _analyze_with_real_core(self, image, edge_w, contour_w, overlay_w, threshold) -> TamperResult:
        try:
            result = self._tamper_detector.analyze(
                image,
                weights={"edge": edge_w, "contour": contour_w, "overlay": overlay_w},
            )
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
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Real TamperDetector failed, falling back: %s", exc)
            return self._analyze_with_fallback(image, edge_w, contour_w, overlay_w, threshold)

    def _analyze_with_fallback(self, image, edge_w, contour_w, overlay_w, threshold) -> TamperResult:
        """
        Lightweight stand-in for EdgeDetector / ContourAnalyzer / OverlayDetector,
        used only when the real qr_shield_core package isn't wired in yet.
        Mirrors the same weighted-scoring shape so the API contract never changes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edge_score = self._edge_irregularity_score(gray)
        contour_score = self._contour_irregularity_score(gray)
        overlay_score = self._overlay_score(image)

        weighted = (
            edge_score * edge_w + contour_score * contour_w + overlay_score * overlay_w
        )

        detectors = [
            DetectorScore(name="EdgeDetector", raw_score=edge_score, weight=edge_w, weighted_score=edge_score * edge_w),
            DetectorScore(name="ContourAnalyzer", raw_score=contour_score, weight=contour_w, weighted_score=contour_score * contour_w),
            DetectorScore(name="OverlayDetector", raw_score=overlay_score, weight=overlay_w, weighted_score=overlay_score * overlay_w),
        ]

        reasons: List[str] = []
        if edge_score > 0.6:
            reasons.append("High edge irregularity detected around QR finder patterns")
        if contour_score > 0.6:
            reasons.append("Contour geometry deviates from expected QR module grid")
        if overlay_score > 0.6:
            reasons.append("Possible overlay/sticker detected on QR surface")

        return TamperResult(
            is_tampered=weighted >= threshold,
            confidence=round(float(weighted), 4),
            threshold=threshold,
            detectors=detectors,
            reasons=reasons,
        )

    @staticmethod
    def _edge_irregularity_score(gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 80, 160)
        density = float(np.count_nonzero(edges)) / edges.size
        return float(np.clip(density * 4, 0, 1))

    @staticmethod
    def _contour_irregularity_score(gray: np.ndarray) -> float:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        irregular = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 20:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.5:
                irregular += 1
        ratio = irregular / max(len(contours), 1)
        return float(np.clip(ratio * 2, 0, 1))

    @staticmethod
    def _overlay_score(image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        colorful_ratio = float(np.count_nonzero(saturation > 60)) / saturation.size
        return float(np.clip(colorful_ratio * 3, 0, 1))


qr_service = QRService()
