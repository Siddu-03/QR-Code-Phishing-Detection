"""
tamper_detector.py
-------------------
Production implementation of the QR Shield Tamper Detection Engine.

Combines four deterministic, OpenCV/NumPy-based analysis stages into a
single weighted confidence score, packaged as a `TamperResult`:

1. Edge consistency     — Canny edge-density uniformity across image
                           quadrants; also captures gross edge-discontinuity
                           damage.
2. Contour consistency  — largest quadrilateral contour squareness / angle
                           deviation, plus corner integrity, finder-pattern
                           damage (via contour hierarchy), and overall shape
                           consistency.
3. Overlay detection    — HSV saturation / local-contrast anomalies typical
                           of a printed sticker placed over a QR code, plus
                           quiet-zone violations at the image border.
4. Pattern interruption — block-wise grayscale variance used to find
                           anomalously "dead" (flat) regions within an
                           otherwise textured module grid — a proxy for
                           basic structural anomaly / module-grid mismatch
                           detection.

No machine learning is used anywhere in this module; every stage is a
deterministic image-processing computation suitable for real-time
execution (webcam / live-scan pipelines, FastAPI request handlers, PWA
client-triggered scans).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .tamper_result import Anomaly, DetectorScore, Severity, TamperResult, TamperType

logger = logging.getLogger("tamper_analysis.tamper_detector")

_WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass
class DetectorWeights:
    """Relative weight of each analysis stage in the final confidence score.

    Values need not be pre-normalized by the caller, but their sum must
    equal 1.0 (within floating-point tolerance) at `TamperDetector`
    construction time — this keeps the weighting explicit and auditable
    rather than silently renormalized.
    """

    edge: float = 0.25
    contour: float = 0.25
    overlay: float = 0.25
    pattern: float = 0.25

    def as_dict(self) -> Dict[str, float]:
        return {
            "edge": self.edge,
            "contour": self.contour,
            "overlay": self.overlay,
            "pattern": self.pattern,
        }


class TamperDetector:
    """Deterministic, multi-stage QR-code graphic tamper detector.

    Parameters
    ----------
    threshold : float, default 0.5
        Confidence at or above which an image is classified as tampered
        (inclusive).
    weights : DetectorWeights, optional
        Per-stage weighting used to combine the four stage scores into a
        single confidence value. Defaults to equal 0.25 weighting per
        stage. Must sum to 1.0.
    block_size : int, default 16
        Grid cell size (pixels) used by the pattern-interruption stage.

    Public API
    ----------
    analyze(image) -> TamperResult
    reset() -> None
    calculate_confidence(edge, contour, overlay, pattern) -> float
    make_decision(confidence) -> bool
    """

    def __init__(
        self,
        threshold: float = 0.5,
        weights: Optional[DetectorWeights] = None,
        block_size: int = 16,
    ) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be within [0.0, 1.0], got {threshold!r}")
        self.threshold = float(threshold)

        self.weights = weights or DetectorWeights()
        weight_sum = sum(self.weights.as_dict().values())
        if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"DetectorWeights must sum to 1.0, got {weight_sum!r} "
                f"({self.weights.as_dict()})"
            )

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size!r}")
        self.block_size = block_size

        self._last_result: Optional[TamperResult] = None
        logger.debug(
            "TamperDetector initialized (threshold=%.2f, weights=%s, block_size=%d)",
            self.threshold,
            self.weights.as_dict(),
            self.block_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image: Optional[np.ndarray]) -> TamperResult:
        """Run the full tamper-analysis pipeline on a single BGR image.

        Parameters
        ----------
        image : numpy.ndarray
            BGR uint8 image containing a single QR code (or a full image
            containing one QR code).

        Returns
        -------
        TamperResult

        Raises
        ------
        ValueError
            If `image` is None, not an ndarray, or empty.
        RuntimeError
            If an unrecoverable internal (OpenCV) failure occurs.
        """
        if image is None:
            raise ValueError("analyze() received image=None")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"analyze() expected numpy.ndarray, got {type(image).__name__}")
        if image.size == 0:
            raise ValueError("analyze() received an empty image array")

        start = time.perf_counter()
        try:
            gray = self._to_grayscale(image)
            edge_score, edge_reasons = self._run_edge_consistency(gray)
            contour_score, contour_reasons = self._run_contour_consistency(gray)
            overlay_score, overlay_reasons = self._run_overlay_detection(image)
            pattern_score, pattern_reasons = self._run_pattern_interruption(gray)
        except cv2.error as exc:
            logger.error("Unrecoverable OpenCV failure during analysis: %s", exc)
            raise RuntimeError(f"OpenCV failure during tamper analysis: {exc}") from exc

        confidence = self.calculate_confidence(
            edge_score, contour_score, overlay_score, pattern_score
        )
        tampered = self.make_decision(confidence)

        reasons: List[str] = []
        for stage_reasons in (edge_reasons, contour_reasons, overlay_reasons, pattern_reasons):
            for reason in stage_reasons:
                if reason not in reasons:
                    reasons.append(reason)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detector_scores = [
            DetectorScore(name="edge_consistency", score=edge_score, weight=self.weights.edge),
            DetectorScore(name="contour_consistency", score=contour_score, weight=self.weights.contour),
            DetectorScore(name="overlay_detection", score=overlay_score, weight=self.weights.overlay),
            DetectorScore(name="pattern_interruption", score=pattern_score, weight=self.weights.pattern),
        ]
        anomalies = self._build_anomalies(edge_score, contour_score, overlay_score, pattern_score, reasons)

        result = TamperResult(
            tampered=tampered,
            confidence=confidence,
            reasons=reasons,
            analysis_time_ms=elapsed_ms,
            metadata={
                "threshold": self.threshold,
                "weights": self.weights.as_dict(),
                "stage_scores": {
                    "edge": round(edge_score, 4),
                    "contour": round(contour_score, 4),
                    "overlay": round(overlay_score, 4),
                    "pattern": round(pattern_score, 4),
                },
            },
            anomalies=anomalies,
            detector_scores=detector_scores,
        )
        self._last_result = result
        logger.info(
            "analyze() complete — tampered=%s confidence=%.4f elapsed=%.2fms",
            tampered,
            confidence,
            elapsed_ms,
        )
        return result

    def reset(self) -> None:
        """Clear cached state from the previous `analyze()` call.

        The detector is otherwise stateless — each `analyze()` call is
        fully self-contained — so this only drops the cached last result.
        """
        self._last_result = None
        logger.debug("TamperDetector state reset")

    # ------------------------------------------------------------------
    # Confidence aggregation
    # ------------------------------------------------------------------
    def calculate_confidence(self, edge: float, contour: float, overlay: float, pattern: float) -> float:
        """Combine four per-stage scores (each in [0.0, 1.0]) into a single
        weighted confidence value in [0.0, 1.0]."""
        for name, value in (("edge", edge), ("contour", contour), ("overlay", overlay), ("pattern", pattern)):
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} score must be within [0.0, 1.0], got {value!r}")

        confidence = (
            edge * self.weights.edge
            + contour * self.weights.contour
            + overlay * self.weights.overlay
            + pattern * self.weights.pattern
        )
        return float(min(max(confidence, 0.0), 1.0))

    def make_decision(self, confidence: float) -> bool:
        """Binary tamper decision — inclusive at the threshold."""
        return confidence >= self.threshold

    # ------------------------------------------------------------------
    # Stage 1 — Edge consistency (+ edge discontinuity)
    # ------------------------------------------------------------------
    def _run_edge_consistency(self, gray: np.ndarray) -> Tuple[float, List[str]]:
        """Detect edge-density non-uniformity across image quadrants.

        A clean QR code's Canny edge map is roughly uniform in density
        across its four quadrants (regular module grid). A localized edit,
        patch, or discontinuity concentrates edge activity in one quadrant
        relative to the others.
        """
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            edges[0:mid_h, 0:mid_w],
            edges[0:mid_h, mid_w:w],
            edges[mid_h:h, 0:mid_w],
            edges[mid_h:h, mid_w:w],
        ]
        densities = np.array(
            [float(np.count_nonzero(q)) / max(q.size, 1) for q in quadrants]
        )
        mean_density = float(np.mean(densities))
        if mean_density <= 1e-9:
            raw_score = 0.0
        else:
            coeff_of_variation = float(np.std(densities) / mean_density)
            raw_score = min(coeff_of_variation / 1.5, 1.0)  # empirically-scaled to [0, 1]

        default_reason = "Edge density is inconsistent across image quadrants (possible localized edit)"
        return self._normalize_result(raw_score, default_reason)

    # ------------------------------------------------------------------
    # Stage 2 — Contour consistency (+ corner integrity, finder pattern,
    #           shape consistency)
    # ------------------------------------------------------------------
    def _run_contour_consistency(self, gray: np.ndarray) -> Tuple[float, List[str]]:
        """Check the squareness/corner regularity of the QR's outer boundary
        contour, and inspect finder-pattern corners for the expected
        nested-square (bullseye) structure."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.5, ["No contours detected — unable to verify shape or finder patterns"]

        reasons: List[str] = []
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h else 0.0
            aspect_deviation = abs(1.0 - aspect_ratio)
            shape_penalty = min(aspect_deviation * 2.0, 1.0)
            if aspect_deviation > 0.1:
                reasons.append("Outer boundary is not square (possible corner damage or crop)")
        else:
            shape_penalty = 0.4
            reasons.append(
                f"Outer boundary approximates to {len(approx)} vertices, expected 4 (quadrilateral)"
            )

        finder_like = self._count_nested_square_contours(contours, hierarchy)
        if finder_like < 3:
            finder_penalty = (3 - finder_like) / 3.0
            reasons.append(
                f"Only {finder_like}/3 finder-pattern (nested square) contours detected — "
                "possible finder pattern damage"
            )
        else:
            finder_penalty = 0.0

        raw_score = min(0.5 * shape_penalty + 0.5 * finder_penalty, 1.0)
        default_reason = "Contour shape deviates from expected QR geometry"
        score, fallback_reasons = self._normalize_result(raw_score, default_reason)
        return score, (reasons if reasons else fallback_reasons)

    @staticmethod
    def _count_nested_square_contours(contours, hierarchy) -> int:
        """Count contours that look like a finder pattern: a roughly-square
        contour with both a parent and at least one child contour (the
        nested square-in-square-in-square structure of a QR finder
        pattern)."""
        if hierarchy is None:
            return 0
        hierarchy = hierarchy[0]
        count = 0
        for i, contour in enumerate(contours):
            _, _, child, parent = hierarchy[i]
            if parent == -1 or child == -1:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4 and cv2.contourArea(contour) > 25:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Stage 3 — Overlay detection (+ quiet-zone inspection)
    # ------------------------------------------------------------------
    def _run_overlay_detection(self, image: np.ndarray) -> Tuple[float, List[str]]:
        """Detect printed-sticker overlays via HSV saturation anomalies, and
        check the image border (quiet zone) for unexpected non-background
        content."""
        if image.ndim == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].astype(np.float32)
        else:
            saturation = np.zeros(image.shape[:2], dtype=np.float32)

        reasons: List[str] = []
        # A clean QR code (black/white/gray) has near-zero saturation
        # everywhere; meaningful high-saturation regions suggest a printed
        # color sticker placed on top.
        high_sat_ratio = float(np.count_nonzero(saturation > 60.0)) / max(saturation.size, 1)
        overlay_raw = min(high_sat_ratio * 3.0, 1.0)
        if high_sat_ratio > 0.02:
            reasons.append(
                f"{high_sat_ratio:.1%} of pixels show elevated color saturation "
                "(possible overlay sticker)"
            )

        quiet_zone_raw = self._quiet_zone_violation_ratio(image)
        if quiet_zone_raw > 0.15:
            reasons.append("Border quiet-zone shows unexpected non-background content")

        raw_score = min(0.7 * overlay_raw + 0.3 * quiet_zone_raw, 1.0)
        default_reason = "Localized color/texture anomaly suggests a physical overlay"
        score, fallback_reasons = self._normalize_result(raw_score, default_reason)
        return score, (reasons if reasons else fallback_reasons)

    @staticmethod
    def _quiet_zone_violation_ratio(image: np.ndarray, margin_ratio: float = 0.06) -> float:
        """Return the fraction of border-margin pixels that deviate strongly
        from the dominant (background) intensity — a proxy for quiet-zone
        violations."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        margin = max(int(min(h, w) * margin_ratio), 1)

        border_pixels = np.concatenate(
            [
                gray[:margin, :].ravel(),
                gray[-margin:, :].ravel(),
                gray[:, :margin].ravel(),
                gray[:, -margin:].ravel(),
            ]
        )
        if border_pixels.size == 0:
            return 0.0

        hist_counts = np.bincount(border_pixels, minlength=256)
        dominant_value = int(np.argmax(hist_counts))
        deviation = np.abs(border_pixels.astype(np.int16) - dominant_value)
        return float(np.count_nonzero(deviation > 60)) / border_pixels.size

    # ------------------------------------------------------------------
    # Stage 4 — Pattern interruption (+ basic structural anomaly / module
    #           grid mismatch)
    # ------------------------------------------------------------------
    def _run_pattern_interruption(self, gray: np.ndarray) -> Tuple[float, List[str]]:
        """Partition the image into a grid of blocks, compute per-block
        texture variance, and flag blocks whose variance is a statistical
        outlier (anomalously "dead"/flat) relative to the rest of the grid.

        A perfectly uniform image (all blocks equally flat) has zero
        variance-of-variance and therefore yields a score of 0.0 — there is
        nothing anomalous about a grid where every block behaves the same.
        """
        h, w = gray.shape
        bs = self.block_size
        min_block_pixels = max((bs * bs) // 4, 1)

        block_variances: List[float] = []
        for y in range(0, h, bs):
            for x in range(0, w, bs):
                block = gray[y:y + bs, x:x + bs]
                if block.size < min_block_pixels:
                    continue
                block_variances.append(float(np.var(block.astype(np.float32))))

        if len(block_variances) < 2:
            return 0.0, []

        variances = np.array(block_variances)
        mean_var = float(np.mean(variances))
        std_var = float(np.std(variances))

        if std_var <= 1e-9:
            # No variation across blocks at all -> nothing anomalous to report.
            return 0.0, []

        z_scores = (variances - mean_var) / std_var
        dead_block_ratio = float(np.count_nonzero(z_scores < -1.5)) / variances.size
        raw_score = min(dead_block_ratio * 4.0, 1.0)

        default_reason = "Anomalously flat regions detected within the module grid (possible occlusion/patch)"
        return self._normalize_result(raw_score, default_reason)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")

    @staticmethod
    def _normalize_result(
        raw: Union[float, int, Dict[str, Any], None],
        default_reason: str,
    ) -> Tuple[float, List[str]]:
        """Normalize a stage's raw output into a `(score, reasons)` pair.

        Accepts:
        - `None`            -> (0.0, [])
        - a bare float/int   -> (score, [default_reason]) if score > 0 else (score, [])
        - a dict with "score" and optionally "issues"
                             -> (score, issues) if "issues" present,
                                else (score, [default_reason] if score > 0 else [])
        """
        if raw is None:
            return 0.0, []

        if isinstance(raw, dict):
            score = float(raw.get("score", 0.0))
            if "issues" in raw:
                reasons = list(raw["issues"])
            else:
                reasons = [default_reason] if score > 0.0 else []
            return score, reasons

        if isinstance(raw, (int, float)):
            score = float(raw)
            reasons = [default_reason] if score > 0.0 else []
            return score, reasons

        raise TypeError(f"_normalize_result received unsupported type: {type(raw).__name__}")

    def _build_anomalies(
        self,
        edge_score: float,
        contour_score: float,
        overlay_score: float,
        pattern_score: float,
        reasons: List[str],
    ) -> List[Anomaly]:
        """Translate stage scores/reasons into rich `Anomaly` records for
        callers that consume the extended TamperResult contract."""
        stage_map = [
            (edge_score, TamperType.EDGE_DISCONTINUITY),
            (contour_score, TamperType.CONTOUR_IRREGULARITY),
            (overlay_score, TamperType.OVERLAY_STICKER),
            (pattern_score, TamperType.MODULE_GRID_MISMATCH),
        ]
        anomalies: List[Anomaly] = []
        reason_iter = iter(reasons)
        for score, tamper_type in stage_map:
            if score <= 0.0:
                continue
            description = next(reason_iter, f"{tamper_type.value} detected")
            anomalies.append(
                Anomaly(
                    type=tamper_type,
                    severity=self._severity_for_score(score),
                    confidence=score,
                    description=description,
                )
            )
        return anomalies

    @staticmethod
    def _severity_for_score(score: float) -> Severity:
        if score >= 0.85:
            return Severity.CRITICAL
        if score >= 0.6:
            return Severity.HIGH
        if score >= 0.3:
            return Severity.MEDIUM
        return Severity.LOW