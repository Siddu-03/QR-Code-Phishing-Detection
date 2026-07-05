"""
tamper_detector.py
-------------------
Tamper Detection Engine — aggregates signals from edge analysis,
contour analysis, overlay detection, and pattern-interruption checks
into a single tamper decision with confidence and reasons.

Member 1 — Tamper Detection Engine
Branch: feature/tamper-engine

ASSUMED UPSTREAM INTERFACES
----------------------------
This module is written defensively against the existing teammate
files (edge_detection.py, contour_analysis.py, overlay_detection.py).
Since their exact function signatures weren't provided, this engine
assumes each module exposes ONE primary analysis function returning
either:
    (a) a float score in [0, 1], or
    (b) a dict like {"score": float, "issues": [str, ...]}

    edge_detection.analyze_edges(image) -> float | dict
    contour_analysis.analyze_contours(image) -> float | dict
    overlay_detection.detect_overlay(image) -> float | dict

If your actual modules use different function/return names, only the
three `_run_*` wrapper methods below need to change — everything else
(aggregation, confidence calc, decision logic) stays the same.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .tamper_result import TamperResult

# ---- Soft imports: keep the engine importable/testable even if a
# teammate's module isn't finished yet or has a different signature.
try:
    from . import edge_detection
except ImportError:
    edge_detection = None

try:
    from . import contour_analysis
except ImportError:
    contour_analysis = None

try:
    from . import overlay_detection
except ImportError:
    overlay_detection = None


ScoreLike = Union[float, int, Dict[str, Any], None]


@dataclass
class DetectorWeights:
    """Relative weight of each signal in the aggregate confidence score.
    Must sum to 1.0 (validated in TamperDetector.__init__)."""
    edge: float = 0.30
    contour: float = 0.30
    overlay: float = 0.30
    pattern: float = 0.10


class TamperDetector:
    """
    Aggregate tamper-detection engine.

    Usage:
        detector = TamperDetector(threshold=0.5)
        result = detector.analyze(image)          # TamperResult
        print(result.to_dict())
    """

    def __init__(
        self,
        threshold: float = 0.5,
        weights: Optional[DetectorWeights] = None,
        pattern_block_size: int = 8,
    ):
        """
        Args:
            threshold: confidence above which the QR is judged tampered.
            weights: relative importance of each sub-detector's signal.
            pattern_block_size: grid size (pixels) used for the simple
                module pattern-interruption heuristic.
        """
        self.threshold = threshold
        self.weights = weights or DetectorWeights()
        self.pattern_block_size = pattern_block_size

        total = (
            self.weights.edge
            + self.weights.contour
            + self.weights.overlay
            + self.weights.pattern
        )
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Detector weights must sum to 1.0, got {total}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, image: np.ndarray) -> TamperResult:
        """
        Run all sub-detectors on `image` and return an aggregate TamperResult.

        Args:
            image: a numpy array (as loaded by DatasetManager / image_loader),
                BGR or grayscale.

        Returns:
            TamperResult with tampered (bool), confidence (float), reasons (list).
        """
        if image is None:
            raise ValueError("image cannot be None")

        edge_score, edge_reasons = self._run_edge_detection(image)
        contour_score, contour_reasons = self._run_contour_analysis(image)
        overlay_score, overlay_reasons = self._run_overlay_detection(image)
        pattern_score, pattern_reasons = self._run_pattern_interruption(image)

        confidence = self.calculate_confidence(
            edge_score, contour_score, overlay_score, pattern_score
        )
        tampered = self.make_decision(confidence)

        result = TamperResult(
            tampered=tampered,
            confidence=confidence,
            details={
                "edge_score": edge_score,
                "contour_score": contour_score,
                "overlay_score": overlay_score,
                "pattern_score": pattern_score,
            },
        )
        result.merge_reasons(edge_reasons, contour_reasons, overlay_reasons, pattern_reasons)

        return result

    # ------------------------------------------------------------------
    # Confidence + decision logic
    # ------------------------------------------------------------------
    def calculate_confidence(
        self,
        edge_score: float,
        contour_score: float,
        overlay_score: float,
        pattern_score: float,
    ) -> float:
        """
        Weighted aggregation of individual signal scores into one
        confidence value in [0, 1]. Each input score is itself expected
        to be in [0, 1], where higher = more evidence of tampering.
        """
        scores = [edge_score, contour_score, overlay_score, pattern_score]
        weights = [self.weights.edge, self.weights.contour, self.weights.overlay, self.weights.pattern]

        for s in scores:
            if not (0.0 <= s <= 1.0):
                raise ValueError(f"Sub-detector score out of range [0,1]: {s}")

        confidence = sum(s * w for s, w in zip(scores, weights))
        return float(np.clip(confidence, 0.0, 1.0))

    def make_decision(self, confidence: float) -> bool:
        """Aggregate tamper decision based on confidence vs. threshold."""
        return confidence >= self.threshold

    # ------------------------------------------------------------------
    # Sub-detector wrappers (adapt these if teammates' signatures differ)
    # ------------------------------------------------------------------
    def _run_edge_detection(self, image: np.ndarray) -> (float, List[str]):
        if edge_detection is None or not hasattr(edge_detection, "analyze_edges"):
            return 0.0, []
        try:
            raw = edge_detection.analyze_edges(image)
            return self._normalize_result(raw, "edge inconsistency detected")
        except Exception as exc:
            return 0.0, [f"edge_detection error: {exc}"]

    def _run_contour_analysis(self, image: np.ndarray) -> (float, List[str]):
        if contour_analysis is None or not hasattr(contour_analysis, "analyze_contours"):
            return 0.0, []
        try:
            raw = contour_analysis.analyze_contours(image)
            return self._normalize_result(raw, "contour mismatch")
        except Exception as exc:
            return 0.0, [f"contour_analysis error: {exc}"]

    def _run_overlay_detection(self, image: np.ndarray) -> (float, List[str]):
        if overlay_detection is None or not hasattr(overlay_detection, "detect_overlay"):
            return 0.0, []
        try:
            raw = overlay_detection.detect_overlay(image)
            return self._normalize_result(raw, "overlay detected")
        except Exception as exc:
            return 0.0, [f"overlay_detection error: {exc}"]

    def _run_pattern_interruption(self, image: np.ndarray) -> (float, List[str]):
        """
        Lightweight built-in heuristic for module-pattern interruption,
        since no dedicated file was assigned for this responsibility.
        Computes local variance across a grid; large "dead zones" of
        near-zero variance among otherwise high-variance QR modules
        suggest an interruption (e.g. a patch or sticker).
        """
        try:
            gray = image if image.ndim == 2 else self._to_grayscale(image)
            h, w = gray.shape
            bs = self.pattern_block_size
            block_vars = []
            for y in range(0, h - bs, bs):
                for x in range(0, w - bs, bs):
                    block = gray[y : y + bs, x : x + bs]
                    block_vars.append(np.var(block))

            if not block_vars:
                return 0.0, []

            block_vars = np.array(block_vars)
            median_var = np.median(block_vars)
            if median_var < 1e-6:
                return 0.0, []

            # Fraction of blocks that are anomalously "flat" relative to
            # the overall pattern — a proxy for interruption/occlusion.
            flat_ratio = float(np.mean(block_vars < 0.1 * median_var))
            score = float(np.clip(flat_ratio * 2.0, 0.0, 1.0))

            reasons = ["pattern interruption detected"] if score > 0.3 else []
            return score, reasons
        except Exception as exc:
            return 0.0, [f"pattern_interruption error: {exc}"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        try:
            import cv2

            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception:
            # Fallback manual luminance conversion if cv2 unavailable
            return np.dot(image[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)

    @staticmethod
    def _normalize_result(raw: ScoreLike, default_reason: str) -> (float, List[str]):
        """
        Normalize a sub-detector's raw return value into (score, reasons).
        Accepts a bare float/int score, or a dict with 'score'/'issues' keys.
        """
        if raw is None:
            return 0.0, []

        if isinstance(raw, (int, float)):
            score = float(np.clip(raw, 0.0, 1.0))
            reasons = [default_reason] if score > 0.3 else []
            return score, reasons

        if isinstance(raw, dict):
            score = float(np.clip(raw.get("score", 0.0), 0.0, 1.0))
            issues = raw.get("issues") or raw.get("reasons") or []
            if not issues and score > 0.3:
                issues = [default_reason]
            return score, list(issues)

        # Unknown type — fail safe rather than crash the pipeline
        return 0.0, []


if __name__ == "__main__":
    # Quick smoke test with a synthetic image (no real QR code needed
    # to verify the pipeline runs end-to-end).
    rng = np.random.default_rng(42)
    fake_image = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)

    detector = TamperDetector(threshold=0.5)
    result = detector.analyze(fake_image)
    print(result.to_json(full=True))
