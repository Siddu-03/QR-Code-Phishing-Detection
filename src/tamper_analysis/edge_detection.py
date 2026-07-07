"""
edge_detection.py
Edge-based tamper detector: flags discontinuities and abnormal edge density
that indicate physical/digital tampering (scratches, tears, patch edges).
"""

import time
import logging
import numpy as np
import cv2
from typing import Dict, Any

logger = logging.getLogger("qr_shield.tamper.edge")


class EdgeDetector:
    """
    Detects tamper signatures using Canny edge maps:
      - edge density outside the expected range for a clean QR
      - long straight-line discontinuities (patch/sticker boundaries)
      - localized high-gradient clusters (scratches, ink smears)
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        expected_density_range: tuple = (0.08, 0.35),
        line_min_length: int = 25,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.expected_density_range = expected_density_range
        self.line_min_length = line_min_length

    def _edge_density(self, edges: np.ndarray) -> float:
        return float(np.count_nonzero(edges)) / edges.size

    def _detect_discontinuity_lines(self, edges: np.ndarray) -> int:
        """Counts long straight line segments that often mark overlay/patch borders."""
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=40,
            minLineLength=self.line_min_length, maxLineGap=4
        )
        return 0 if lines is None else len(lines)

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            image: grayscale or BGR numpy array of the QR region.
        Returns:
            dict with score (0-1), details, and processing_time_ms.
        """
        start = time.perf_counter()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        density = self._edge_density(edges)
        line_count = self._detect_discontinuity_lines(edges)

        lo, hi = self.expected_density_range
        if density < lo:
            density_penalty = (lo - density) / lo
        elif density > hi:
            density_penalty = min((density - hi) / hi, 1.0)
        else:
            density_penalty = 0.0

        # Excessive straight lines beyond QR's natural module grid suggest an overlay edge.
        line_penalty = min(max(line_count - 12, 0) / 30.0, 1.0)

        score = float(np.clip(0.6 * density_penalty + 0.4 * line_penalty, 0.0, 1.0))
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "EdgeDetector: density=%.4f lines=%d score=%.4f (%.2f ms)",
            density, line_count, score, elapsed_ms,
        )

        return {
            "score": score,
            "processing_time_ms": elapsed_ms,
            "details": {
                "edge_density": round(density, 4),
                "line_count": line_count,
                "density_penalty": round(density_penalty, 4),
                "line_penalty": round(line_penalty, 4),
            },
            "edge_map": edges,
        }