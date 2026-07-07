"""
contour_analysis.py
Contour-based tamper detector: validates QR finder-pattern squareness and
flags irregular/broken contours that suggest physical damage or tampering.
"""

import time
import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("qr_shield.tamper.contour")


class ContourAnalyzer:
    """
    Detects tamper signatures using contour geometry:
      - the three QR finder patterns should be near-perfect squares
      - broken/irregular contours in the module grid indicate damage
      - overall contour count vs. expected range for a clean QR
    """

    def __init__(
        self,
        min_contour_area: int = 15,
        squareness_tolerance: float = 0.18,
        expected_finder_count: int = 3,
    ):
        self.min_contour_area = min_contour_area
        self.squareness_tolerance = squareness_tolerance
        self.expected_finder_count = expected_finder_count

    @staticmethod
    def _squareness_score(contour: np.ndarray) -> float:
        """0 = perfect square, higher = more irregular."""
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return 1.0
        aspect_penalty = abs(1.0 - (w / h))

        area = cv2.contourArea(contour)
        bbox_area = w * h
        fill_penalty = abs(1.0 - (area / bbox_area)) if bbox_area > 0 else 1.0

        return float(np.clip(0.5 * aspect_penalty + 0.5 * fill_penalty, 0.0, 1.0))

    def _find_finder_candidates(self, contours: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area:
                continue
            score = self._squareness_score(c)
            if score < self.squareness_tolerance * 2:
                candidates.append((c, score))
        candidates.sort(key=lambda t: t[1])
        return candidates[:6]

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            image: grayscale or BGR numpy array of the QR region.
        Returns:
            dict with score (0-1), details, and processing_time_ms.
        """
        start = time.perf_counter()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        finder_candidates = self._find_finder_candidates(contours)

        finder_count = len(finder_candidates)
        avg_finder_irregularity = (
            float(np.mean([s for _, s in finder_candidates])) if finder_candidates else 1.0
        )

        finder_count_penalty = min(
            abs(finder_count - self.expected_finder_count) / self.expected_finder_count, 1.0
        )
        irregularity_penalty = min(avg_finder_irregularity / self.squareness_tolerance, 1.0) \
            if self.squareness_tolerance > 0 else avg_finder_irregularity

        score = float(np.clip(0.5 * finder_count_penalty + 0.5 * irregularity_penalty, 0.0, 1.0))
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "ContourAnalyzer: finder_count=%d irregularity=%.4f score=%.4f (%.2f ms)",
            finder_count, avg_finder_irregularity, score, elapsed_ms,
        )

        return {
            "score": score,
            "processing_time_ms": elapsed_ms,
            "details": {
                "total_contours": len(contours),
                "finder_candidates_found": finder_count,
                "expected_finder_count": self.expected_finder_count,
                "avg_finder_irregularity": round(avg_finder_irregularity, 4),
            },
            "finder_contours": [c for c, _ in finder_candidates],
        }