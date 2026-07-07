"""
overlay_detection.py
Overlay/sticker tamper detector: flags regions where a foreign patch has
been placed over part of the QR code (color inconsistency, texture
mismatch, and module-grid uniformity violations).
"""

import time
import logging
import numpy as np
import cv2
from typing import Dict, Any, List

logger = logging.getLogger("qr_shield.tamper.overlay")


class OverlayDetector:
    """
    Detects tamper signatures using local color/texture statistics:
      - a clean QR is strictly bimodal (black/white modules); an overlay
        sticker introduces color and mid-tone pixels
      - grid-cell variance uniformity is broken where a patch was placed
      - suspicious regions are returned as bounding boxes for visualization
    """

    def __init__(
        self,
        grid_size: int = 21,          # approx. QR module grid resolution to sample
        color_variance_threshold: float = 12.0,
        saturation_threshold: int = 40,
    ):
        self.grid_size = grid_size
        self.color_variance_threshold = color_variance_threshold
        self.saturation_threshold = saturation_threshold

    def _cell_grid_scores(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        cell_h, cell_w = max(h // self.grid_size, 1), max(w // self.grid_size, 1)
        scores = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y0, y1 = i * cell_h, min((i + 1) * cell_h, h)
                x0, x1 = j * cell_w, min((j + 1) * cell_w, w)
                cell = gray[y0:y1, x0:x1]
                if cell.size == 0:
                    continue
                # A clean module is near-uniform (low std). Mixed/blended pixels
                # from an overlay edge raise std well above pure black/white cells.
                scores[i, j] = cell.std()
        return scores

    def _color_saturation_map(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1]

    def _suspicious_regions(self, mask: np.ndarray, min_area: int = 40) -> List[List[int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([int(x), int(y), int(w), int(h)])
        return boxes

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            image: BGR numpy array of the QR region (color required for
                   saturation-based overlay/sticker detection).
        Returns:
            dict with score (0-1), details, processing_time_ms, and bboxes.
        """
        start = time.perf_counter()

        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # A pure black/white QR should have near-zero saturation everywhere.
        saturation = self._color_saturation_map(bgr)
        sat_mask = (saturation > self.saturation_threshold).astype(np.uint8) * 255
        sat_ratio = float(np.count_nonzero(sat_mask)) / sat_mask.size

        # Grid-cell std deviation flags non-uniform (blended/overlaid) modules.
        grid_scores = self._cell_grid_scores(gray)
        high_variance_ratio = float(
            np.count_nonzero(grid_scores > self.color_variance_threshold) / grid_scores.size
        )

        sat_penalty = min(sat_ratio / 0.05, 1.0)          # >5% saturated pixels is suspicious
        variance_penalty = min(high_variance_ratio / 0.15, 1.0)  # >15% noisy cells is suspicious

        score = float(np.clip(0.55 * sat_penalty + 0.45 * variance_penalty, 0.0, 1.0))

        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
        suspected_regions = self._suspicious_regions(clean_mask)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "OverlayDetector: sat_ratio=%.4f high_var_ratio=%.4f score=%.4f (%.2f ms)",
            sat_ratio, high_variance_ratio, score, elapsed_ms,
        )

        return {
            "score": score,
            "processing_time_ms": elapsed_ms,
            "details": {
                "saturated_pixel_ratio": round(sat_ratio, 4),
                "high_variance_cell_ratio": round(high_variance_ratio, 4),
                "suspected_region_count": len(suspected_regions),
            },
            "suspected_regions": suspected_regions,
        }