"""
qr_detector.py
==============
Detection pipeline
------------------
1. Try OpenCV QRCodeDetector (fast, handles single & multi-QR images).
2. Fall back to pyzbar if OpenCV finds nothing or raises an exception.

Supported coordinate formats
-----------------------------
corner_points  – [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]  (project-wide standard)
bbox_tuple     – [x, y, w, h]
bbox_dict      – {"x": x, "y": y, "w": w, "h": h}

Output structure
----------------
::

    {
        "detected":      bool,
        "count":         int,
        "detector_used": "opencv" | "pyzbar" | "none",
        "image_info":    {"width": int, "height": int},
        "detections": [
            {
                "data":          str,
                "confidence":    None,
                "corner_points": [[x,y], …],
                "bbox_tuple":    [x, y, w, h],
                "bbox_dict":     {"x":…, "y":…, "w":…, "h":…},
            }
        ]
    }

Usage example (visualisation module)
-------------------------------------
    result = detect_qr("image.png")
    for det in result["detections"]:
        pts  = det["corner_points"]   # preferred
        x, y, w, h = det["bbox_tuple"]
        bbox = det["bbox_dict"]

    # Optionally persist results:
    save_results_json(result, "output/scan.json")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from pyzbar import pyzbar

# ---------------------------------------------------------------------------
# Logging
 # ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
CornerPoints = list[list[int]]          # [[x,y], [x,y], [x,y], [x,y]]
BboxTuple    = list[int]                 # [x, y, w, h]
BboxDict     = dict[str, int]           # {"x":…, "y":…, "w":…, "h":…}

Detection = dict[str, Any]              # single QR detection record
DetectionResult = dict[str, Any]        # top-level return value of detect_qr
ImageInfo = dict[str, int]              # {"width": …, "height": …}


# ===========================================================================
# Public API
# ===========================================================================

def load_image(image_path: str | os.PathLike) -> np.ndarray:
    """Load and validate an image from *image_path*.

    Parameters
    ----------
    image_path:
        Absolute or relative path to the image file.

    Returns
    -------
    numpy.ndarray
        BGR image array as returned by ``cv2.imread``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *image_path*.
    ValueError
        If OpenCV cannot decode the file (corrupt or unsupported format).
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")
    if not path.is_file():
        raise FileNotFoundError(f"Path is not a regular file: {path.resolve()}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(
            f"OpenCV could not read the image file. "
            f"It may be corrupt or an unsupported format: {path.resolve()}"
        )

    logger.info("Loaded image '%s'  shape=%s", path.name, image.shape)
    return image


# ---------------------------------------------------------------------------

def detect_qr_opencv(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Detect QR codes in *image* using OpenCV's ``QRCodeDetector``.

    Tries the multi-QR detector first (``detectAndDecodeMulti``).  If that
    is unavailable (older OpenCV builds), falls back to the single-QR
    detector (``detectAndDecode``).

    Parameters
    ----------
    image:
        BGR image as a ``numpy.ndarray``.

    Returns
    -------
    list of (data, points) tuples
        *data*   – decoded string (may be empty if decoding failed).
        *points* – ``numpy.ndarray`` of shape ``(4, 2)`` or ``(1, 4, 2)``
                   containing the four corner coordinates.
        Returns an empty list when no QR code is detected.

    Raises
    ------
    RuntimeError
        If OpenCV raises an unexpected exception during detection.
    """
    detections: list[tuple[str, np.ndarray]] = []

    try:
        detector = cv2.QRCodeDetector()

        # --- Multi-QR path (OpenCV ≥ 4.5.4) ---------------------------------
        try:
            ok, decoded_list, points_list, _ = detector.detectAndDecodeMulti(image)
            if ok and points_list is not None and len(points_list) > 0:
                for data, pts in zip(decoded_list, points_list):
                    detections.append((data or "", pts))
                logger.debug(
                    "OpenCV multi-QR found %d code(s).", len(detections)
                )
                return detections
        except AttributeError:
            # detectAndDecodeMulti not available — fall through to single
            logger.debug("detectAndDecodeMulti unavailable; trying single-QR.")

        # --- Single-QR fallback ----------------------------------------------
        data, points, _ = detector.detectAndDecode(image)
        if points is not None and data:
            detections.append((data, points))
            logger.debug("OpenCV single-QR found 1 code.")

    except cv2.error as exc:
        raise RuntimeError(f"OpenCV detection error: {exc}") from exc

    return detections


# ---------------------------------------------------------------------------

def detect_qr_pyzbar(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Detect QR codes in *image* using pyzbar.

    Parameters
    ----------
    image:
        BGR image as a ``numpy.ndarray``.

    Returns
    -------
    list of (data, points) tuples
        *data*   – decoded UTF-8 string.
        *points* – ``numpy.ndarray`` of shape ``(4, 2)`` with int32 dtype,
                   representing the four corner coordinates.
        Returns an empty list when no QR code is detected.

    Raises
    ------
    RuntimeError
        If pyzbar raises an unexpected exception during detection.
    """
    detections: list[tuple[str, np.ndarray]] = []

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        decoded_objects = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])

        for obj in decoded_objects:
            try:
                data = obj.data.decode("utf-8")
            except UnicodeDecodeError:
                data = obj.data.decode("latin-1", errors="replace")

            # pyzbar returns a list of Point namedtuples
            polygon = obj.polygon
            if len(polygon) == 4:
                pts = np.array(
                    [[p.x, p.y] for p in polygon], dtype=np.int32
                )
            else:
                # Fallback: use bounding rect corners when polygon != 4 pts
                rect = obj.rect
                pts = np.array(
                    [
                        [rect.left,              rect.top],
                        [rect.left + rect.width, rect.top],
                        [rect.left + rect.width, rect.top + rect.height],
                        [rect.left,              rect.top + rect.height],
                    ],
                    dtype=np.int32,
                )

            detections.append((data, pts))
            logger.debug("pyzbar found: '%s'", data[:40] if data else "<empty>")

    except Exception as exc:  # pyzbar may raise non-cv2 errors
        raise RuntimeError(f"pyzbar detection error: {exc}") from exc

    logger.debug("pyzbar total: %d code(s).", len(detections))
    return detections


# ---------------------------------------------------------------------------

def convert_coordinates(
    raw_points: np.ndarray,
) -> tuple[CornerPoints, BboxTuple, BboxDict]:
    """Convert raw detector corner points into all supported coordinate formats.

    Parameters
    ----------
    raw_points:
        ``numpy.ndarray`` of any shape that contains exactly four (x, y) pairs,
        e.g. ``(4, 2)``, ``(1, 4, 2)``.

    Returns
    -------
    corner_points : list[list[int]]
        ``[[x0,y0], [x1,y1], [x2,y2], [x3,y3]]`` — project-wide standard.
    bbox_tuple : list[int]
        ``[x, y, w, h]`` bounding rectangle.
    bbox_dict : dict[str, int]
        ``{"x": x, "y": y, "w": w, "h": h}`` bounding rectangle.

    Raises
    ------
    ValueError
        If *raw_points* does not contain exactly four coordinate pairs.
    """
    pts = np.array(raw_points, dtype=np.float32).reshape(-1, 2)

    if pts.shape[0] != 4:
        raise ValueError(
            f"Expected exactly 4 corner points, got {pts.shape[0]}."
        )

    # --- corner_points -------------------------------------------------------
    corner_points: CornerPoints = pts.astype(int).tolist()

    # --- bounding box --------------------------------------------------------
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    x_min = int(np.floor(x_coords.min()))
    y_min = int(np.floor(y_coords.min()))
    x_max = int(np.ceil(x_coords.max()))
    y_max = int(np.ceil(y_coords.max()))
    w = x_max - x_min
    h = y_max - y_min

    bbox_tuple: BboxTuple = [x_min, y_min, w, h]
    bbox_dict:  BboxDict  = {"x": x_min, "y": y_min, "w": w, "h": h}

    return corner_points, bbox_tuple, bbox_dict


# ---------------------------------------------------------------------------

def detect_qr(image_path: str | os.PathLike) -> DetectionResult:
    """Main QR detection pipeline.

    Loads the image, attempts OpenCV detection first and automatically falls
    back to pyzbar when OpenCV finds nothing.

    Parameters
    ----------
    image_path:
        Path to the image file to analyse.

    Returns
    -------
    DetectionResult
        A dictionary with the following structure::

            {
                "detected":      bool,
                "count":         int,
                "detector_used": str,          # "opencv" | "pyzbar" | "none"
                "image_info":    {"width": int, "height": int},
                "detections": [
                    {
                        "data":          str,
                        "confidence":    None,          # reserved for future use
                        "corner_points": [[x,y], …],   # preferred format
                        "bbox_tuple":    [x, y, w, h],
                        "bbox_dict":     {"x":…, "y":…, "w":…, "h":…},
                    },
                    …
                ],
            }

        When *detected* is ``False``, *detections* is an empty list.

    Raises
    ------
    FileNotFoundError
        Propagated from :func:`load_image` when the file is missing.
    ValueError
        Propagated from :func:`load_image` when the file cannot be decoded.
    """
    # 1. Load image -----------------------------------------------------------
    image = load_image(image_path)
    img_h, img_w = image.shape[:2]
    image_info: ImageInfo = {"width": img_w, "height": img_h}

    raw_detections: list[tuple[str, np.ndarray]] = []
    detector_used = "none"

    # 2. Try OpenCV -----------------------------------------------------------
    try:
        raw_detections = detect_qr_opencv(image)
        if raw_detections:
            detector_used = "opencv"
            logger.info(
                "OpenCV detected %d QR code(s).", len(raw_detections)
            )
    except RuntimeError as exc:
        logger.warning("OpenCV detection failed (%s). Trying pyzbar…", exc)

    # 3. Fall back to pyzbar --------------------------------------------------
    if not raw_detections:
        try:
            raw_detections = detect_qr_pyzbar(image)
            if raw_detections:
                detector_used = "pyzbar"
                logger.info(
                    "pyzbar detected %d QR code(s).", len(raw_detections)
                )
        except RuntimeError as exc:
            logger.error("pyzbar detection also failed: %s", exc)

    # 4. No QR found ----------------------------------------------------------
    if not raw_detections:
        logger.info("No QR codes found in '%s'.", image_path)
        return {
            "detected":      False,
            "count":         0,
            "detector_used": detector_used,
            "image_info":    image_info,
            "detections":    [],
        }

    # 5. Build structured result ----------------------------------------------
    detections: list[Detection] = []
    for data, raw_points in raw_detections:
        try:
            corner_points, bbox_tuple, bbox_dict = convert_coordinates(raw_points)
        except ValueError as exc:
            logger.warning("Skipping detection — coordinate error: %s", exc)
            continue

        detections.append(
            {
                "data":          data,
                "confidence":    None,   # not available from either backend
                "corner_points": corner_points,
                "bbox_tuple":    bbox_tuple,
                "bbox_dict":     bbox_dict,
            }
        )

    logger.info(
        "Detection complete — %d valid QR code(s) via %s.",
        len(detections),
        detector_used,
    )

    return {
        "detected":      len(detections) > 0,
        "count":         len(detections),
        "detector_used": detector_used,
        "image_info":    image_info,
        "detections":    detections,
    }


# ===========================================================================
# Utility helpers
# ===========================================================================

def save_results_json(
    result: DetectionResult,
    output_path: str | os.PathLike,
    *,
    indent: int = 2,
) -> Path:
    """Persist a :func:`detect_qr` result dictionary to a JSON file.

    Parent directories are created automatically if they do not exist.
    The file is written with UTF-8 encoding and pretty-printed with
    *indent* spaces per level.

    Parameters
    ----------
    result:
        The ``DetectionResult`` dictionary returned by :func:`detect_qr`.
    output_path:
        Destination file path (e.g. ``"results/scan_001.json"``).
    indent:
        Number of spaces used for JSON indentation.  Defaults to ``2``.

    Returns
    -------
    pathlib.Path
        Resolved absolute path of the written file.

    Raises
    ------
    OSError
        If the file cannot be written (permissions, disk full, etc.).
    TypeError
        If *result* contains values that are not JSON-serialisable.

    Example
    -------
    ::

        result = detect_qr("sample.png")
        saved_path = save_results_json(result, "output/results.json")
        print(f"Saved to {saved_path}")
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=indent, ensure_ascii=False)

    logger.info("Results saved to '%s'.", dest.resolve())
    return dest.resolve()


# ===========================================================================
# Demo / CLI entry-point
# ===========================================================================

def main() -> None:
    """Demonstration entry-point.

    Usage::

        python qr_detector.py <image_path> [output_json_path]

    Prints decoded QR content, all coordinate formats, image metadata,
    detector used, and total count.  Optionally saves the result to JSON.
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python qr_detector.py <image_path> [output_json_path]\n"
            "Example: python qr_detector.py sample_qr.png results/out.json"
        )
        sys.exit(1)

    image_path = sys.argv[1]
    output_json: Optional[str] = sys.argv[2] if len(sys.argv) >= 3 else None

    print(f"\n{'='*60}")
    print(f"  QR Code Detector — {Path(image_path).name}")
    print(f"{'='*60}\n")

    try:
        result = detect_qr(image_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    # --- Image metadata ------------------------------------------------------
    info = result["image_info"]
    print(f"Image dimensions    : {info['width']} × {info['height']} px")
    print(f"Detector used       : {result['detector_used']}")
    print(f"Total QR codes found: {result['count']}\n")

    if not result["detected"]:
        print("No QR codes were found in the image.")
        sys.exit(0)

    for idx, det in enumerate(result["detections"], start=1):
        print(f"--- QR Code #{idx} {'─'*40}")
        print(f"  Decoded data   : {det['data']}")
        print(f"  Confidence     : {det['confidence']}")
        print(f"  corner_points  : {det['corner_points']}")
        print(f"  bbox_tuple     : {det['bbox_tuple']}")
        print(f"  bbox_dict      : {det['bbox_dict']}")
        print()

    # ------------------------------------------------------------------
    # Example: how the visualisation module can consume the output
    # ------------------------------------------------------------------
    print("--- Visualisation integration example (corner_points) ---")
    for det in result["detections"]:
        pts = np.array(det["corner_points"], dtype=np.int32)
        print(f"  cv2.polylines() input shape : {pts.reshape((-1, 1, 2)).shape}")

    # --- Optional JSON export ------------------------------------------------
    if output_json:
        saved = save_results_json(result, output_json)
        print(f"\nResults saved → {saved}")

    print("\nDone.")


if __name__ == "__main__":
    main()