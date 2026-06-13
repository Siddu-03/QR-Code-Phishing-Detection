"""
live_scan.py
=============
Live webcam QR scanning module.

Reuses the existing project pipeline:
    - src.qr_detector.qr_detector  : detect_qr_opencv, detect_qr_pyzbar,
                                      convert_coordinates (same detection
                                      + coordinate logic as detect_qr())
    - src.visualization.draw_box   : same drawing style (semi-transparent
                                      overlay, border rectangle, coordinate
                                      label) replicated for live frames

No detection or visualisation logic is rewritten — frame-level detection
calls the same OpenCV/pyzbar functions used by qr_detector.detect_qr(),
and frame-level drawing mirrors draw_box.py's overlay/rectangle/label style.

Usage
-----
    python -m src.live_camera.live_scan

Controls
--------
    q  -  quit the live scan
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

# Reuse existing detection building blocks (no detection logic rewritten)
from src.qr_detector.qr_detector import (
    detect_qr_opencv,
    detect_qr_pyzbar,
    convert_coordinates,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("live_camera.live_scan")

# ---------------------------------------------------------------------------
# Constants — mirror draw_box.py's visual style
# ---------------------------------------------------------------------------
CAMERA_INDEX     = 0
BOX_COLOUR_BGR   = (0, 255, 0)   # green  — matches draw_box.py
LABEL_COLOUR_BGR = (0, 0, 255)   # red    — matches draw_box.py
OVERLAY_ALPHA    = 0.3
BOX_THICKNESS    = 3
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.7
FONT_THICKNESS   = 2
EXIT_KEY         = ord("q")


def detect_qr_frame(frame: np.ndarray) -> dict:
    """Run QR detection on a single in-memory frame.

    Mirrors the OpenCV-first / pyzbar-fallback pipeline of
    :func:`qr_detector.detect_qr`, but operates directly on a frame
    array instead of an image path (avoids disk I/O per frame).

    Parameters
    ----------
    frame:
        BGR image array captured from the webcam.

    Returns
    -------
    dict
        Same structure as ``qr_detector.detect_qr``'s return value
        (``detected``, ``count``, ``detector_used``, ``detections``).
    """
    raw_detections: list[tuple[str, np.ndarray]] = []
    detector_used = "none"

    try:
        raw_detections = detect_qr_opencv(frame)
        if raw_detections:
            detector_used = "opencv"
    except RuntimeError as exc:
        logger.debug("OpenCV detection failed (%s); trying pyzbar.", exc)

    if not raw_detections:
        try:
            raw_detections = detect_qr_pyzbar(frame)
            if raw_detections:
                detector_used = "pyzbar"
        except RuntimeError as exc:
            logger.debug("pyzbar detection failed: %s", exc)

    detections = []
    for data, raw_points in raw_detections:
        try:
            corner_points, bbox_tuple, bbox_dict = convert_coordinates(raw_points)
        except ValueError as exc:
            logger.debug("Skipping detection — coordinate error: %s", exc)
            continue
        detections.append(
            {
                "data": data,
                "confidence": None,
                "corner_points": corner_points,
                "bbox_tuple": bbox_tuple,
                "bbox_dict": bbox_dict,
            }
        )

    img_h, img_w = frame.shape[:2]

    return {
        "detected": len(detections) > 0,
        "count": len(detections),
        "detector_used": detector_used,
        "image_info": {"width": img_w, "height": img_h},
        "detections": detections,
    }


def draw_detections(frame: np.ndarray, result: dict) -> np.ndarray:
    """Draw bounding boxes and labels on *frame*, in draw_box.py's style.

    For each detection draws:
        * a semi-transparent filled rectangle (alpha overlay)
        * a solid border rectangle
        * a coordinate label above the box
        * the decoded QR data below the box

    Parameters
    ----------
    frame:
        BGR frame to annotate (modified in place and returned).
    result:
        Output of :func:`detect_qr_frame`.

    Returns
    -------
    numpy.ndarray
        The annotated frame (same object as *frame*).
    """
    for idx, det in enumerate(result["detections"], start=1):
        x, y, w, h = det["bbox_tuple"]

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), BOX_COLOUR_BGR, -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

        # Solid border
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOUR_BGR, BOX_THICKNESS)

        # Coordinate label above box
        coord_text = f"x={x}, y={y}"
        cv2.putText(
            frame, coord_text, (x, max(y - 10, 10)),
            FONT, FONT_SCALE, LABEL_COLOUR_BGR, FONT_THICKNESS,
        )

        # Decoded data below box
        data = det["data"] or "<empty>"
        data_preview = data[:50] + ("…" if len(data) > 50 else "")
        cv2.putText(
            frame, f"QR #{idx}: {data_preview}", (x, y + h + 20),
            FONT, 0.55, BOX_COLOUR_BGR, 2,
        )

    return frame


def draw_status(frame: np.ndarray, result: dict) -> np.ndarray:
    """Overlay a status line (detector used, QR count) on *frame*."""
    if result["detected"]:
        status = f"Detector: {result['detector_used']} | QR codes: {result['count']}"
    else:
        status = "Detector: none | No QR detected"

    cv2.putText(
        frame, status, (10, 25),
        FONT, 0.6, (255, 255, 0), 2,
    )
    cv2.putText(
        frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
        FONT, 0.5, (255, 255, 255), 1,
    )
    return frame


def run_live_scan(camera_index: int = CAMERA_INDEX) -> int:
    """Open the webcam and run continuous live QR scanning.

    Workflow
    --------
    1. Open webcam (``camera_index``).
    2. Capture frame.
    3. Detect QR codes (reusing existing detection functions).
    4. Draw bounding boxes + labels (draw_box.py style).
    5. Show status (detector used, QR count).
    6. Display live feed.
    7. Repeat until 'q' is pressed.

    Returns
    -------
    int
        Exit code: ``0`` success, ``1`` camera unavailable.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error("Could not open camera (index=%d).", camera_index)
        print(f"❌ Error: Camera (index {camera_index}) could not be opened.")
        return 1

    print("✅ Camera opened. Press 'q' to quit.")

    # Tracks decoded content of QR codes currently visible on screen.
    seen_codes: set[str] = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Frame capture failed; skipping frame.")
                continue

            try:
                result = detect_qr_frame(frame)
            except Exception as exc:  # noqa: BLE001 - keep stream alive
                logger.exception("Unexpected detection error: %s", exc)
                result = {"detected": False, "count": 0, "detector_used": "none", "detections": []}

            if result["detected"]:
                draw_detections(frame, result)
                current_codes: set[str] = set()
                for det in result["detections"]:
                    data = det["data"]
                    bbox = tuple(det["bbox_tuple"])
                    current_codes.add(data)
                    if data not in seen_codes:
                        print(f"[QR] {data!r}  bbox={bbox}  "
                              f"detector={result['detector_used']}")
                seen_codes = current_codes
            else:
                seen_codes = set()

            draw_status(frame, result)
            cv2.imshow("Live QR Scan", frame)

            if cv2.waitKey(1) & 0xFF == EXIT_KEY:
                print("Exiting live scan.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run_live_scan())