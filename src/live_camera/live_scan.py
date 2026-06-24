"""
live_scan.py
=============
Live webcam QR scanning module.
Week 2 – final production version.

Reuses the existing project pipeline:
    - src.qr_detector.qr_detector    : detect_qr_opencv, detect_qr_pyzbar,
                                        convert_coordinates
    - src.qr_detector.qr_enhancement : auto_enhance() — applied to every
                                        frame before QR detection
    - src.preprocessing.image_enhancement : preprocess_for_qr() — optional
                                        denoising pass applied BEFORE
                                        auto_enhance() (controlled by
                                        ENABLE_PREPROCESSING flag)

Frame pipeline (per captured frame)
------------------------------------
    raw frame
        → [preprocess_for_qr()]          ← optional (ENABLE_PREPROCESSING)
        → auto_enhance()
        → detect_qr_frame()
        → draw_detections() / draw_status()
        → cv2.imshow()

Preprocessing on live frames — design decisions
-----------------------------------------------
Resize is always disabled (``resize_target=None``) for live frames.
Camera output is a stable resolution; resizing would shift bounding-box
coordinates relative to the displayed original frame, causing misaligned
boxes.

When ENABLE_PREPROCESSING is True:

    * ``denoise=True``    — Gaussian (k=3) + median (k=3) denoising.
      Because ``auto_enhance(try_blur=True)`` also applies Gaussian blur
      internally, enabling denoise here disables ``try_blur`` in the
      auto_enhance call to prevent double-blurring.

    * ``normalize=False`` — brightness normalisation is disabled for live
      frames.  ``auto_enhance(try_low_light=True)`` already applies CLAHE
      on the LAB L-channel.  Running ``normalize_brightness`` (which also
      applies CLAHE) before it causes double CLAHE and over-enhances
      contrast.  Brightness adaptation is left entirely to auto_enhance.

Performance impact at 720p (approximate)
-----------------------------------------
    ENABLE_PREPROCESSING = False   0 ms overhead
    ENABLE_PREPROCESSING = True    ~1.5–3 ms overhead (Gaussian + median)

At 30 fps (33 ms budget) this is < 10 % overhead.  For higher-resolution
cameras or stricter latency requirements, disable preprocessing.

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
    convert_coordinates,
    detect_qr_opencv,
    detect_qr_pyzbar,
)

# QR enhancement — applied to every frame before detection
from src.qr_detector.qr_enhancement import auto_enhance

# Preprocessing — optional denoising pass applied BEFORE auto_enhance
from src.preprocessing.image_enhancement import preprocess_for_qr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("live_camera.live_scan")

# ---------------------------------------------------------------------------
# Constants — visual style matches draw_box.py
# ---------------------------------------------------------------------------
CAMERA_INDEX     = 0
BOX_COLOUR_BGR   = (0, 255, 0)   # green — matches draw_box.py
LABEL_COLOUR_BGR = (0, 0, 255)   # red   — matches draw_box.py
OVERLAY_ALPHA    = 0.3
BOX_THICKNESS    = 3
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.7
FONT_THICKNESS   = 2
EXIT_KEY         = ord("q")

# ---------------------------------------------------------------------------
# Preprocessing toggle
# ---------------------------------------------------------------------------
# Set True to run preprocess_for_qr() on every frame before auto_enhance().
#
# When True:  denoise=True, normalize=False (see module docstring for why).
#             try_blur=False is passed to auto_enhance to prevent double
#             Gaussian blurring.
# When False: frames go directly to auto_enhance() — lowest latency path.
#
# Recommended: False for clean cameras / real-time use.
#              True  for noisy sensors or poor-quality USB cameras.
ENABLE_PREPROCESSING: bool = False


# ===========================================================================
# Detection
# ===========================================================================

def detect_qr_frame(frame: np.ndarray) -> dict:
    """Run QR detection on a single in-memory frame.

    Mirrors the OpenCV-first / pyzbar-fallback pipeline of
    :func:`qr_detector.detect_qr`, operating directly on a frame array to
    avoid disk I/O per frame.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image array captured from the webcam.

    Returns
    -------
    dict
        Same structure as ``qr_detector.detect_qr``'s return value
        (``detected``, ``count``, ``detector_used``, ``image_info``,
        ``detections``).
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
                "data":          data,
                "confidence":    None,
                "corner_points": corner_points,
                "bbox_tuple":    bbox_tuple,
                "bbox_dict":     bbox_dict,
            }
        )

    img_h, img_w = frame.shape[:2]

    return {
        "detected":      len(detections) > 0,
        "count":         len(detections),
        "detector_used": detector_used,
        "image_info":    {"width": img_w, "height": img_h},
        "detections":    detections,
    }


# ===========================================================================
# Drawing
# ===========================================================================

def draw_detections(frame: np.ndarray, result: dict) -> np.ndarray:
    """Draw bounding boxes and labels on *frame* in draw_box.py style.

    For each detection draws:
        * A semi-transparent filled rectangle (alpha overlay)
        * A solid border rectangle
        * A coordinate label above the box
        * The decoded QR data below the box

    Detection coordinates are always in the raw-frame pixel space because
    resize is never applied to live frames.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR frame to annotate (modified in place and returned).
    result : dict
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
        data         = det["data"] or "<empty>"
        data_preview = data[:50] + ("…" if len(data) > 50 else "")
        cv2.putText(
            frame, f"QR #{idx}: {data_preview}", (x, y + h + 20),
            FONT, 0.55, BOX_COLOUR_BGR, 2,
        )

    return frame


def draw_status(
    frame: np.ndarray,
    result: dict,
    enhancement_technique: str = "none",
    preprocessing_active: bool = False,
) -> np.ndarray:
    """Overlay a status line on *frame*.

    Displays: detector used, QR code count, enhancement technique, and
    whether preprocessing is active.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR frame to annotate.
    result : dict
        Output of :func:`detect_qr_frame`.
    enhancement_technique : str
        Name of the technique selected by ``auto_enhance()``.
    preprocessing_active : bool
        True when :func:`preprocess_for_qr` ran on this frame.
    """
    prep_label = "ON" if preprocessing_active else "OFF"

    if result["detected"]:
        status = (
            f"Detector: {result['detector_used']} | "
            f"QR codes: {result['count']} | "
            f"Enhance: {enhancement_technique} | "
            f"Preprocess: {prep_label}"
        )
    else:
        status = (
            f"Detector: none | No QR detected | "
            f"Enhance: {enhancement_technique} | "
            f"Preprocess: {prep_label}"
        )

    cv2.putText(
        frame, status, (10, 25),
        FONT, 0.6, (255, 255, 0), 2,
    )
    cv2.putText(
        frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
        FONT, 0.5, (255, 255, 255), 1,
    )
    return frame


# ===========================================================================
# Main loop
# ===========================================================================

def run_live_scan(
    camera_index: int = CAMERA_INDEX,
    enable_preprocessing: bool = ENABLE_PREPROCESSING,
) -> int:
    """Open the webcam and run continuous live QR scanning.

    Workflow (per frame)
    --------------------
    1. Capture raw frame.
    2. [Optional] Preprocess: Gaussian + median denoise.
       (resize always disabled; normalize always disabled to avoid double CLAHE)
    3. Enhance via ``auto_enhance()``.
       - ``try_blur`` is set to ``not enable_preprocessing`` to prevent
         double Gaussian blurring when preprocessing is active.
       - ``try_low_light`` is always True (brightness correction handled
         entirely by auto_enhance; normalize is off in preprocessing).
    4. Detect QR codes on the enhanced frame.
    5. Draw boxes on the **raw captured** frame (coordinates are valid
       because no resize was applied).
    6. Display status overlay.
    7. Show live feed.
    8. Repeat until 'q' is pressed.

    Parameters
    ----------
    camera_index : int
        OpenCV camera index.  Defaults to ``0`` (primary webcam).
    enable_preprocessing : bool
        When ``True``, runs denoising before enhancement.  Adds ~1.5–3 ms
        per frame at 720p.  Defaults to ``False``.

    Returns
    -------
    int
        ``0`` on clean exit, ``1`` if the camera could not be opened.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error("Could not open camera (index=%d).", camera_index)
        print(f"❌ Error: Camera (index {camera_index}) could not be opened.")
        return 1

    prep_state = "ENABLED" if enable_preprocessing else "DISABLED"
    logger.info("Live scan started — preprocessing: %s", prep_state)
    print(f"✅ Camera opened. Preprocessing: {prep_state}. Press 'q' to quit.")

    # Tracks decoded content of QR codes currently visible on screen,
    # used to suppress duplicate console prints.
    seen_codes: set[str] = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Frame capture failed; skipping frame.")
                continue

            # ── Step 2: Optional preprocessing ────────────────────────────────
            # resize_target=None  — never resize live frames (coordinate safety)
            # normalize=False     — auto_enhance handles brightness via CLAHE;
            #                       enabling normalize would apply CLAHE twice.
            if enable_preprocessing:
                try:
                    prep = preprocess_for_qr(
                        frame,
                        resize_target=None,
                        denoise=True,
                        normalize=False,
                    )
                    work_frame = prep.processed_image
                except Exception as exc:  # noqa: BLE001 — keep stream alive
                    logger.warning(
                        "Preprocessing failed, using raw frame: %s", exc
                    )
                    work_frame = frame
            else:
                work_frame = frame

            # ── Step 3: Enhancement ───────────────────────────────────────────
            # try_rotation=False : rotation changes canvas dimensions and would
            #   shift bbox coordinates out of sync with the displayed frame.
            # try_blur : disabled when preprocessing ran Gaussian denoise to
            #   prevent applying Gaussian blur twice.
            # try_low_light : always True — brightness normalisation delegated
            #   entirely to auto_enhance (normalize=False in preprocessing).
            try:
                enh = auto_enhance(
                    work_frame,
                    try_rotation=False,
                    try_low_light=True,
                    try_blur=not enable_preprocessing,   # avoid double Gaussian
                    try_contrast=True,
                )
                enhanced_frame      = enh.enhanced_image
                enhancement_technique = enh.technique
            except Exception as exc:  # noqa: BLE001 — keep stream alive
                logger.warning("Enhancement failed, using work frame: %s", exc)
                enhanced_frame        = work_frame
                enhancement_technique = "none"

            # ── Step 4: Detection ─────────────────────────────────────────────
            try:
                result = detect_qr_frame(enhanced_frame)
            except Exception as exc:  # noqa: BLE001 — keep stream alive
                logger.exception("Unexpected detection error: %s", exc)
                result = {
                    "detected":      False,
                    "count":         0,
                    "detector_used": "none",
                    "detections":    [],
                }
                enhancement_technique = "none"

            # ── Step 5: Draw boxes on the raw frame ───────────────────────────
            # Coordinates are in enhanced-frame space, which is identical to
            # raw-frame space because no resize was applied (try_rotation=False
            # in auto_enhance, resize_target=None in preprocessing).
            if result["detected"]:
                draw_detections(frame, result)
                current_codes: set[str] = set()
                for det in result["detections"]:
                    data = det["data"]
                    bbox = tuple(det["bbox_tuple"])
                    current_codes.add(data)
                    if data not in seen_codes:
                        print(
                            f"[QR] {data!r}  bbox={bbox}  "
                            f"detector={result['detector_used']}"
                        )
                seen_codes = current_codes
            else:
                seen_codes = set()

            # ── Steps 6–7: Status overlay and display ─────────────────────────
            draw_status(frame, result, enhancement_technique, enable_preprocessing)
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