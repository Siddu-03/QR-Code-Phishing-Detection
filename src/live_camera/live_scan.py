"""
live_scan.py
=============
Live webcam QR scanning module.
Week 2 – final production version.
Week 3 update — Tamper Analysis and Risk Assessment integrated in-line.

Reuses the existing project pipeline:
    - src.qr_detector.qr_detector    : detect_qr_opencv, detect_qr_pyzbar,
                                        convert_coordinates
    - src.qr_detector.qr_enhancement : auto_enhance() — applied to every
                                        frame before QR detection
    - src.preprocessing.image_enhancement : preprocess_for_qr() — optional
                                        denoising pass applied BEFORE
                                        auto_enhance() (controlled by
                                        ENABLE_PREPROCESSING flag)
    - src.tamper_analysis.tamper_detector : TamperDetector.analyze() — run
                                        on the raw frame once a QR code is
                                        detected (see AnalysisCache)
    - src.risk_assessment.risk_engine : RiskEngine.assess() — run
                                        immediately after Tamper Analysis,
                                        consuming both DetectionResult and
                                        TamperResult

Frame pipeline (per captured frame)
------------------------------------
    raw frame
        → [preprocess_for_qr()]          ← optional (ENABLE_PREPROCESSING)
        → auto_enhance()
        → detect_qr_frame()
        → draw_detections() / draw_status()
        → [TamperDetector.analyze() on the RAW frame]   ← only if QR found
        → [RiskEngine.assess()]                         ← only if QR found
        → draw_security_overlay()
        → cv2.imshow()

This module intentionally does NOT perform URL Analysis, report
generation, JSON/Markdown export, or evaluation-framework statistics.
Those remain the responsibility of the desktop image pipeline; see
``analyze_security()`` below for the documented Week 4 URL Analyzer
insertion point.

Console output — event-driven, not frame-driven
------------------------------------------------
The console reports *state changes*, not frames. Per distinct decoded
QR payload:

    [QR DETECTED] 'payload'
    [ANALYSIS] Running Tamper Analysis...
    [RISK] SAFE / SUSPICIOUS / HIGH_RISK
    [MONITORING] No further output while this QR remains visible.
    ...
    [QR LOST] 'payload' — cache expired after 0.7s without detection.

While a payload's ``AnalysisCache`` entry is alive — which continuous
detection maintains indefinitely, since every successful decode pushes
its expiration deadline ``CACHE_TIMEOUT_SECONDS`` further out — nothing
further is printed for it, no matter how many frames go by. Expiration
is judged by elapsed wall-clock time, not a frame count, so it behaves
identically at any frame rate and across camera sources (webcam, USB,
RTSP/IP stream, video file). Multiple simultaneously visible QR codes
are each tracked and announced independently. ``logger.info``/
``logger.warning`` calls alongside these prints are unchanged and
remain available for DEBUG-level troubleshooting; they fire at the
same "new payload" event, never per-frame.

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
    python -m src.live_camera.live_scan --enable-preprocessing
    python -m src.live_camera.live_scan --disable-preprocessing
    python -m src.live_camera.live_scan --camera-index 1
    python -m src.live_camera.live_scan --camera-source "http://192.168.1.5:8080/video"
    python -m src.live_camera.live_scan --camera-source path/to/video.mp4

Camera acquisition
------------------
Frame acquisition (laptop webcam, USB webcam, IP/MJPEG camera stream,
or local video file) is handled entirely by ``CameraStream`` — see
``camera_stream.py`` in this package. This module never talks to
``cv2.VideoCapture`` directly; it only ever calls
``CameraStream.get_latest_frame()``, which always returns the newest
frame available and silently discards any stale buffered ones. This is
what eliminates the multi-second latency previously seen on IP-camera
sources (see camera_stream.py's module docstring for why that
buffering happened and how the fix works).

Preprocessing defaults to the module-level ``ENABLE_PREPROCESSING``
constant (``False``) when neither flag is given; the flags let it be
toggled per-run without editing this file. The scanner prints its
resolved mode ("Preprocessing: ENABLED"/"DISABLED") at startup.

Controls
--------
    q  -  quit the live scan
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Low-latency camera acquisition — see camera_stream.py. live_scan.py no
# longer talks to cv2.VideoCapture directly; all frame acquisition
# (webcam, USB camera, IP/MJPEG stream, or video file) is delegated to
# CameraStream, which discards stale buffered frames automatically.
from src.live_camera.camera_stream import CameraSource, CameraStream

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

# Week 3 modules — Tamper Analysis and Risk Assessment (reused, not rewritten)
from src.tamper_analysis.tamper_detector import TamperDetector
from src.tamper_analysis.tamper_result import TamperResult
from src.risk_assessment.risk_engine import RiskEngine
from src.risk_assessment.risk_result import RiskLevel, RiskResult

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
# CAMERA_SOURCE may additionally be a string: an IP-camera HTTP/MJPEG
# stream URL (e.g. "http://192.168.1.5:8080/video") or a local video
# file path. See CameraStream in camera_stream.py for details. When
# None, CAMERA_INDEX is used instead (see _resolve_camera_source()).
CAMERA_SOURCE: Optional[Union[int, str]] = None
BOX_COLOUR_BGR   = (0, 255, 0)   # green — matches draw_box.py
LABEL_COLOUR_BGR = (0, 0, 255)   # red   — matches draw_box.py
OVERLAY_ALPHA    = 0.3
BOX_THICKNESS    = 3
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.7
FONT_THICKNESS   = 2
EXIT_KEY         = ord("q")

# ---------------------------------------------------------------------------
# Tamper Analysis / Risk Assessment — cache configuration
# ---------------------------------------------------------------------------
# A fresh TamperDetector.analyze() + RiskEngine.assess() pass is only run
# when the decoded QR payload changes (or the cache has expired). While the
# same payload remains on screen, the cached TamperResult / RiskResult are
# reused so the overlay can refresh every frame without re-running the
# (comparatively expensive) analysis stages on each one.
#
# CACHE_TIMEOUT_SECONDS: wall-clock seconds with no successful detection
# of a given payload before its cache entry is dropped. A small tolerance
# absorbs brief detection flicker (motion blur, glare, momentary
# occlusion, a dropped network frame) without forcing a full re-analysis
# the instant the code reappears. Timestamp-based (rather than a
# consecutive-frame counter) so behaviour is identical regardless of
# camera frame rate, network latency/jitter, dropped frames, or camera
# source (webcam, USB, RTSP/IP stream, video file) — 0.7s of absence is
# 0.7s of absence whether the pipeline is running at 5 fps or 60 fps.
CACHE_TIMEOUT_SECONDS: float = 0.7

# Risk-level → BGR colour used for the overlay text. RiskResult.risk_level
# is one of RiskLevel.SAFE / SUSPICIOUS / HIGH_RISK (see risk_result.py).
# There is no separate "MEDIUM" / "CRITICAL" tier in the current
# RiskLevel enum, so SUSPICIOUS is rendered in amber (a SAFE→HIGH_RISK
# midpoint) and HIGH_RISK in red.
RISK_LEVEL_COLOURS_BGR = {
    RiskLevel.SAFE:       (0, 200, 0),     # green
    RiskLevel.SUSPICIOUS: (0, 165, 255),   # orange/amber
    RiskLevel.HIGH_RISK:  (0, 0, 255),     # red
}
UNAVAILABLE_COLOUR_BGR = (160, 160, 160)  # grey — analysis unavailable

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
# Tamper Analysis / Risk Assessment
# ===========================================================================
#
# Insertion point for future Week 4 integration
# ----------------------------------------------
# The Week 4 URL Analyzer slots in immediately AFTER Tamper Analysis and
# BEFORE Risk Assessment, turning the pipeline into:
#
#     Camera → Preprocessing → Enhancement → Detection → Tamper
#            → URL Analysis → Risk Assessment → Overlay → Display
#
# Concretely, that will mean: inside `analyze_security()` below, call the
# (future) `URLAnalyzer.analyze(decoded_payload)` right after
# `tamper_detector.analyze(raw_frame)` succeeds, and pass its result to
# `risk_engine.assess(...)` alongside `tamper_result` (the engine's
# `assess()` signature already anticipates additional optional inputs).
# No other function in this module should need to change.

# Reuse a single detector / engine instance for the lifetime of the process
# (constructing these per-frame would be needless allocation and defeats
# any internal setup cost amortisation).
_TAMPER_DETECTOR = TamperDetector()
_RISK_ENGINE = RiskEngine()


@dataclass
class CacheEntry:
    """Cached Tamper Analysis / Risk Assessment outcome for one QR payload.

    Attributes
    ----------
    tamper_result : TamperResult, optional
        Cached Tamper Analysis output, or ``None`` if analysis failed.
    risk_result : RiskResult, optional
        Cached Risk Assessment output, or ``None`` if assessment failed.
    last_seen_time : float
        ``time.monotonic()`` timestamp of the most recent frame in which
        this payload was successfully detected. Updated every time the
        payload is seen (see :meth:`AnalysisCache.refresh`); expiration
        is judged purely from elapsed wall-clock time since this value,
        never from a frame count — so it behaves identically regardless
        of camera frame rate, network latency/jitter, dropped frames, or
        camera source (webcam, USB, RTSP/IP stream, video file).
    """

    tamper_result: Optional[TamperResult] = None
    risk_result: Optional[RiskResult] = None
    last_seen_time: float = 0.0


class AnalysisCache:
    """Holds the most recent Tamper Analysis / Risk Assessment outcome
    for every decoded QR payload currently (or recently) on screen.

    The cache lets the live loop avoid re-running `TamperDetector.analyze()`
    and `RiskEngine.assess()` on every frame — both are re-executed only
    when a *new* decoded QR payload appears. Keying by decoded content
    (rather than holding a single slot) means multiple simultaneously
    visible QR codes are each analysed once and reused independently —
    one code entering or leaving the frame never invalidates another
    code's cached result.

    Expiration is timestamp-based: each entry is dropped once
    ``time.monotonic() - entry.last_seen_time`` exceeds ``timeout_seconds``
    (default :data:`CACHE_TIMEOUT_SECONDS`). This is deliberately *not* a
    consecutive-frame counter — a frame count implicitly assumes a
    roughly constant frame rate, which does not hold across sources (a
    laptop webcam at 30 fps vs. an RTSP/IP stream that stalls, jitters,
    or briefly drops frames over the network vs. a video file decoded
    faster or slower than real time). Using wall-clock elapsed time
    instead means "0.7 seconds of absence" means the same thing — and
    survives the same amount of real-world occlusion/flicker — no matter
    which of those sources is in use or how its frame rate varies from
    moment to moment.
    """

    def __init__(self, timeout_seconds: float = CACHE_TIMEOUT_SECONDS) -> None:
        self.entries: dict[str, CacheEntry] = {}
        self.timeout_seconds = timeout_seconds

    def get(self, payload: str) -> Optional[CacheEntry]:
        """Return the cached entry for *payload*, or ``None`` if absent."""
        return self.entries.get(payload)

    def is_valid_for(self, payload: str) -> bool:
        """True if a cache entry already exists for *payload*."""
        return payload in self.entries

    def store(
        self,
        payload: str,
        tamper_result: Optional[TamperResult],
        risk_result: Optional[RiskResult],
        now: Optional[float] = None,
    ) -> CacheEntry:
        """Cache a freshly computed result for *payload* and return it.

        *now*, if given, is the ``time.monotonic()`` timestamp to record
        as this entry's initial ``last_seen_time``; defaults to the
        current time when omitted.
        """
        entry = CacheEntry(
            tamper_result=tamper_result,
            risk_result=risk_result,
            last_seen_time=now if now is not None else time.monotonic(),
        )
        self.entries[payload] = entry
        return entry

    def refresh(self, seen_payloads: set, now: Optional[float] = None) -> list:
        """Update last-seen timestamps for *seen_payloads*; expire the rest.

        For every payload currently in *seen_payloads*, its entry's
        ``last_seen_time`` is bumped to *now* (so a continuously visible
        QR code's cache stays alive indefinitely — every detection frame
        pushes its expiration deadline back out by ``timeout_seconds``).
        Any entry whose payload is NOT in *seen_payloads* and whose
        elapsed time since ``last_seen_time`` has exceeded
        ``timeout_seconds`` is dropped.

        Parameters
        ----------
        seen_payloads : set
            Decoded QR payloads detected on the current frame.
        now : float, optional
            ``time.monotonic()`` timestamp to treat as "now"; defaults
            to the current time when omitted (tests may pass an
            explicit value for deterministic timing).

        Returns
        -------
        list
            Payloads expired (dropped) by this call.
        """
        if now is None:
            now = time.monotonic()

        expired: list = []
        for payload, entry in list(self.entries.items()):
            if payload in seen_payloads:
                entry.last_seen_time = now
            elif (now - entry.last_seen_time) >= self.timeout_seconds:
                del self.entries[payload]
                expired.append(payload)
        return expired

    def clear(self) -> None:
        self.entries.clear()


def analyze_security(
    raw_frame: np.ndarray,
    detection_result: dict,
    payload: str,
) -> Tuple[Optional[TamperResult], Optional[RiskResult]]:
    """Run Tamper Analysis followed immediately by Risk Assessment.

    Both stages are fail-safe from the caller's perspective: any internal
    exception is caught and logged here so a single bad frame can never
    terminate the live scanner. On failure the corresponding result is
    ``None``, and the caller (the overlay) is expected to render
    "Unavailable" in its place.

    Parameters
    ----------
    raw_frame : numpy.ndarray
        The **original**, unprocessed camera frame (never the
        preprocessed/enhanced frame) covering the moment of detection.
    detection_result : dict
        Output of :func:`detect_qr_frame` (the `DetectionResult` contract
        consumed by `RiskEngine.assess`).
    payload : str
        Decoded content of the (primary) detected QR code — used only for
        log messages here; caching is handled by the caller.

    Returns
    -------
    tuple[TamperResult | None, RiskResult | None]
    """
    tamper_result: Optional[TamperResult] = None
    risk_result: Optional[RiskResult] = None

    try:
        tamper_result = _TAMPER_DETECTOR.analyze(raw_frame)
    except Exception as exc:  # noqa: BLE001 — keep stream alive
        logger.warning("Tamper Analysis failed for %r: %s", payload, exc)

    try:
        risk_result = _RISK_ENGINE.assess(
            detection_result,
            tamper_result=tamper_result,
        )
    except Exception as exc:  # noqa: BLE001 — keep stream alive
        logger.warning("Risk Assessment failed for %r: %s", payload, exc)

    return tamper_result, risk_result


# ===========================================================================
# Security overlay
# ===========================================================================

def draw_security_overlay(
    frame: np.ndarray,
    entry: Optional[CacheEntry],
    is_cached: bool,
) -> np.ndarray:
    """Draw the Tamper Analysis / Risk Assessment status block.

    Rendered every frame from whatever *entry* currently holds (freshly
    computed or reused from the cache), so the overlay stays smooth even
    on frames where no new analysis ran. Displays: Tamper Status, Tamper
    Confidence, Risk Level, Risk Score, Recommendation, LIVE/CACHED
    indicator, and Processing Time. Falls back to a grey "Unavailable"
    line for any stage that failed or hasn't run (``entry is None``).

    Parameters
    ----------
    frame : numpy.ndarray
        BGR frame to annotate (modified in place and returned).
    entry : CacheEntry, optional
        The (primary) QR payload's current cache entry to render, or
        ``None`` if nothing has been analysed / is still on screen.
    is_cached : bool
        ``True`` if this frame's values were reused from a prior analysis
        rather than freshly computed.
    """
    y = 55
    line_height = 24
    source_label = "CACHED" if is_cached else "LIVE"

    tamper_result = entry.tamper_result if entry is not None else None
    risk_result = entry.risk_result if entry is not None else None

    if tamper_result is None:
        cv2.putText(
            frame, f"Tamper: Unavailable ({source_label})", (10, y),
            FONT, 0.55, UNAVAILABLE_COLOUR_BGR, 2,
        )
        y += line_height
    else:
        tr = tamper_result
        tamper_status = "TAMPERED" if tr.tampered else "CLEAN"
        tamper_colour = (0, 0, 255) if tr.tampered else (0, 200, 0)
        cv2.putText(
            frame,
            f"Tamper: {tamper_status} (conf={tr.confidence:.0%}) [{source_label}]",
            (10, y), FONT, 0.55, tamper_colour, 2,
        )
        y += line_height

    if risk_result is None:
        cv2.putText(
            frame, "Risk: Unavailable", (10, y),
            FONT, 0.55, UNAVAILABLE_COLOUR_BGR, 2,
        )
        y += line_height
        cv2.putText(
            frame, "Recommendation: Unavailable", (10, y),
            FONT, 0.5, UNAVAILABLE_COLOUR_BGR, 1,
        )
    else:
        rr = risk_result
        risk_colour = RISK_LEVEL_COLOURS_BGR.get(rr.risk_level, UNAVAILABLE_COLOUR_BGR)
        cv2.putText(
            frame,
            f"Risk: {rr.risk_level.display_label} (score={rr.score:.1f})",
            (10, y), FONT, 0.55, risk_colour, 2,
        )
        y += line_height
        cv2.putText(
            frame, f"Recommendation: {rr.recommendation}", (10, y),
            FONT, 0.5, risk_colour, 1,
        )
        y += line_height
        cv2.putText(
            frame, f"Processing: {rr.processing_time_ms:.1f} ms", (10, y),
            FONT, 0.5, (200, 200, 200), 1,
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
    6. If a QR code is detected: run Tamper Analysis (on the **raw**
       frame, never the preprocessed/enhanced one) and Risk Assessment
       for the primary decoded payload — but only when that payload is
       new or the analysis cache has expired; otherwise reuse the
       cached `TamperResult` / `RiskResult`.
    7. Display the QR status overlay and the Tamper/Risk security overlay
       (the security overlay renders every frame from whatever is
       currently cached, live or reused).
    8. Show live feed.
    9. Repeat until 'q' is pressed.

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
def run_live_scan(
    camera_index: int = CAMERA_INDEX,
    enable_preprocessing: bool = ENABLE_PREPROCESSING,
    camera_source: Optional[CameraSource] = None,
) -> int:
    """Open the camera stream and run continuous live QR scanning.

    Workflow (per frame)
    --------------------
    1. Capture raw frame (via ``CameraStream`` — always the newest frame
       available, never a stale buffered one; see camera_stream.py).
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
    6. If a QR code is detected: run Tamper Analysis (on the **raw**
       frame, never the preprocessed/enhanced one) and Risk Assessment
       for the primary decoded payload — but only when that payload is
       new or the analysis cache has expired; otherwise reuse the
       cached `TamperResult` / `RiskResult`.
    7. Display the QR status overlay and the Tamper/Risk security overlay
       (the security overlay renders every frame from whatever is
       currently cached, live or reused).
    8. Show live feed.
    9. Repeat until 'q' is pressed.

    Parameters
    ----------
    camera_index : int
        OpenCV camera index. Defaults to ``0`` (primary webcam). Ignored
        if *camera_source* is given.
    enable_preprocessing : bool
        When ``True``, runs denoising before enhancement.  Adds ~1.5–3 ms
        per frame at 720p.  Defaults to ``False``.
    camera_source : int or str, optional
        Explicit camera source: an OpenCV device index, an IP-camera
        HTTP/MJPEG stream URL, or a local video file path. When
        provided, this takes precedence over *camera_index*. When
        ``None`` (the default), *camera_index* is used, preserving
        the original webcam-only call signature.

    Returns
    -------
    int
        ``0`` on clean exit, ``1`` if the camera could not be opened.
    """
    resolved_source: CameraSource = (
        camera_source if camera_source is not None else camera_index
    )

    stream = CameraStream(resolved_source)
    stream.start()

    if not stream.is_opened():
        logger.error("Could not open camera source (%r).", resolved_source)
        print(f"❌ Error: Camera source ({resolved_source!r}) could not be opened.")
        stream.stop()
        return 1

    prep_state = "ENABLED" if enable_preprocessing else "DISABLED"
    logger.info("Live scan started — preprocessing: %s", prep_state)
    print(f"✅ Camera opened. Preprocessing: {prep_state}. Press 'q' to quit.")

    # Detection-flicker tolerance and "have we already announced this
    # payload" state both live in `cache` now (see AnalysisCache below):
    # a payload is considered "new" for console purposes exactly when it
    # has no live cache entry, which is also exactly when a fresh Tamper
    # Analysis / Risk Assessment pass is required. A single dropped
    # detection frame (motion blur, glare) does NOT remove the cache
    # entry (see CACHE_TIMEOUT_SECONDS), so it no longer triggers a
    # duplicate "[QR DETECTED]" print either — one payload, one
    # announcement, for as long as it stays within its visibility
    # timeout.

    # Tamper Analysis / Risk Assessment cache — keyed by decoded QR
    # payload, so each distinct code is analysed once and reused for as
    # long as it (or any other code) remains on screen (see AnalysisCache).
    cache = AnalysisCache()

    # Payload currently driving the security overlay. Kept "sticky" across
    # brief detection flicker: it only changes when a new payload becomes
    # primary, or is cleared once its cache entry actually expires.
    displayed_payload: Optional[str] = None

    try:
        while True:
            frame = stream.get_latest_frame()
            if frame is None:
                if not stream.is_running():
                    logger.error(
                        "Camera stream stopped unexpectedly (source=%r).",
                        resolved_source,
                    )
                    print("❌ Camera stream ended unexpectedly.")
                    break
                # No frame yet (e.g. right after start()) — brief wait
                # instead of a hot spin, then retry.
                time.sleep(0.01)
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
            #
            # ── Step 6: Tamper Analysis + Risk Assessment (cached) ────────────
            # The security overlay reports on one QR code (the primary /
            # first valid detection) at a time, but the underlying
            # AnalysisCache is keyed by decoded content, so if multiple
            # distinct QR codes are visible each is analysed at most once
            # and reused independently — one code entering/leaving the
            # frame never invalidates another's cached result. All
            # decoded codes are still boxed and console-logged (see below),
            # regardless of which one is currently "primary".
            is_cached = True
            entry: Optional[CacheEntry] = None

            if result["detected"]:
                draw_detections(frame, result)

            # Empty/None decoded strings (OpenCV sometimes detects a QR's
            # position but fails to decode its payload) are ignored for
            # analysis purposes entirely — no Tamper Analysis, no Risk
            # Assessment, no caching, no console print. The bounding box
            # is still drawn above so the user can see *something* was
            # detected.
            valid_detections = [
                det for det in result["detections"] if det["data"]
            ] if result["detected"] else []

            current_codes: set[str] = {det["data"] for det in valid_detections}

            # ── Step 6: Tamper Analysis + Risk Assessment (event-driven) ──────
            # Every distinct decoded payload currently on screen is handled
            # independently and analysed at most once: a payload with no
            # live cache entry (first-ever appearance, OR a reappearance
            # after its previous entry actually expired) gets exactly one
            # Tamper Analysis + Risk Assessment pass and one console
            # announcement here. A payload that already has a live cache
            # entry is skipped entirely — no re-analysis, no re-print —
            # whether it's the primary on-screen code or an additional one,
            # so multiple simultaneously visible QR codes are each tracked
            # and reported on their own independent timeline.
            newly_analyzed: set[str] = set()

            for det in valid_detections:
                payload = det["data"]
                if payload in cache.entries:
                    continue  # already announced and still alive — silent

                print(f"[QR DETECTED] {payload!r}")
                print("[ANALYSIS] Running Tamper Analysis...")

                # Uses the ORIGINAL camera frame, never the preprocessed/
                # enhanced one (tamper cues live in the raw pixel data).
                tamper_result, risk_result = analyze_security(frame, result, payload)
                cache.store(payload, tamper_result, risk_result)
                newly_analyzed.add(payload)

                if tamper_result is not None:
                    logger.info(
                        "[Tamper] %r — %s (confidence=%.2f)",
                        payload,
                        "TAMPERED" if tamper_result.tampered else "clean",
                        tamper_result.confidence,
                    )
                else:
                    logger.warning(
                        "[Tamper] %r — analysis unavailable", payload
                    )

                if risk_result is not None:
                    print(f"[RISK] {risk_result.risk_level.value}")
                    logger.info(
                        "[Risk] %r — %s (score=%.1f) — %s",
                        payload,
                        risk_result.risk_level.value,
                        risk_result.score,
                        risk_result.recommendation,
                    )
                else:
                    print("[RISK] unavailable")
                    logger.warning(
                        "[Risk] %r — assessment unavailable", payload
                    )

                print(
                    "[MONITORING] No further output while this QR remains "
                    "visible."
                )

            if valid_detections:
                # Only the *primary* (first) valid detection drives the
                # on-screen security overlay — every decoded code is boxed
                # and independently analysed/cached above, but the overlay
                # has room for one status block at a time.
                primary_payload = valid_detections[0]["data"]
                displayed_payload = primary_payload

                entry = cache.get(primary_payload)
                is_cached = primary_payload not in newly_analyzed

            # Update last-seen timestamps for every payload decoded this
            # frame (including, when no valid QR was decoded at all,
            # updating none); drop any entry whose payload has been
            # absent for CACHE_TIMEOUT_SECONDS of *wall-clock* time. A
            # small tolerance absorbs brief detection flicker (motion
            # blur, glare, momentary occlusion, a dropped network frame)
            # without forcing a full re-analysis the instant a code
            # reappears — this is also exactly what keeps a
            # continuously-visible QR code's cache alive indefinitely:
            # every frame it's decoded on pushes its expiration deadline
            # CACHE_TIMEOUT_SECONDS further into the future, regardless
            # of how many (or how few) frames arrive per second.
            expired = cache.refresh(current_codes)
            if expired:
                for lost_payload in expired:
                    print(
                        f"[QR LOST] {lost_payload!r} — cache expired after "
                        f"{CACHE_TIMEOUT_SECONDS:.1f}s without detection."
                    )
                logger.info(
                    "[Cache] Expired after %.1fs without a QR code: %s",
                    CACHE_TIMEOUT_SECONDS,
                    expired,
                )
            if displayed_payload in expired:
                displayed_payload = None

            if entry is None and displayed_payload is not None:
                # No new/primary detection this frame — keep showing the
                # last analysed payload's overlay for as long as its
                # cache entry survives (reused, i.e. CACHED).
                entry = cache.get(displayed_payload)
                is_cached = True

            # ── Step 7: Status + security overlays, Step 8: display ───────────
            draw_status(frame, result, enhancement_technique, enable_preprocessing)
            draw_security_overlay(frame, entry, is_cached)
            cv2.imshow("Live QR Scan", frame)

            if cv2.waitKey(1) & 0xFF == EXIT_KEY:
                print("Exiting live scan.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()

    return 0


def _parse_args(argv: Optional[list] = None) -> "argparse.Namespace":
    """Parse command-line arguments for the live scanner.

    ``--enable-preprocessing`` / ``--disable-preprocessing`` let the
    denoising pass (see :data:`ENABLE_PREPROCESSING` and the module
    docstring) be toggled at runtime without editing this file. When
    neither flag is given, the module-level ``ENABLE_PREPROCESSING``
    default is used.
    """
    parser = argparse.ArgumentParser(
        description="QR Shield — live webcam QR scanner.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help=f"OpenCV camera index (default: {CAMERA_INDEX}). Ignored if "
             "--camera-source is given.",
    )
    parser.add_argument(
        "--camera-source",
        type=str,
        default=None,
        help=(
            "Camera source: an IP-camera HTTP/MJPEG stream URL (e.g. "
            "'http://192.168.1.5:8080/video') or a local video file "
            "path. Takes precedence over --camera-index when given."
        ),
    )
    prep_group = parser.add_mutually_exclusive_group()
    prep_group.add_argument(
        "--enable-preprocessing",
        dest="enable_preprocessing",
        action="store_true",
        default=None,
        help=(
            "Run the denoising preprocessing pass (Gaussian + median "
            "blur, ~1.5-3 ms/frame at 720p) before enhancement. Useful "
            "for noisy sensors / poor-quality USB cameras."
        ),
    )
    prep_group.add_argument(
        "--disable-preprocessing",
        dest="enable_preprocessing",
        action="store_false",
        default=None,
        help="Skip the denoising preprocessing pass (lowest-latency path).",
    )
    args = parser.parse_args(argv)
    if args.enable_preprocessing is None:
        args.enable_preprocessing = ENABLE_PREPROCESSING
    return args


if __name__ == "__main__":
    _args = _parse_args()
    raise SystemExit(
        run_live_scan(
            camera_index=_args.camera_index,
            enable_preprocessing=_args.enable_preprocessing,
            camera_source=_args.camera_source,
        )
    )