"""
main.py
-------
Week 3 – Member 4: Integration Module
Project: Computer Vision-Based Graphic Tamper Detection for QR Code Phishing Prevention

Orchestrates the full detection pipeline:

    Image Input → Load → Validate → Preprocess → QR Enhancement
                → Detect QR → Tamper Analysis → URL Analysis
                → Risk Assessment → Report Generation → Visualise
                → Save → Summary

Pipeline steps
--------------
1. Load and validate the input image (image_loader).
2. Preprocess: denoise + brightness normalisation (image_enhancement).
   Resize is disabled by default; detection coordinates remain valid in
   the original image's pixel space.
3. Detect QR codes on the preprocessed image (qr_detector).
   Because resize is disabled the bbox coordinates need no remapping
   before visualisation.  If resize is ever re-enabled, call
   remap_to_original() here before step 4.
4. Tamper Analysis: run TamperDetector.analyze() on the ORIGINAL
   (un-preprocessed) BGR image — never the denoised/normalised copy —
   so that overlay/edge/pattern cues are not smoothed away. (Week 3)
5. URL Analysis: run URLAnalyzer.analyze() on the decoded QR payload,
   when QR Detection produced one; all URL scoring/rule logic lives
   inside the URL Analyzer module itself. (Week 4)
6. Risk Assessment: run RiskEngine.assess() using the QR
   DetectionResult and the TamperResult; all weighting/scoring logic
   lives inside RiskEngine. The URL Analysis result, when available, is
   attached as read-only context via ``extra_metadata`` — RiskEngine's
   own component-weight combination logic is untouched and not
   duplicated here. (Week 3 + Week 4)
7. Report Generation: assemble a unified Report via ReportGenerator,
   combining DetectionResult, TamperResult, RiskResult and the URL
   Analysis result. (Week 3 + Week 4)
8. Visualise: draw bounding boxes on the original source image.
9. Print summary.

Week 4 integration notes
-------------------------
URL Analysis now runs immediately after Tamper Analysis, at the spot
previously documented with ``# FUTURE-URL`` markers. Its result (a
``URLResult``) is threaded into ``RiskEngine.assess()`` via
``extra_metadata`` — because ``RiskEngine``'s own ``url_analysis``
component weight remains reserved at ``0.0`` (see
``risk_engine.RiskEngineConfig``), so no risk-scoring logic is
duplicated in this file — and into ``ReportGenerator.generate()`` via
its existing ``url_analysis_result`` parameter. URL Analysis is an
independently-recoverable stage, like Tamper Analysis: a failure never
terminates the pipeline. The Evaluation Framework remains OUT OF SCOPE
for this integration.

Exit codes
----------
0  – pipeline completed successfully
1  – image loading / validation failure
2  – preprocessing failure
3  – QR detection failure
4  – visualisation / output save failure
5  – invalid command-line arguments

Tamper Analysis, Risk Assessment, and Report Generation are treated as
independently-recoverable stages rather than fatal pipeline errors: a
failure in any one of them is logged and gracefully degraded (see the
``run_pipeline`` docstring), and does NOT change the process exit code
on its own, so pre-existing automation that checks exit codes 0–5
keeps working unmodified.

Usage
-----
    python src/integration/main.py <image_path> [output_path]

    image_path   – path to a JPG or PNG image containing one or more QR codes
    output_path  – (optional) destination for the annotated image
                   defaults to  output/annotated_<original_filename>

Examples
--------
    python src/integration/main.py data/test_qr/sample_qr.jpg
    python src/integration/main.py data/test_qr/sample_qr.jpg output/result.jpg
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2

# ---------------------------------------------------------------------------
# Pipeline module imports
# ---------------------------------------------------------------------------

# Member 1 – Image Loader
from src.image_loader.image_loader import (
    ImageLoaderError,
    InvalidFileError,
    UnsupportedFormatError,
    FileNotFoundError as LoaderFileNotFoundError,
    load_image,
)

# Member 2 – QR Detector
from src.qr_detector.qr_detector import (
    detect_qr,
)

# Member 4 – Preprocessing (Week 2)
# resize_target=None by default: detection coordinates remain valid in the
# original image's pixel space so visualisation requires no remapping.
# If resize is ever enabled, call remap_to_original() between step 3 and
# step 4 below.
from src.preprocessing.image_enhancement import (
    PreprocessResult,
    preprocess_for_qr,
    remap_to_original,
)

# Member 3 – Tamper Detection Engine (Week 3)
# TamperDetector.analyze() must be called with the ORIGINAL BGR image (the
# array loaded in step 1), never the preprocessed/denoised copy from step 2
# — see step_tamper_analysis() below.
from src.tamper_analysis.tamper_detector import TamperDetector
from src.tamper_analysis.tamper_result import TamperResult

# Member 2 – Risk Assessment Engine (Week 3)
# create_default_engine() is the module's own documented convenience
# factory for CLI/single-shot call sites such as this one.
from src.risk_assessment.risk_engine import create_default_engine
from src.risk_assessment.risk_result import RiskResult

# Report Generation (Week 3)
# NOTE: report_generator.py's own package location was not specified by the
# Week 3 work order. It is assumed to live alongside its sibling packages
# as ``src.reporting``, consistent with ``src.tamper_analysis`` and
# ``src.risk_assessment``. If your repository places it elsewhere, this is
# the only import line that needs to change.
from src.reporting.report_generator import Report, generate_report

# URL Analyzer (Week 4)
# NOTE: url_analyzer.py's own package location was likewise not specified.
# It is assumed to live alongside its sibling packages as ``src.url_analyzer``,
# consistent with ``src.tamper_analysis.tamper_detector`` / ``tamper_result``
# and ``src.risk_assessment.risk_engine`` / ``risk_result``. If your
# repository places it elsewhere, this is the only import line that needs
# to change.
from src.url_analyzer.url_analyzer import URLAnalyzer
from src.url_analyzer.url_result import URLResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("integration.main")

# ---------------------------------------------------------------------------
# Constants — visual style matches draw_box.py
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path("output")
BOX_COLOUR_BGR     = (0, 255, 0)   # green
LABEL_COLOUR_BGR   = (0, 0, 255)   # red
OVERLAY_ALPHA      = 0.3
BOX_THICKNESS      = 3
FONT               = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE         = 0.7
FONT_THICKNESS     = 2
SEPARATOR          = "=" * 60


# ===========================================================================
# Step helpers
# ===========================================================================

def step_load_image(image_path: str) -> dict:
    """Step 1 — Load and validate the input image via image_loader.

    Parameters
    ----------
    image_path : str
        Path to the source image file.

    Returns
    -------
    dict
        Result dictionary from :func:`image_loader.load_image`, including
        keys ``path``, ``width``, ``height``, ``channels``, ``format``.

    Raises
    ------
    LoaderFileNotFoundError
        File does not exist.
    UnsupportedFormatError
        Not a JPG / PNG.
    InvalidFileError
        Corrupt or unreadable image.
    ImageLoaderError
        Any other loader-level failure.
    """
    logger.info("Loading image: %s", image_path)
    result = load_image(image_path, backend="opencv")
    logger.info(
        "Image loaded — %d × %d px, %d channel(s), format: %s",
        result["width"],
        result["height"],
        result["channels"],
        result["format"].upper(),
    )
    return result


def step_preprocess(image: "np.ndarray") -> tuple[str, PreprocessResult]:
    """Step 2 — Preprocess the BGR image array before detection.

    Applies Gaussian denoise, median denoise, and brightness normalisation
    via :func:`preprocess_for_qr`.  Resize is intentionally disabled
    (``resize_target=None``) so that detector bounding-box coordinates
    remain valid in the original image's pixel space.

    The preprocessed array is written to a temporary file so that
    :func:`qr_detector.detect_qr` (which requires a file path) can consume
    it without modification.  The caller is responsible for deleting the
    temporary file after detection.

    If resize is ever re-enabled here, you must call
    :func:`remap_to_original` on the detection result before visualisation.

    Parameters
    ----------
    image : numpy.ndarray
        BGR uint8 array of the source image, already loaded by
        :func:`step_load_image`.

    Returns
    -------
    preprocessed_path : str
        Absolute path to the temporary file containing the preprocessed image.
    prep_result : PreprocessResult
        Full preprocessing result including timing and spatial_params.

    Raises
    ------
    RuntimeError
        If the temporary file cannot be written.
    ValueError
        Propagated from :func:`preprocess_for_qr` on invalid image data.
    """
    # normalize_brightness=True uses CLAHE on the LAB L-channel.
    # auto_enhance is called in step 3 of qr_detector.detect_qr() — to
    # prevent double CLAHE, ensure auto_enhance is called with
    # try_low_light=False if you add a QR enhancement step here.
    # Currently detect_qr() does not call auto_enhance internally, so
    # there is no double CLAHE risk in the current pipeline.
    prep_result = preprocess_for_qr(
        image,
        resize_target=None,   # IMPORTANT: keep None — see module docstring
        denoise=True,
        normalize=True,
    )

    # Write to a temp file so detect_qr() (file-path API) can consume it.
    suffix = ".jpg"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="qr_prep_")
    os.close(tmp_fd)

    ok = cv2.imwrite(tmp_path, prep_result.processed_image)
    if not ok:
        Path(tmp_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"step_preprocess: cv2.imwrite failed writing temp file '{tmp_path}'."
        )

    logger.info(
        "Preprocessing complete — steps=%s  elapsed=%.1f ms",
        prep_result.processing_steps or ["none"],
        prep_result.elapsed_ms,
    )
    return tmp_path, prep_result


def step_detect_qr(image_path: str) -> dict:
    """Step 3 — Run QR detection via qr_detector.

    Receives the path to the **preprocessed** temporary file so that the
    detector operates on the cleaned image.  Because resize is disabled in
    step 2, the returned bounding-box coordinates are in the same pixel
    space as the original source image and can be drawn directly without
    remapping.

    Parameters
    ----------
    image_path : str
        Absolute path to the preprocessed image file.

    Returns
    -------
    dict
        ``DetectionResult`` from :func:`qr_detector.detect_qr`.
        Structure is unchanged from the Week 1 version.

    Raises
    ------
    FileNotFoundError
        Propagated from qr_detector.
    ValueError
        Propagated from qr_detector (unreadable file).
    """
    logger.info("Running QR detection on preprocessed image.")
    detection_result = detect_qr(image_path)
    logger.info(
        "Detection complete — %d QR code(s) found via %s.",
        detection_result["count"],
        detection_result["detector_used"],
    )
    return detection_result


def step_tamper_analysis(image: "np.ndarray") -> TamperResult:
    """Step 4 — Run graphic tamper analysis via TamperDetector.

    Must be called with the ORIGINAL, un-preprocessed BGR image (the array
    read directly from disk in ``run_pipeline``), **not** the denoised /
    brightness-normalised image produced by :func:`step_preprocess`.
    Denoising can smooth away exactly the edge, overlay, and pattern
    anomalies this stage is designed to detect.

    Parameters
    ----------
    image : numpy.ndarray
        The ORIGINAL BGR image array loaded in step 1.

    Returns
    -------
    TamperResult
        Full tamper-analysis verdict (``tampered``, ``confidence``,
        ``reasons``, plus extended diagnostics).

    Raises
    ------
    ValueError
        Propagated from :meth:`TamperDetector.analyze` for invalid image
        data.
    RuntimeError
        Propagated from :meth:`TamperDetector.analyze` on an unrecoverable
        internal (OpenCV) failure.

    Notes
    -----
    Any exception raised here is caught by the caller (``run_pipeline``),
    which treats Tamper Analysis as an independently-recoverable stage: on
    failure, the pipeline continues with ``tamper_result=None``, which
    ``RiskEngine.assess()`` is documented to accept for backward-compatible,
    QR-only scoring.
    """
    logger.info("Running tamper analysis on the original source image.")
    detector = TamperDetector()
    tamper_result = detector.analyze(image)
    logger.info(
        "Tamper analysis complete — tampered=%s confidence=%.1f%% (%.1f ms)",
        tamper_result.tampered,
        tamper_result.confidence * 100.0,
        tamper_result.analysis_time_ms,
    )
    return tamper_result


def _extract_decoded_payload(detection_result: dict) -> Optional[str]:
    """Return the first non-empty decoded QR payload, or ``None``.

    A single image can contain multiple detected QR codes; the URL
    Analyzer's public API (:meth:`URLAnalyzer.analyze`) takes one URL
    string at a time, so this helper picks the first detection whose
    ``"data"`` field is non-empty to hand to Week 4's URL Analysis stage.

    Parameters
    ----------
    detection_result : dict
        ``DetectionResult`` produced by :func:`step_detect_qr` /
        :func:`remap_to_original`.

    Returns
    -------
    str or None
        The first non-empty decoded payload, or ``None`` if there are no
        detections or none of them decoded to non-empty data.
    """
    for det in detection_result.get("detections", []):
        data = det.get("data")
        if data:
            return data
    return None


def step_url_analysis(detection_result: dict) -> Optional[URLResult]:
    """Step 5 — Analyse the decoded QR payload via the URL Analyzer. (Week 4)

    Runs only when QR Detection produced a non-empty decoded payload (see
    :func:`_extract_decoded_payload`); all URL parsing, scoring and rule
    evaluation lives inside :class:`~src.url_analyzer.url_analyzer.URLAnalyzer`
    — nothing is duplicated here.

    Parameters
    ----------
    detection_result : dict
        ``DetectionResult`` produced by :func:`step_detect_qr` /
        :func:`remap_to_original`.

    Returns
    -------
    URLResult or None
        The URL Analyzer's result, or ``None`` when there is no decoded
        payload to analyse.

    Raises
    ------
    Exception
        Propagated from :meth:`URLAnalyzer.analyze`. The caller
        (``run_pipeline``) treats URL Analysis as an independently
        recoverable stage — on failure, ``url_result`` stays ``None`` and
        the pipeline continues, exactly like Tamper Analysis.
    """
    payload = _extract_decoded_payload(detection_result)
    if payload is None:
        logger.info("URL Analysis skipped — no decoded QR payload available.")
        return None

    logger.info("Running URL analysis on decoded QR payload.")
    analyzer = URLAnalyzer()
    url_result = analyzer.analyze(payload)
    logger.info(
        "URL analysis complete — recommendation=%s score=%d confidence=%.2f%%",
        url_result.recommendation,
        url_result.risk_score,
        url_result.confidence,
    )
    return url_result


def step_risk_assessment(
    detection_result: dict,
    tamper_result: Optional[TamperResult],
    url_result: Optional[URLResult] = None,
) -> RiskResult:
    """Step 6 — Assess overall risk via RiskEngine.

    All weighting and classification logic lives inside
    :class:`~src.risk_assessment.risk_engine.RiskEngine`; this helper only
    orchestrates the call — no risk calculation is duplicated here.

    ``url_result``, when available, is attached as read-only context via
    ``extra_metadata`` rather than fed into scoring: ``RiskEngine``'s own
    ``component_weights["url_analysis"]`` remains reserved at ``0.0`` (see
    ``RiskEngineConfig``), so URL Analysis does not influence
    ``RiskResult.score`` until that weight is intentionally activated
    inside ``risk_engine.py`` itself — this integration does not touch it.

    Parameters
    ----------
    detection_result : dict
        ``DetectionResult`` produced by :func:`step_detect_qr` /
        :func:`remap_to_original`.
    tamper_result : TamperResult, optional
        Result of :func:`step_tamper_analysis`. May be ``None`` if that
        stage failed; ``RiskEngine.assess()`` falls back to QR-only
        scoring in that case.
    url_result : URLResult, optional
        Result of :func:`step_url_analysis`. May be ``None`` if that stage
        failed or was skipped (no decoded payload). When present, its
        ``to_dict()`` is merged into ``RiskResult.metadata["url_analysis"]``
        via ``extra_metadata`` — a key that does not clash with any of
        ``RiskEngine.assess()``'s reserved metadata keys.

    Returns
    -------
    RiskResult
        RiskEngine is fail-safe internally and always returns a valid
        result (never raises for scoring failures), but this call is still
        wrapped by the caller for defensive consistency with the other
        stages.
    """
    logger.info("Running risk assessment.")
    engine = create_default_engine()
    extra_metadata = (
        {"url_analysis": url_result.to_dict()} if url_result is not None else None
    )
    risk_result = engine.assess(
        detection_result=detection_result,
        tamper_result=tamper_result,
        extra_metadata=extra_metadata,
    )
    logger.info(
        "Risk assessment complete — level=%s score=%.4f confidence=%.1f%%",
        risk_result.risk_level.value,
        risk_result.score,
        risk_result.confidence * 100.0,
    )
    return risk_result


def step_generate_report(
    detection_result: dict,
    tamper_result: TamperResult,
    risk_result: RiskResult,
    load_result: dict,
    processing_stats: dict,
    url_result: Optional[URLResult] = None,
) -> Report:
    """Step 7 — Assemble the unified Report via ReportGenerator.

    Passes ``DetectionResult``, ``TamperResult``, ``RiskResult`` and (when
    available) the URL Analysis result into
    :func:`~src.reporting.report_generator.generate_report`; no report
    formatting or status derivation is duplicated in main.py.

    Parameters
    ----------
    detection_result : dict
        ``DetectionResult`` from QR Detection.
    tamper_result : TamperResult
        Result of :func:`step_tamper_analysis`. Required by
        ``ReportGenerator`` — see the ``run_pipeline`` docstring for how a
        missing tamper result is handled (report generation is skipped).
    risk_result : RiskResult
        Result of :func:`step_risk_assessment`.
    load_result : dict
        Result of :func:`step_load_image`, used to populate the report's
        image-information section.
    processing_stats : dict
        Free-form timing dict (e.g. ``preprocessing_time_ms``,
        ``qr_detection_time_ms``) merged into the report's processing
        statistics.
    url_result : URLResult, optional
        Result of :func:`step_url_analysis`. ``ReportGenerator.generate()``
        requires its ``url_analysis_result`` argument to be a
        ``Mapping[str, Any]`` (it calls ``dict(...)`` on it internally),
        so ``url_result.to_dict()`` is passed rather than the raw
        dataclass instance. When ``None``, ``ReportGenerator`` fills in
        its own "not yet available" placeholder section — nothing extra
        is done here.

    Returns
    -------
    Report
        The unified, serialisable report object.

    Raises
    ------
    Exception
        Propagated from ``ReportGenerator.generate`` (e.g.
        ``ReportGenerationError``) for the caller to handle as an
        independently-recoverable stage failure.
    """
    logger.info("Generating unified report.")
    report = generate_report(
        tamper_result=tamper_result,
        risk_result=risk_result,
        qr_detection_result=detection_result,
        image_info=load_result,
        processing_stats=processing_stats,
        url_analysis_result=url_result.to_dict() if url_result is not None else None,
    )
    logger.info("Report generated — id=%s status=%s", report.report_id, report.overall_status)
    return report


def step_visualise(
    image_path: str,
    detection_result: dict,
    output_path: str,
) -> None:
    """Step 8 — Annotate the original source image with bounding boxes.

    Reads the original (un-preprocessed) source image and draws boxes using
    the detection coordinates.  This is safe because resize was disabled in
    step 2, so coordinates are in the original image's pixel space.

    If resize is ever re-enabled in step 2, the caller must pass a
    *remapped* detection result (from :func:`remap_to_original`) instead of
    the raw detection result.

    For each detected QR code draws:

    * A semi-transparent filled rectangle (alpha overlay — draw_box.py style)
    * A solid border rectangle
    * A coordinate label above the box
    * The decoded QR data below the box

    Parameters
    ----------
    image_path : str
        Path to the original (un-annotated, un-preprocessed) source image.
    detection_result : dict
        ``DetectionResult`` dict.  Coordinates must be in the original
        image's pixel space (i.e. no resize was applied, or remap_to_original
        was already called).
    output_path : str
        Destination file path for the annotated image.

    Raises
    ------
    RuntimeError
        If OpenCV cannot read the source image or write the output.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(
            f"Visualisation step failed: OpenCV could not read '{image_path}'."
        )

    for idx, det in enumerate(detection_result["detections"], start=1):
        x, y, w, h = det["bbox_tuple"]

        # Semi-transparent fill (draw_box.py pattern)
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), BOX_COLOUR_BGR, -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, image, 1 - OVERLAY_ALPHA, 0, image)

        # Solid border
        cv2.rectangle(image, (x, y), (x + w, y + h), BOX_COLOUR_BGR, BOX_THICKNESS)

        # Coordinate label above box (clamped so it stays on screen)
        coord_text = f"x={x}, y={y}"
        cv2.putText(
            image,
            coord_text,
            (x, max(y - 10, 10)),
            FONT, FONT_SCALE, LABEL_COLOUR_BGR, FONT_THICKNESS,
        )

        # Decoded data below box
        data_preview = det["data"][:50] + ("…" if len(det["data"]) > 50 else "")
        cv2.putText(
            image,
            f"QR #{idx}: {data_preview}",
            (x, y + h + 20),
            FONT, 0.55, BOX_COLOUR_BGR, 2,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(output_path, image)
    if not success:
        raise RuntimeError(
            f"cv2.imwrite failed — could not save annotated image to '{output_path}'."
        )

    logger.info("Annotated image saved: %s", output_path)


# ===========================================================================
# Console summary
# ===========================================================================

def print_summary(
    image_path: str,
    load_result: dict,
    prep_result: PreprocessResult,
    detection_result: dict,
    output_path: str,
    elapsed_sec: float,
    tamper_result: Optional[TamperResult] = None,
    url_result: Optional[URLResult] = None,
    risk_result: Optional[RiskResult] = None,
    report: Optional[Report] = None,
    report_path: Optional[str] = None,
) -> None:
    """Print a human-readable pipeline summary to stdout.

    The Tamper Analysis, URL Analysis, Risk Assessment, and Report
    Generation sections are printed only when the corresponding stage
    produced a result — each is independently optional so this summary
    degrades gracefully if any stage failed or (for URL Analysis) was
    skipped upstream.
    """
    print(f"\n{SEPARATOR}")
    print("  QR Code Tamper Detection — Pipeline Summary")
    print(SEPARATOR)
    print(f"  Input image   : {image_path}")
    print(f"  Resolution    : {load_result['width']} × {load_result['height']} px")
    print(f"  Format        : {load_result['format'].upper()}")
    print(f"  Preprocess    : {prep_result.processing_steps or ['none']}")
    print(f"  Prep time     : {prep_result.elapsed_ms:.1f} ms")
    print(f"  Detector used : {detection_result['detector_used']}")
    print(SEPARATOR)

    count = detection_result["count"]
    print(f"\n  QR codes detected : {count}")

    if count == 0:
        print("\n  ⚠  No QR codes were found in the image.")
    else:
        for idx, det in enumerate(detection_result["detections"], start=1):
            x, y, w, h = det["bbox_tuple"]
            print(f"\n  ── QR Code #{idx} {'─' * 38}")
            print(f"     Decoded content : {det['data'] or '<empty>'}")
            print(f"     Bounding box    : x={x}, y={y}, w={w}, h={h}")
            print(f"     Corner points   : {det['corner_points']}")

    # ── Tamper Analysis (Week 3) ─────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Tamper Analysis")
    print(SEPARATOR)
    if tamper_result is not None:
        status = "TAMPERED" if tamper_result.tampered else "CLEAN"
        print(f"  Tamper status     : {status}")
        print(f"  Tamper confidence : {tamper_result.confidence:.1%}")
        if tamper_result.reasons:
            print("  Reasons:")
            for reason in tamper_result.reasons:
                print(f"    - {reason}")
    else:
        print("  ⚠  Tamper analysis unavailable (stage failed; see log).")

    # ── URL Analysis (Week 4) ─────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  URL Analysis")
    print(SEPARATOR)
    if url_result is not None:
        print(f"  Analyzed URL      : {url_result.url}")
        print(f"  URL risk score    : {url_result.risk_score}/100")
        print(f"  URL confidence    : {url_result.confidence:.2f}%")
        print(f"  Recommendation    : {url_result.recommendation}")
        if url_result.flags:
            print("  Flags:")
            for flag in url_result.flags:
                print(f"    - {flag}")
    else:
        print("  ⚠  URL analysis unavailable (no decoded payload, or stage failed; see log).")

    # ── Risk Assessment (Week 3) ─────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Risk Assessment")
    print(SEPARATOR)
    if risk_result is not None:
        print(f"  Risk level        : {risk_result.risk_level.value}")
        print(f"  Risk score        : {risk_result.score:.4f}")
        print(f"  Risk confidence   : {risk_result.confidence:.1%}")
        print(f"  Recommendation    : {risk_result.recommendation}")
    else:
        print("  ⚠  Risk assessment unavailable (stage failed; see log).")

    # ── Report Generation (Week 3) ───────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  Report")
    print(SEPARATOR)
    if report is not None:
        print(f"  Report status     : {report.overall_status}")
        if report_path:
            print(f"  Report saved to   : {report_path}")
    else:
        print("  ⚠  Report was not generated (stage failed or skipped; see log).")

    print(f"\n{SEPARATOR}")
    print(f"  Output image saved : {output_path}")
    print(f"  Processing Time    : {elapsed_sec:.3f}s")
    print(f"\n{SEPARATOR}")
    print("  ✅  Pipeline completed successfully")
    print(f"{SEPARATOR}\n")


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(image_path: str, output_path: Optional[str] = None) -> int:
    """Execute the full QR Code Tamper Detection pipeline.

    Pipeline steps
    --------------
    1. Load image           (image_loader)
    2. Preprocess           (image_enhancement — denoise + brightness, no resize)
    3. Detect QR codes      (qr_detector — runs on preprocessed temp file)
    4. Tamper Analysis      (TamperDetector — runs on the ORIGINAL image; Week 3)
    5. URL Analysis         (URLAnalyzer — runs on the decoded QR payload, if any; Week 4)
    6. Risk Assessment      (RiskEngine — consumes DetectionResult + TamperResult,
                             with the URL Analysis result attached as read-only
                             extra_metadata; Week 3 + Week 4)
    7. Report Generation    (ReportGenerator — consumes DetectionResult + TamperResult
                             + RiskResult + URL Analysis result; Week 3 + Week 4)
    8. Visualise            (draws on original source image using raw coordinates)
    9. Print summary

    Coordinates are valid across steps 3–8 because resize is disabled in
    step 2.  If resize is ever re-enabled, :func:`remap_to_original` must
    be called between steps 3 and 8.

    Independent stage recovery (Week 3 + Week 4)
    ----------------------------------------------
    Steps 4–7 each have their own exception handling and are treated as
    *recoverable*, not fatal:

    * If Tamper Analysis (step 4) fails, ``tamper_result`` is set to
      ``None`` and the pipeline continues. ``RiskEngine.assess()``
      accepts ``tamper_result=None`` and falls back to QR-only scoring.
    * If URL Analysis (step 5) fails — or is skipped because QR Detection
      produced no decoded payload — ``url_result`` is set to ``None`` and
      the pipeline continues. Both ``RiskEngine.assess()`` (via
      ``extra_metadata``) and ``ReportGenerator.generate()`` (via
      ``url_analysis_result``) accept ``url_result=None``.
    * If Risk Assessment (step 6) fails, ``risk_result`` is set to
      ``None``; Report Generation is then skipped (it requires a
      ``RiskResult``), but visualisation and the console summary still
      run.
    * If Report Generation (step 7) fails — or was skipped because
      ``tamper_result`` or ``risk_result`` is unavailable — ``report`` is
      set to ``None``; visualisation and the console summary still run,
      per the work order's requirement that a report failure must not
      block those later steps.

    None of these four stages change the function's exit code on their
    own; only failures in steps 1–2, 3, or 8 (image loading, preprocessing,
    QR detection, or visualisation) affect the exit code, preserving the
    original exit-code contract.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_path : str, optional
        Destination for the annotated output image.
        Defaults to ``output/annotated_<filename>``.

    Returns
    -------
    int
        Exit code — ``0`` on success, non-zero on error.
    """
    import numpy as np  # imported locally to keep module-level imports clean

    start_time = time.perf_counter()

    if output_path is None:
        stem        = Path(image_path).stem
        suffix      = Path(image_path).suffix or ".jpg"
        output_path = str(DEFAULT_OUTPUT_DIR / f"annotated_{stem}{suffix}")

    print(f"\n{SEPARATOR}")
    print("  QR Code Tamper Detection Pipeline — Starting")
    print(SEPARATOR)

    # ── Step 1: Load image ───────────────────────────────────────────────────
    print("\n[1/9] Loading image …")
    try:
        load_result = step_load_image(image_path)
    except LoaderFileNotFoundError as exc:
        print(f"\n  ❌  File not found: {exc}")
        print("       Check that the file path is correct and the file exists.")
        return 1
    except UnsupportedFormatError as exc:
        print(f"\n  ❌  Unsupported format: {exc}")
        print("       Only JPG and PNG images are accepted.")
        return 1
    except (InvalidFileError, ImageLoaderError) as exc:
        print(f"\n  ❌  Image could not be loaded: {exc}")
        return 1

    abs_path = load_result["path"]
    print(f"       ✔  Image loaded  ({load_result['width']}×{load_result['height']} px)")

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    # Load the BGR array once and pass it directly to preprocessing,
    # avoiding a second disk read.
    print("\n[2/9] Preprocessing image …")
    preprocessed_path: Optional[str] = None
    try:
        bgr = cv2.imread(abs_path)
        if bgr is None:
            raise RuntimeError(
                f"OpenCV could not read '{abs_path}' for preprocessing."
            )
        preprocessed_path, prep_result = step_preprocess(bgr)
    except (RuntimeError, ValueError) as exc:
        print(f"\n  ❌  Preprocessing error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ❌  Unexpected preprocessing error: {exc}")
        logger.exception("Unexpected error during preprocessing.")
        return 2

    print(
        f"       ✔  Preprocessing complete  "
        f"(steps={prep_result.processing_steps or ['none']}, "
        f"{prep_result.elapsed_ms:.1f} ms)"
    )

    # ── Step 3: Detect QR codes ──────────────────────────────────────────────
    print("\n[3/9] Detecting QR codes …")
    try:
        detection_result = step_detect_qr(preprocessed_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ❌  QR detection error: {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ❌  Unexpected detection error: {exc}")
        logger.exception("Unexpected error during QR detection.")
        return 3
    finally:
        # Always delete the temp file regardless of detection outcome.
        if preprocessed_path and Path(preprocessed_path).exists():
            try:
                Path(preprocessed_path).unlink()
                logger.debug("Removed temp file: %s", preprocessed_path)
            except OSError as exc:
                logger.warning(
                    "Could not remove temp file '%s': %s", preprocessed_path, exc
                )

    # Resize is disabled in step_preprocess → spatial_params is empty →
    # remap_to_original returns a deep copy with coordinates unchanged.
    # Calling it unconditionally makes the pipeline safe even if resize is
    # re-enabled in the future without updating this call site.
    detection_result = remap_to_original(detection_result, prep_result)

    count = detection_result["count"]
    print(
        f"       ✔  {count} QR code(s) detected  "
        f"(detector: {detection_result['detector_used']})"
    )

    if count == 0:
        print("\n  ℹ  No QR codes found — annotated image will be saved without boxes.")

    # ── Step 4: Tamper Analysis (Week 3) ─────────────────────────────────────
    # Uses the ORIGINAL image array (`bgr`, loaded for step 2 above) — NOT
    # `preprocessed_path` — so that denoising never masks tamper cues.
    # Independently recoverable: on failure, tamper_result stays None and
    # the pipeline continues (RiskEngine.assess() supports this).
    print("\n[4/9] Running tamper analysis …")
    tamper_result: Optional[TamperResult] = None
    try:
        tamper_result = step_tamper_analysis(bgr)
        print(
            f"       ✔  Tamper analysis complete  "
            f"(tampered={tamper_result.tampered}, "
            f"confidence={tamper_result.confidence:.1%})"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ⚠  Tamper analysis failed, continuing without it: {exc}")
        logger.exception("Recoverable error during tamper analysis.")

    # ── Step 5: URL Analysis (Week 4) ────────────────────────────────────────
    # Runs on the decoded QR payload (if any), immediately after Tamper
    # Analysis and before Risk Assessment. Independently recoverable: on
    # failure — or when there is no decoded payload — url_result stays
    # None and the pipeline continues (RiskEngine.assess() and
    # ReportGenerator.generate() both accept url_result=None).
    print("\n[5/9] Running URL analysis …")
    url_result: Optional[URLResult] = None
    try:
        url_result = step_url_analysis(detection_result)
        if url_result is not None:
            print(
                f"       ✔  URL analysis complete  "
                f"(recommendation={url_result.recommendation}, "
                f"score={url_result.risk_score})"
            )
        else:
            print("       ℹ  No decoded QR payload — URL analysis skipped.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ⚠  URL analysis failed, continuing without it: {exc}")
        logger.exception("Recoverable error during URL analysis.")

    # ── Step 6: Risk Assessment (Week 3 + Week 4) ────────────────────────────
    # Consumes DetectionResult and TamperResult; RiskEngine performs all
    # weighting internally — no risk calculation is duplicated here.
    # url_result, when available, is attached as read-only context via
    # extra_metadata (see step_risk_assessment docstring) rather than fed
    # into scoring, since RiskEngine's url_analysis component weight is
    # still reserved at 0.0.
    # Independently recoverable: on failure, risk_result stays None.
    print("\n[6/9] Running risk assessment …")
    risk_result: Optional[RiskResult] = None
    try:
        risk_result = step_risk_assessment(detection_result, tamper_result, url_result)
        print(
            f"       ✔  Risk assessment complete  "
            f"(level={risk_result.risk_level.value}, score={risk_result.score:.4f})"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ⚠  Risk assessment failed, continuing without it: {exc}")
        logger.exception("Recoverable error during risk assessment.")

    # ── Step 7: Report Generation (Week 3 + Week 4) ──────────────────────────
    # Consumes DetectionResult, TamperResult, RiskResult and (when
    # available) the URL Analysis result; no report formatting is
    # duplicated here. Requires both tamper_result and risk_result — if
    # either is unavailable, generation is skipped rather than fed
    # partial/invalid data, and the pipeline still continues on to
    # visualisation and the console summary.
    print("\n[7/9] Generating report …")
    report: Optional[Report] = None
    report_path: Optional[str] = None
    if tamper_result is None or risk_result is None:
        print("       ⚠  Skipping report generation (missing tamper or risk result).")
    else:
        try:
            elapsed_so_far_ms = (time.perf_counter() - start_time) * 1000.0
            report = step_generate_report(
                detection_result,
                tamper_result,
                risk_result,
                load_result,
                processing_stats={
                    "preprocessing_time_ms": prep_result.elapsed_ms,
                    "total_pipeline_time_ms": elapsed_so_far_ms,
                },
                url_result=url_result,
            )
            stem = Path(image_path).stem
            report_path = str(DEFAULT_OUTPUT_DIR / f"report_{stem}.json")
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(report.to_json(), encoding="utf-8")
            print(f"       ✔  Report generated  (status={report.overall_status})")
            print(f"       ✔  Report saved → {report_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ⚠  Report generation failed, continuing without it: {exc}")
            logger.exception("Recoverable error during report generation.")
            report = None
            report_path = None

    # ── Step 8: Visualise detections ─────────────────────────────────────────
    # Draws on abs_path (original source image).  Coordinates are valid
    # because no resize was applied (spatial_params is empty).
    print("\n[8/9] Generating annotated image …")
    try:
        step_visualise(abs_path, detection_result, output_path)
    except RuntimeError as exc:
        print(f"\n  ❌  Visualisation failed: {exc}")
        return 4
    except OSError as exc:
        print(f"\n  ❌  Could not save output image: {exc}")
        return 4
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ❌  Unexpected visualisation error: {exc}")
        logger.exception("Unexpected error during visualisation.")
        return 4

    print(f"       ✔  Output image saved → {output_path}")

    # ── Step 9: Print summary ────────────────────────────────────────────────
    print("\n[9/9] Generating detection summary …")
    elapsed = time.perf_counter() - start_time
    print_summary(
        image_path, load_result, prep_result,
        detection_result, output_path, elapsed,
        tamper_result=tamper_result,
        url_result=url_result,
        risk_result=risk_result,
        report=report,
        report_path=report_path,
    )

    return 0


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    """Command-line entry point for the QR Code Tamper Detection pipeline.

    Usage::

        python src/integration/main.py <image_path> [output_path]

    Arguments
    ---------
    image_path : str
        Path to the input image (JPG or PNG).
    output_path : str, optional
        Where to save the annotated result image.

    Exit codes
    ----------
    0  – pipeline completed successfully
    1  – image loading / validation failure
    2  – preprocessing failure
    3  – QR detection failure
    4  – visualisation / output save failure
    5  – invalid command-line arguments
    """
    if len(sys.argv) < 2:
        print(
            "\nUsage: python src/integration/main.py <image_path> [output_path]\n"
            "\n"
            "  image_path   – path to a JPG or PNG image file\n"
            "  output_path  – (optional) where to save the annotated image\n"
            "\n"
            "Example:\n"
            "  python src/integration/main.py data/test_qr/sample_qr.jpg\n"
            "  python src/integration/main.py data/test_qr/sample_qr.jpg "
            "output/result.jpg\n"
        )
        sys.exit(5)

    image_path:  str           = sys.argv[1]
    output_path: Optional[str] = sys.argv[2] if len(sys.argv) >= 3 else None

    sys.exit(run_pipeline(image_path, output_path))


if __name__ == "__main__":
    main()