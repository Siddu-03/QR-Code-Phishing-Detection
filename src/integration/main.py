"""
main.py
-------
Week 2 – Member 4: Integration Module
Project: Computer Vision-Based Graphic Tamper Detection for QR Code Phishing Prevention

Orchestrates the full detection pipeline:

    Image Input → Load → Validate → Preprocess → QR Enhancement
                → Detect QR → Visualise → Save → Summary

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
4. Visualise: draw bounding boxes on the original source image.
5. Print summary.

Exit codes
----------
0  – pipeline completed successfully
1  – image loading / validation failure
2  – preprocessing failure
3  – QR detection failure
4  – visualisation / output save failure
5  – invalid command-line arguments

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


def step_visualise(
    image_path: str,
    detection_result: dict,
    output_path: str,
) -> None:
    """Step 4 — Annotate the original source image with bounding boxes.

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
) -> None:
    """Print a human-readable pipeline summary to stdout."""
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

    print(f"\n  Output image saved : {output_path}")
    print(f"  Pipeline elapsed   : {elapsed_sec:.3f}s")
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
    1. Load image          (image_loader)
    2. Preprocess          (image_enhancement — denoise + brightness, no resize)
    3. Detect QR codes     (qr_detector — runs on preprocessed temp file)
    4. Visualise           (draws on original source image using raw coordinates)
    5. Print summary

    Coordinates are valid across steps 3–4 because resize is disabled in
    step 2.  If resize is ever re-enabled, :func:`remap_to_original` must
    be called between steps 3 and 4.

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
    print("\n[1/5] Loading image …")
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
    print("\n[2/5] Preprocessing image …")
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
    print("\n[3/5] Detecting QR codes …")
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

    # ── Step 4: Visualise detections ─────────────────────────────────────────
    # Draws on abs_path (original source image).  Coordinates are valid
    # because no resize was applied (spatial_params is empty).
    print("\n[4/5] Generating annotated image …")
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

    # ── Step 5: Print summary ────────────────────────────────────────────────
    print("\n[5/5] Generating detection summary …")
    elapsed = time.perf_counter() - start_time
    print_summary(
        image_path, load_result, prep_result,
        detection_result, output_path, elapsed,
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