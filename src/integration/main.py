"""
main.py
-------
Week 4 - Member 4: Integration Module
Project: QR Code Tamper Detection using Computer Vision

Orchestrates the full detection pipeline:
    Image → Load → Validate → Detect QR → Visualise → Save → Summary

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

import os
import sys
import time
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party pipeline modules (do NOT duplicate their logic)
# ---------------------------------------------------------------------------
import cv2

# Member 1 – Image Loader
from src.image_loader.image_loader import (
    load_image,
    ImageLoaderError,
    InvalidFileError,
    UnsupportedFormatError,
    FileNotFoundError as LoaderFileNotFoundError,
)

# Member 2 – QR Detector
from src.qr_detector.qr_detector import (
    detect_qr,
    save_results_json,
)

# Member 3 – Visualisation  (draw_box.py exposes its logic as a script;
#             we replicate the visualisation call using cv2 as it does,
#             but driven by real detection data rather than hard-coded values)
import src.visualization.draw_box as draw_box  # imported for awareness; actual cv2 calls mirror its approach

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
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR  = Path("output")
BOX_COLOUR_BGR      = (0, 255, 0)   # green  — matches draw_box.py
LABEL_COLOUR_BGR    = (0, 0, 255)   # red    — matches draw_box.py
OVERLAY_ALPHA       = 0.3            # semi-transparent fill opacity
BOX_THICKNESS       = 3
FONT                = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE          = 0.7
FONT_THICKNESS      = 2
SEPARATOR           = "=" * 60


# ===========================================================================
# Step helpers
# ===========================================================================

def step_load_image(image_path: str) -> dict:
    """
    Step 1 – Load and validate the input image via image_loader.

    Parameters
    ----------
    image_path : str
        Path to the source image file.

    Returns
    -------
    dict
        Result dictionary from :func:`image_loader.load_image`.

    Raises
    ------
    LoaderFileNotFoundError  – file does not exist
    UnsupportedFormatError   – not a JPG / PNG
    InvalidFileError         – corrupt or unreadable image
    ImageLoaderError         – any other loader-level failure
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


def step_detect_qr(image_path: str) -> dict:
    """
    Step 2 – Run QR detection via qr_detector.

    Parameters
    ----------
    image_path : str
        Absolute path to the validated image file.

    Returns
    -------
    dict
        ``DetectionResult`` from :func:`qr_detector.detect_qr`.

    Raises
    ------
    FileNotFoundError  – propagated from qr_detector
    ValueError         – propagated from qr_detector (unreadable file)
    """
    logger.info("Running QR detection on: %s", image_path)
    detection_result = detect_qr(image_path)
    logger.info(
        "Detection complete — %d QR code(s) found via %s",
        detection_result["count"],
        detection_result["detector_used"],
    )
    return detection_result


def step_visualise(
    image_path: str,
    detection_result: dict,
    output_path: str,
) -> None:
    """
    Step 3 – Annotate the image with bounding boxes and labels.

    Mirrors the logic of draw_box.py but iterates over all real detections
    instead of operating on hard-coded coordinates.  For each detected QR
    code it draws:

    * A semi-transparent filled rectangle (alpha composite, as in draw_box.py)
    * A solid border rectangle
    * A coordinate label above the box (same style as draw_box.py)
    * The decoded QR data below the box

    Parameters
    ----------
    image_path : str
        Path to the original (un-annotated) image.
    detection_result : dict
        ``DetectionResult`` dict from :func:`qr_detector.detect_qr`.
    output_path : str
        Destination file path for the annotated image.

    Raises
    ------
    RuntimeError  – if OpenCV cannot read the source image or write the output.
    """
    # Load a fresh BGR copy for annotation (draw_box.py uses cv2.imread)
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(
            f"Visualisation step failed: OpenCV could not read '{image_path}'."
        )

    for idx, det in enumerate(detection_result["detections"], start=1):
        x, y, w, h = det["bbox_tuple"]   # use bbox_tuple as draw_box.py does

        # ── Semi-transparent fill (draw_box.py pattern) ────────────────────
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), BOX_COLOUR_BGR, -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, image, 1 - OVERLAY_ALPHA, 0, image)

        # ── Solid border (draw_box.py pattern) ─────────────────────────────
        cv2.rectangle(image, (x, y), (x + w, y + h), BOX_COLOUR_BGR, BOX_THICKNESS)

        # ── Coordinate label above box (draw_box.py pattern) ───────────────
        coord_text = f"x={x}, y={y}"
        cv2.putText(
            image,
            coord_text,
            (x, max(y - 10, 10)),   # clamp so text doesn't go off-screen
            FONT,
            FONT_SCALE,
            LABEL_COLOUR_BGR,
            FONT_THICKNESS,
        )

        # ── Decoded QR data below box ───────────────────────────────────────
        data_preview = det["data"][:50] + ("…" if len(det["data"]) > 50 else "")
        cv2.putText(
            image,
            f"QR #{idx}: {data_preview}",
            (x, y + h + 20),
            FONT,
            0.55,
            BOX_COLOUR_BGR,
            2,
        )

    # ── Save annotated image ────────────────────────────────────────────────
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
    detection_result: dict,
    output_path: str,
    elapsed_sec: float,
) -> None:
    """Print a human-readable detection summary to stdout."""
    print(f"\n{SEPARATOR}")
    print("  QR Code Tamper Detection — Pipeline Summary")
    print(SEPARATOR)
    print(f"  Input image   : {image_path}")
    print(f"  Resolution    : {load_result['width']} × {load_result['height']} px")
    print(f"  Format        : {load_result['format'].upper()}")
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
    """
    Execute the full QR Code Tamper Detection pipeline.

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
    start_time = time.perf_counter()

    # ── Resolve output path ─────────────────────────────────────────────────
    if output_path is None:
        stem     = Path(image_path).stem
        suffix   = Path(image_path).suffix or ".jpg"
        output_path = str(DEFAULT_OUTPUT_DIR / f"annotated_{stem}{suffix}")

    print(f"\n{SEPARATOR}")
    print("  QR Code Tamper Detection Pipeline — Starting")
    print(SEPARATOR)

    # ── Step 1: Load image ───────────────────────────────────────────────────
    print("\n[1/4] Loading image …")
    try:
        load_result = step_load_image(image_path)
    except LoaderFileNotFoundError as exc:
        print(f"\n  ❌  Error: {exc}")
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
    print(f"       ✔  Image loaded successfully  "
          f"({load_result['width']}×{load_result['height']} px)")

    # ── Step 2: Detect QR codes ──────────────────────────────────────────────
    print("\n[2/4] Detecting QR codes …")
    try:
        detection_result = step_detect_qr(abs_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  ❌  QR detection error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ❌  Unexpected detection error: {exc}")
        logger.exception("Unexpected error during QR detection.")
        return 2

    count = detection_result["count"]
    print(f"       ✔  {count} QR code(s) detected  "
          f"(detector: {detection_result['detector_used']})")

    if count == 0:
        print("\n  ℹ  No QR codes found — annotated image will be saved without boxes.")

    # ── Step 3: Visualise detections ─────────────────────────────────────────
    print("\n[3/4] Generating annotated image …")
    try:
        step_visualise(abs_path, detection_result, output_path)
    except RuntimeError as exc:
        print(f"\n  ❌  Visualisation failed: {exc}")
        return 3
    except OSError as exc:
        print(f"\n  ❌  Could not save output image: {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ❌  Unexpected visualisation error: {exc}")
        logger.exception("Unexpected error during visualisation.")
        return 3

    print(f"       ✔  Output image saved successfully → {output_path}")

    # ── Step 4: Print summary ────────────────────────────────────────────────
    print("\n[4/4] Generating detection summary …")
    elapsed = time.perf_counter() - start_time
    print_summary(image_path, load_result, detection_result, output_path, elapsed)

    return 0


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    """
    Command-line entry point for the QR Code Tamper Detection pipeline.

    Usage::

        python src/integration/main.py <image_path> [output_path]

    Arguments
    ---------
    image_path   : str
        Path to the input image (JPG or PNG).
    output_path  : str, optional
        Where to save the annotated result image.

    Exit codes
    ----------
    0  – pipeline completed successfully
    1  – image loading / validation failure
    2  – QR detection failure
    3  – visualisation / output save failure
    4  – invalid command-line arguments
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
            "  python src/integration/main.py data/test_qr/sample_qr.jpg output/result.jpg\n"
        )
        sys.exit(4)

    image_path: str        = sys.argv[1]
    output_path: Optional[str] = sys.argv[2] if len(sys.argv) >= 3 else None

    exit_code = run_pipeline(image_path, output_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
