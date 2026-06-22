"""
image_enhancement.py
====================
Week 2 – Member 4: Preprocessing Module
Project: Computer Vision-Based Graphic Tamper Detection for QR Code Phishing Prevention

Purpose
-------
Sits **before** ``qr_enhancement.py`` (and therefore before ``qr_detector.py``)
in the full pipeline::

    Image Input → Preprocessing → QR Enhancement → QR Detection
               → Tamper Analysis → Visualization

Every public function accepts a BGR ``numpy.ndarray`` (the native format
used by ``cv2.imread`` and every other module in this project) and returns a
BGR ``numpy.ndarray``, keeping this module a transparent drop-in within the
existing pipeline.

Functions
---------
``resize_image(image, target_size, keep_aspect_ratio)``
    Normalise image dimensions while preserving aspect ratio via
    letterboxing or stretching.

``convert_grayscale(image, as_bgr)``
    Convert to grayscale for downstream vision processing.  Optionally
    re-wraps the single channel into a 3-channel BGR image so that the
    rest of the pipeline (which expects BGR) receives a compatible array.

``gaussian_denoise(image, kernel_size, sigma)``
    Reduce continuous sensor / camera noise with a Gaussian blur kernel.

``median_denoise(image, kernel_size)``
    Remove salt-and-pepper noise with a median filter — preserves edges
    better than Gaussian for binary-like QR module patterns.

``normalize_brightness(image, target_mean, clip_limit)``
    Standardise per-channel mean brightness so that images captured
    under varying lighting enter the QR enhancement step consistently.

``preprocess_pipeline(image, config)``
    Compose the above steps in the recommended order with a single call.
    Steps are controlled via a :class:`PreprocessConfig` dataclass and
    logged individually.

Integration example
-------------------
::

    from src.preprocessing.image_enhancement import preprocess_pipeline, PreprocessConfig
    from src.qr_detector.qr_enhancement import auto_enhance
    from src.qr_detector.qr_detector import detect_qr_opencv, detect_qr_pyzbar

    config = PreprocessConfig(resize_target=(800, 800), denoise_gaussian=True)
    prep   = preprocess_pipeline(raw_bgr_image, config)
    enh    = auto_enhance(prep.processed_image)
    result = detect_qr_opencv(enh.enhanced_image)

Compatibility
-------------
* OpenCV ≥ 4.5   (cv2)
* NumPy ≥ 1.21   (numpy)
* Python ≥ 3.9
* No changes to qr_detector.py, qr_enhancement.py, or live_scan.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging  — mirrors qr_detector.py / qr_enhancement.py format exactly
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("preprocessing.image_enhancement")

# ---------------------------------------------------------------------------
# Type aliases  — mirror qr_detector.py conventions
# ---------------------------------------------------------------------------
BgrArray = np.ndarray   # shape (H, W, 3), dtype uint8, BGR channel order
GrayArray = np.ndarray  # shape (H, W),    dtype uint8


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class PreprocessResult:
    """Wraps a preprocessed image with a full audit trail of applied steps.

    Attributes
    ----------
    processed_image:
        The final BGR ``numpy.ndarray`` ready for ``qr_enhancement.auto_enhance``
        or direct use in ``qr_detector.detect_qr_opencv`` / ``detect_qr_pyzbar``.
    processing_steps:
        Ordered list of step names that were actually executed, e.g.
        ``["resize", "grayscale", "gaussian_denoise", "normalize_brightness"]``.
        Steps that were skipped (disabled in :class:`PreprocessConfig`) are
        **not** included.
    elapsed_ms:
        Total wall-clock time for the full pipeline in milliseconds.
    original_shape:
        ``(H, W, C)`` tuple of the input image before any transformation.
    final_shape:
        ``(H, W, C)`` tuple of ``processed_image``.
    step_elapsed_ms:
        Per-step timing in milliseconds, keyed by step name.
    """

    processed_image: BgrArray
    processing_steps: list[str]
    elapsed_ms: float
    original_shape: Tuple[int, ...]
    final_shape: Tuple[int, ...]
    step_elapsed_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class PreprocessConfig:
    """Declarative configuration for :func:`preprocess_pipeline`.

    All boolean flags default to ``False``; callers opt into each step
    explicitly so the pipeline is transparent and reproducible.

    Attributes
    ----------
    resize_target:
        ``(width, height)`` in pixels.  Set to ``None`` to skip resizing.
        Aspect ratio is preserved by default; see ``keep_aspect_ratio``.
    keep_aspect_ratio:
        If ``True`` (default), pad the resized image with black pixels
        (letterboxing) to reach exactly ``resize_target``.  If ``False``,
        stretch the image to fill the target exactly.
    convert_to_gray:
        Convert the image to grayscale.  The result is re-wrapped as a
        3-channel BGR array (all three channels equal) so downstream
        modules that expect BGR continue to work.
    denoise_gaussian:
        Apply Gaussian blur for continuous noise suppression.
    gaussian_kernel_size:
        Odd integer ≥ 1.  Kernel size for :func:`gaussian_denoise`.
    gaussian_sigma:
        Standard deviation for the Gaussian kernel.  ``0`` lets OpenCV
        compute it automatically from ``gaussian_kernel_size``.
    denoise_median:
        Apply median filter for salt-and-pepper noise.
    median_kernel_size:
        Odd integer ≥ 1.  Aperture for :func:`median_denoise`.
    normalize_brightness:
        Standardise per-channel mean brightness via CLAHE on the L channel
        of LAB colour space.  Complements ``qr_enhancement.enhance_low_light_qr``
        by normalising images that are *too bright* or *too dark* before the
        enhancement step selects its technique.
    brightness_target_mean:
        Target mean pixel value in [0, 255].  Defaults to 128 (mid-grey).
    brightness_clip_limit:
        CLAHE clip limit for brightness normalisation.  Defaults to 2.0
        (matches the CLAHE parameters used in ``qr_enhancement.py``).
    """

    # --- Resize ---
    resize_target: Optional[Tuple[int, int]] = None   # (width, height)
    keep_aspect_ratio: bool = True

    # --- Grayscale ---
    convert_to_gray: bool = False

    # --- Gaussian denoise ---
    denoise_gaussian: bool = False
    gaussian_kernel_size: int = 3
    gaussian_sigma: float = 0.0

    # --- Median denoise ---
    denoise_median: bool = False
    median_kernel_size: int = 3

    # --- Brightness normalisation ---
    normalize_brightness: bool = False
    brightness_target_mean: float = 128.0
    brightness_clip_limit: float = 2.0


# ===========================================================================
# Individual preprocessing functions
# ===========================================================================

def resize_image(
    image: BgrArray,
    target_size: Tuple[int, int],
    *,
    keep_aspect_ratio: bool = True,
) -> BgrArray:
    """Resize *image* to *target_size*, optionally preserving aspect ratio.

    When *keep_aspect_ratio* is ``True`` the image is scaled so that neither
    dimension exceeds *target_size* and then letterboxed (padded with black
    pixels) to fill the canvas exactly.  This guarantees the output always
    has the precise dimensions requested without distorting the QR finder
    patterns.

    When *keep_aspect_ratio* is ``False`` the image is stretched directly
    to *target_size* using bilinear interpolation.

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.
    target_size:
        ``(width, height)`` in pixels.  Both dimensions must be ≥ 1.
    keep_aspect_ratio:
        ``True`` (default) — letterbox; ``False`` — stretch.

    Returns
    -------
    numpy.ndarray
        Resized BGR array of shape ``(target_size[1], target_size[0], 3)``.

    Raises
    ------
    ValueError
        If *image* is ``None``, not a 3-channel array, or *target_size*
        contains non-positive values.
    """
    _validate_bgr_image(image, caller="resize_image")

    tw, th = target_size
    if tw < 1 or th < 1:
        raise ValueError(
            f"resize_image: target_size values must be ≥ 1, got {target_size}."
        )

    src_h, src_w = image.shape[:2]

    if not keep_aspect_ratio:
        resized = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
        logger.debug(
            "resize_image: stretched %dx%d → %dx%d",
            src_w, src_h, tw, th,
        )
        return resized

    # --- Aspect-ratio-preserving path (letterbox) ----------------------------
    scale = min(tw / src_w, th / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create black canvas and paste scaled image at top-left
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    pad_y = (th - new_h) // 2
    pad_x = (tw - new_w) // 2
    canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = scaled

    logger.debug(
        "resize_image: letterboxed %dx%d → %dx%d (pad x=%d, y=%d, scale=%.4f)",
        src_w, src_h, tw, th, pad_x, pad_y, scale,
    )
    return canvas


def convert_grayscale(
    image: BgrArray,
    *,
    as_bgr: bool = True,
) -> BgrArray:
    """Convert *image* to grayscale.

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.
    as_bgr:
        If ``True`` (default) the grayscale channel is replicated across
        all three BGR channels, returning a ``(H, W, 3)`` array.  This
        keeps the array shape compatible with every downstream function in
        the project that expects a 3-channel image.

        If ``False`` a true single-channel ``(H, W)`` array is returned.
        **Note:** passing a single-channel array to ``qr_enhancement.auto_enhance``
        or ``qr_detector.detect_qr_opencv`` without first converting back to
        3-channel BGR will raise an error; use ``as_bgr=False`` only when
        you intend to handle the conversion yourself.

    Returns
    -------
    numpy.ndarray
        Grayscale image as a 3-channel BGR array (``as_bgr=True``) or a
        1-channel array (``as_bgr=False``).

    Raises
    ------
    ValueError
        If *image* is not a valid BGR array.
    """
    _validate_bgr_image(image, caller="convert_grayscale")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not as_bgr:
        logger.debug("convert_grayscale: returning single-channel (H, W) array.")
        return gray

    bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    logger.debug("convert_grayscale: returning 3-channel BGR-wrapped grayscale.")
    return bgr_gray


def gaussian_denoise(
    image: BgrArray,
    kernel_size: int = 3,
    sigma: float = 0.0,
) -> BgrArray:
    """Apply a Gaussian blur to reduce continuous (sensor / camera) noise.

    Gaussian blurring smooths high-frequency noise while preserving the
    overall structure of QR finder patterns and module edges.  Use
    :func:`median_denoise` instead for images with isolated bright/dark
    pixel artefacts (salt-and-pepper noise).

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.
    kernel_size:
        Odd integer ≥ 1.  Larger values increase smoothing at the cost of
        edge sharpness.  Defaults to ``3``.
    sigma:
        Standard deviation of the Gaussian kernel.  ``0`` (default) lets
        OpenCV compute an appropriate value from *kernel_size*.

    Returns
    -------
    numpy.ndarray
        Denoised BGR array with the same shape as *image*.

    Raises
    ------
    ValueError
        If *image* is invalid or *kernel_size* is even or less than 1.
    """
    _validate_bgr_image(image, caller="gaussian_denoise")
    _validate_odd_kernel(kernel_size, caller="gaussian_denoise")

    denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    logger.debug(
        "gaussian_denoise: kernel=%dx%d  sigma=%.2f",
        kernel_size, kernel_size, sigma,
    )
    return denoised


def median_denoise(
    image: BgrArray,
    kernel_size: int = 3,
) -> BgrArray:
    """Apply a median filter to remove salt-and-pepper noise.

    The median filter replaces each pixel with the median value of its
    neighbourhood, which is highly effective for isolated impulse noise
    and preserves the sharp module boundaries of QR codes better than
    a Gaussian blur.

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.
    kernel_size:
        Odd integer ≥ 1.  Must be odd (OpenCV requirement).  Defaults to ``3``.

    Returns
    -------
    numpy.ndarray
        Denoised BGR array with the same shape as *image*.

    Raises
    ------
    ValueError
        If *image* is invalid or *kernel_size* is even or less than 1.
    """
    _validate_bgr_image(image, caller="median_denoise")
    _validate_odd_kernel(kernel_size, caller="median_denoise")

    denoised = cv2.medianBlur(image, kernel_size)
    logger.debug("median_denoise: kernel=%d", kernel_size)
    return denoised


def normalize_brightness(
    image: BgrArray,
    target_mean: float = 128.0,
    clip_limit: float = 2.0,
) -> BgrArray:
    """Standardise image brightness using CLAHE on the LAB L-channel.

    Works in the LAB colour space so that colour hue is preserved while only
    the luminance channel is adjusted.  The approach is consistent with
    ``qr_enhancement.enhance_low_light_qr``, which also uses CLAHE in LAB,
    meaning the two steps complement rather than fight each other.

    The *target_mean* parameter adds a global gamma correction pass after
    CLAHE so that images that are uniformly too dark or too bright are
    shifted toward a standard working level before the enhancement step
    decides which technique to apply.

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.
    target_mean:
        Desired mean luminance in [0, 255].  Defaults to ``128`` (mid-grey).
        Values below 128 shift the image darker; values above shift brighter.
    clip_limit:
        CLAHE clip limit controlling contrast amplification.  Defaults to
        ``2.0`` — the same value used in ``qr_enhancement.enhance_low_light_qr``
        to ensure predictable interactions between the two modules.

    Returns
    -------
    numpy.ndarray
        Brightness-normalised BGR array with the same shape as *image*.

    Raises
    ------
    ValueError
        If *image* is invalid, or *target_mean* is outside [0, 255], or
        *clip_limit* is non-positive.
    """
    _validate_bgr_image(image, caller="normalize_brightness")

    if not (0.0 <= target_mean <= 255.0):
        raise ValueError(
            f"normalize_brightness: target_mean must be in [0, 255], "
            f"got {target_mean}."
        )
    if clip_limit <= 0.0:
        raise ValueError(
            f"normalize_brightness: clip_limit must be > 0, got {clip_limit}."
        )

    # Step 1: CLAHE on L channel in LAB space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)

    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Step 2: global gamma / scale to reach target_mean
    current_mean = float(np.mean(result))
    if current_mean > 1e-6:  # guard against all-black images
        scale = target_mean / current_mean
        # Use convertScaleAbs so values are clipped to [0, 255] safely
        result = cv2.convertScaleAbs(result, alpha=scale, beta=0)

    logger.debug(
        "normalize_brightness: clip_limit=%.2f  target_mean=%.1f  "
        "actual_mean=%.1f",
        clip_limit, target_mean, float(np.mean(result)),
    )
    return result


# ===========================================================================
# Pipeline composer
# ===========================================================================

def preprocess_pipeline(
    image: BgrArray,
    config: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    """Apply preprocessing steps in the recommended order.

    The execution order is fixed to avoid destructive interactions:

    1. **Resize** — normalise dimensions before any pixel-level operations.
    2. **Grayscale** — convert colour channels (if requested).
    3. **Gaussian denoise** — smooth continuous noise.
    4. **Median denoise** — remove impulse noise after Gaussian pass.
    5. **Brightness normalisation** — standardise luminance level last so
       denoising does not shift the mean used by CLAHE.

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.  Must be a valid 3-channel image.
    config:
        :class:`PreprocessConfig` instance controlling which steps run.
        If ``None``, a default :class:`PreprocessConfig` is used (no
        transformations applied — the function becomes a no-op that still
        returns a fully populated :class:`PreprocessResult`).

    Returns
    -------
    PreprocessResult
        Dataclass containing the processed image, list of applied step names,
        per-step timing, total elapsed time, and shape information.

    Raises
    ------
    ValueError
        Propagated from individual step functions on invalid input.

    Notes
    -----
    When both ``denoise_gaussian`` and ``denoise_median`` are enabled,
    Gaussian runs first.  This is intentional: Gaussian blur reduces
    background noise, making the subsequent median filter more effective
    at isolating and removing residual impulse artefacts.
    """
    if config is None:
        config = PreprocessConfig()

    _validate_bgr_image(image, caller="preprocess_pipeline")

    original_shape = image.shape
    t_pipeline_start = time.perf_counter()

    current: BgrArray = image.copy()
    steps_applied: list[str] = []
    step_timings: dict[str, float] = {}

    # ── Step 1: Resize ───────────────────────────────────────────────────────
    if config.resize_target is not None:
        t0 = time.perf_counter()
        current = resize_image(
            current,
            config.resize_target,
            keep_aspect_ratio=config.keep_aspect_ratio,
        )
        step_timings["resize"] = (time.perf_counter() - t0) * 1000.0
        steps_applied.append("resize")
        logger.info(
            "preprocess_pipeline [resize]: %s → %s  (%.1f ms)",
            original_shape[:2][::-1],          # (W, H) for readability
            current.shape[:2][::-1],
            step_timings["resize"],
        )

    # ── Step 2: Grayscale ────────────────────────────────────────────────────
    if config.convert_to_gray:
        t0 = time.perf_counter()
        current = convert_grayscale(current, as_bgr=True)
        step_timings["grayscale"] = (time.perf_counter() - t0) * 1000.0
        steps_applied.append("grayscale")
        logger.info(
            "preprocess_pipeline [grayscale]: converted to 3-ch grey  (%.1f ms)",
            step_timings["grayscale"],
        )

    # ── Step 3: Gaussian denoise ─────────────────────────────────────────────
    if config.denoise_gaussian:
        t0 = time.perf_counter()
        current = gaussian_denoise(
            current,
            kernel_size=config.gaussian_kernel_size,
            sigma=config.gaussian_sigma,
        )
        step_timings["gaussian_denoise"] = (time.perf_counter() - t0) * 1000.0
        steps_applied.append("gaussian_denoise")
        logger.info(
            "preprocess_pipeline [gaussian_denoise]: kernel=%d  sigma=%.2f  (%.1f ms)",
            config.gaussian_kernel_size,
            config.gaussian_sigma,
            step_timings["gaussian_denoise"],
        )

    # ── Step 4: Median denoise ───────────────────────────────────────────────
    if config.denoise_median:
        t0 = time.perf_counter()
        current = median_denoise(current, kernel_size=config.median_kernel_size)
        step_timings["median_denoise"] = (time.perf_counter() - t0) * 1000.0
        steps_applied.append("median_denoise")
        logger.info(
            "preprocess_pipeline [median_denoise]: kernel=%d  (%.1f ms)",
            config.median_kernel_size,
            step_timings["median_denoise"],
        )

    # ── Step 5: Brightness normalisation ─────────────────────────────────────
    if config.normalize_brightness:
        t0 = time.perf_counter()
        current = normalize_brightness(
            current,
            target_mean=config.brightness_target_mean,
            clip_limit=config.brightness_clip_limit,
        )
        step_timings["normalize_brightness"] = (time.perf_counter() - t0) * 1000.0
        steps_applied.append("normalize_brightness")
        logger.info(
            "preprocess_pipeline [normalize_brightness]: "
            "target_mean=%.1f  clip_limit=%.2f  (%.1f ms)",
            config.brightness_target_mean,
            config.brightness_clip_limit,
            step_timings["normalize_brightness"],
        )

    elapsed_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

    if not steps_applied:
        logger.info(
            "preprocess_pipeline: no steps enabled — returning copy of input  "
            "(%.1f ms)",
            elapsed_ms,
        )
    else:
        logger.info(
            "preprocess_pipeline: completed %d step(s) %s in %.1f ms total.",
            len(steps_applied),
            steps_applied,
            elapsed_ms,
        )

    return PreprocessResult(
        processed_image=current,
        processing_steps=steps_applied,
        elapsed_ms=elapsed_ms,
        original_shape=original_shape,
        final_shape=current.shape,
        step_elapsed_ms=step_timings,
    )


# ===========================================================================
# Private validation helpers
# ===========================================================================

def _validate_bgr_image(image: object, *, caller: str = "unknown") -> None:
    """Raise :class:`ValueError` if *image* is not a valid BGR array.

    Checks performed:
    - Not ``None``.
    - Is a ``numpy.ndarray``.
    - Has exactly 3 dimensions.
    - Third dimension (channel count) is 3.
    - ``dtype`` is ``uint8``.

    Parameters
    ----------
    image:
        Value to validate.
    caller:
        Name of the calling function, used in the error message.

    Raises
    ------
    ValueError
        Describing the specific validation failure.
    """
    if image is None:
        raise ValueError(f"{caller}: image must not be None.")
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"{caller}: expected numpy.ndarray, got {type(image).__name__}."
        )
    if image.ndim != 3:
        raise ValueError(
            f"{caller}: expected 3-dimensional array (H, W, C), "
            f"got ndim={image.ndim}  shape={image.shape}."
        )
    if image.shape[2] != 3:
        raise ValueError(
            f"{caller}: expected 3-channel (BGR) array, "
            f"got {image.shape[2]} channel(s)."
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"{caller}: expected dtype uint8, got {image.dtype}.  "
            "Convert the image with image.astype(numpy.uint8) before calling."
        )


def _validate_odd_kernel(kernel_size: int, *, caller: str = "unknown") -> None:
    """Raise :class:`ValueError` if *kernel_size* is not a positive odd integer.

    Parameters
    ----------
    kernel_size:
        Value to validate.
    caller:
        Name of the calling function, used in the error message.

    Raises
    ------
    ValueError
        If *kernel_size* is even, zero, or negative.
    """
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"{caller}: kernel_size must be a positive odd integer, "
            f"got {kernel_size!r}."
        )


# ===========================================================================
# Convenience: quick single-call entry point
# ===========================================================================

def preprocess_for_qr(
    image: BgrArray,
    *,
    resize_target: Optional[Tuple[int, int]] = (800, 800),
    denoise: bool = True,
    normalize: bool = True,
) -> PreprocessResult:
    """Convenience wrapper applying a sensible default preprocessing config.

    Suitable for most still-image QR scanning scenarios.  Live-camera frames
    are typically already at a usable resolution and benefit most from
    denoising alone.

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.
    resize_target:
        Target ``(width, height)`` for resizing.  Defaults to ``(800, 800)``.
        Pass ``None`` to skip resizing (e.g. for webcam frames).
    denoise:
        If ``True`` (default) applies both Gaussian and median denoising.
    normalize:
        If ``True`` (default) normalises brightness to 128 mean.

    Returns
    -------
    PreprocessResult
        Result from :func:`preprocess_pipeline`.
    """
    config = PreprocessConfig(
        resize_target=resize_target,
        keep_aspect_ratio=True,
        convert_to_gray=False,          # keep BGR for qr_enhancement compatibility
        denoise_gaussian=denoise,
        gaussian_kernel_size=3,
        gaussian_sigma=0.0,
        denoise_median=denoise,
        median_kernel_size=3,
        normalize_brightness=normalize,
        brightness_target_mean=128.0,
        brightness_clip_limit=2.0,
    )
    return preprocess_pipeline(image, config)


# ===========================================================================
# CLI entry-point / demo
# ===========================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print(
            "\nUsage: python image_enhancement.py <image_path> [output_path]\n"
            "\n"
            "Runs the default preprocessing pipeline (resize 800×800, denoise,\n"
            "brightness normalisation) and saves the result.\n"
            "\n"
            "Example:\n"
            "  python image_enhancement.py test_qr.jpg preprocessed_qr.jpg\n"
        )
        sys.exit(0)

    src_path = sys.argv[1]
    dst_path = sys.argv[2] if len(sys.argv) > 2 else None

    raw = cv2.imread(src_path)
    if raw is None:
        print(f"[ERROR] Could not read image: {src_path}")
        sys.exit(1)

    result = preprocess_for_qr(raw)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Preprocessing Module — Result")
    print(sep)
    print(f"  Source       : {src_path}")
    print(f"  Original     : {result.original_shape}")
    print(f"  Final shape  : {result.final_shape}")
    print(f"  Steps run    : {result.processing_steps}")
    print(f"  Total time   : {result.elapsed_ms:.2f} ms")
    for step, ms in result.step_elapsed_ms.items():
        print(f"    {step:<30} {ms:.2f} ms")
    print(sep)

    if dst_path:
        cv2.imwrite(dst_path, result.processed_image)
        print(f"\n✅ Saved preprocessed image → {Path(dst_path).resolve()}\n")
    else:
        stem = Path(src_path).stem
        suffix = Path(src_path).suffix
        out = f"preprocessed_{stem}{suffix}"
        cv2.imwrite(out, result.processed_image)
        print(f"\n✅ Saved preprocessed image → {Path(out).resolve()}\n")