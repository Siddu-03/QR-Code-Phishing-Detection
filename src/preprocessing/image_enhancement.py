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

    .. warning::
        Resizing changes the pixel coordinate space of the output image.
        Bounding-box coordinates returned by the detector will be in the
        **resized** image's space.  If you intend to draw those coordinates
        on the **original** image you must call
        :func:`remap_to_original` to transform them back.
        Resizing is therefore **disabled by default** in
        :func:`preprocess_for_qr`.

``convert_grayscale(image, as_bgr)``
    Convert to grayscale for downstream vision processing.  Optionally
    re-wraps the single channel into a 3-channel BGR image so that the
    rest of the pipeline (which expects BGR) receives a compatible array.

``gaussian_denoise(image, kernel_size, sigma)``
    Reduce continuous sensor / camera noise with a Gaussian blur kernel.

    .. note::
        When this step is enabled alongside ``qr_enhancement.auto_enhance``
        with ``try_blur=True``, Gaussian blur runs twice (once here, once
        inside ``enhance_blurred_qr``).  Pass ``try_blur=False`` to
        ``auto_enhance`` to prevent double-blurring.

``median_denoise(image, kernel_size)``
    Remove salt-and-pepper noise with a median filter — preserves edges
    better than Gaussian for binary-like QR module patterns.

``normalize_brightness(image, target_mean, clip_limit)``
    Standardise luminance via CLAHE on the LAB L-channel.

    .. warning::
        ``qr_enhancement.enhance_low_light_qr`` also applies CLAHE on the
        LAB L-channel.  Running both in sequence applies CLAHE twice and
        over-enhances contrast.  When this step is enabled, pass
        ``try_low_light=False`` to ``auto_enhance`` to prevent double CLAHE.

``preprocess_pipeline(image, config)``
    Compose the above steps in a controlled, fixed order.

``remap_to_original(detection_result, prep_result)``
    Inverse-transform detector bounding-box coordinates from preprocessed
    image space back to original image space.  Required when
    ``resize_target`` is set and visualisation draws on the original image.

Integration example (resize disabled — coordinates always valid)
----------------------------------------------------------------
::

    from src.preprocessing.image_enhancement import preprocess_for_qr
    from src.qr_detector.qr_enhancement import auto_enhance
    from src.qr_detector.qr_detector import detect_qr_opencv

    # resize_target=None by default — no coordinate remapping needed.
    prep   = preprocess_for_qr(raw_bgr_image)
    enh    = auto_enhance(prep.processed_image,
                          try_low_light=False,   # avoid double CLAHE
                          try_blur=False)         # avoid double Gaussian
    result = detect_qr_opencv(enh.enhanced_image)

Integration example (resize enabled — remap required)
------------------------------------------------------
::

    from src.preprocessing.image_enhancement import (
        preprocess_for_qr, remap_to_original,
    )

    prep   = preprocess_for_qr(raw_bgr_image, resize_target=(800, 800))
    enh    = auto_enhance(prep.processed_image, try_low_light=False)
    result = detect_qr_opencv(enh.enhanced_image)
    result = remap_to_original(result, prep)   # <-- required after resize
    # result coordinates are now in original image space

Compatibility
-------------
* OpenCV ≥ 4.5   (cv2)
* NumPy ≥ 1.21   (numpy)
* Python ≥ 3.9
* No changes to qr_detector.py or qr_enhancement.py required.
"""

from __future__ import annotations

import copy
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
BgrArray  = np.ndarray   # shape (H, W, 3), dtype uint8, BGR channel order
GrayArray = np.ndarray   # shape (H, W),    dtype uint8


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
        ``["gaussian_denoise", "normalize_brightness"]``.  Steps that were
        skipped (disabled in :class:`PreprocessConfig`) are **not** included.
    elapsed_ms:
        Total wall-clock time for the full pipeline in milliseconds.
    original_shape:
        ``(H, W, C)`` tuple of the input image before any transformation.
    final_shape:
        ``(H, W, C)`` tuple of ``processed_image``.
    step_elapsed_ms:
        Per-step timing in milliseconds, keyed by step name.
    spatial_params:
        Dictionary populated **only** when ``resize_target`` is set.
        Contains the inverse-transform parameters needed by
        :func:`remap_to_original`::

            {
                "scale":  float,   # uniform scale factor applied to the image
                "pad_x":  int,     # horizontal padding added by letterboxing
                "pad_y":  int,     # vertical padding added by letterboxing
                "resize_mode": "letterbox" | "stretch",
                "scale_x": float,  # x-axis scale (stretch mode only)
                "scale_y": float,  # y-axis scale (stretch mode only)
            }

        Empty dict when no resize was performed.
    """

    processed_image:  BgrArray
    processing_steps: list[str]
    elapsed_ms:       float
    original_shape:   Tuple[int, ...]
    final_shape:      Tuple[int, ...]
    step_elapsed_ms:  dict[str, float] = field(default_factory=dict)
    spatial_params:   dict             = field(default_factory=dict)


@dataclass
class PreprocessConfig:
    """Declarative configuration for :func:`preprocess_pipeline`.

    All boolean flags default to ``False``; callers opt into each step
    explicitly so the pipeline is transparent and reproducible.

    Attributes
    ----------
    resize_target:
        ``(width, height)`` in pixels.  Defaults to ``None`` (disabled).

        .. warning::
            When set, detection coordinates will be in the *resized* image's
            pixel space.  You **must** call :func:`remap_to_original` before
            drawing boxes on the original image.  Disable resizing (leave as
            ``None``) whenever downstream visualisation uses the original.

    keep_aspect_ratio:
        If ``True`` (default), letterbox the image to ``resize_target``.
        If ``False``, stretch to fill the target exactly (both axes may
        scale by different factors).
    convert_to_gray:
        Convert the image to grayscale.  The result is re-wrapped as a
        3-channel BGR array (all three channels equal) so downstream
        modules that expect BGR continue to work.
    denoise_gaussian:
        Apply Gaussian blur for continuous noise suppression.

        .. note::
            Mutually exclusive with ``auto_enhance(try_blur=True)``.
            Enabling both applies Gaussian blur twice.

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
        Standardise luminance via CLAHE on the LAB L-channel.

        .. warning::
            Mutually exclusive with ``auto_enhance(try_low_light=True)``.
            Enabling both applies CLAHE twice, over-enhancing contrast.

    brightness_target_mean:
        Target mean pixel value in [0, 255].  Defaults to 128 (mid-grey).
    brightness_clip_limit:
        CLAHE clip limit.  Defaults to 2.0, matching
        ``qr_enhancement.enhance_low_light_qr``.
    """

    # --- Resize (disabled by default — enables coordinate remapping) ---------
    resize_target:         Optional[Tuple[int, int]] = None
    keep_aspect_ratio:     bool  = True

    # --- Grayscale ---
    convert_to_gray:       bool  = False

    # --- Gaussian denoise ---
    denoise_gaussian:      bool  = False
    gaussian_kernel_size:  int   = 3
    gaussian_sigma:        float = 0.0

    # --- Median denoise ---
    denoise_median:        bool  = False
    median_kernel_size:    int   = 3

    # --- Brightness normalisation ---
    normalize_brightness:  bool  = False
    brightness_target_mean: float = 128.0
    brightness_clip_limit:  float = 2.0


# ===========================================================================
# Individual preprocessing functions
# ===========================================================================

def resize_image(
    image: BgrArray,
    target_size: Tuple[int, int],
    *,
    keep_aspect_ratio: bool = True,
) -> tuple[BgrArray, dict]:
    """Resize *image* to *target_size*, optionally preserving aspect ratio.

    When *keep_aspect_ratio* is ``True`` the image is scaled so that neither
    dimension exceeds *target_size* and then letterboxed (padded with black
    pixels) to fill the canvas exactly.  This guarantees the output always
    has the precise dimensions requested without distorting QR finder patterns.

    When *keep_aspect_ratio* is ``False`` the image is stretched directly
    to *target_size* using bilinear interpolation.

    .. warning::
        Resizing changes the pixel coordinate space.  The returned
        *spatial_params* dict contains the parameters needed to invert the
        transform via :func:`remap_to_original`.

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
    resized : numpy.ndarray
        Resized BGR array of shape ``(target_size[1], target_size[0], 3)``.
    spatial_params : dict
        Inverse-transform parameters for :func:`remap_to_original`.

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
        spatial_params = {
            "resize_mode": "stretch",
            "scale_x": tw / src_w,
            "scale_y": th / src_h,
            "scale":   None,
            "pad_x":   0,
            "pad_y":   0,
        }
        logger.debug(
            "resize_image: stretched %dx%d → %dx%d",
            src_w, src_h, tw, th,
        )
        return resized, spatial_params

    # --- Aspect-ratio-preserving path (letterbox) ----------------------------
    scale = min(tw / src_w, th / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    pad_y = (th - new_h) // 2
    pad_x = (tw - new_w) // 2
    canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = scaled

    spatial_params = {
        "resize_mode": "letterbox",
        "scale":       scale,
        "pad_x":       pad_x,
        "pad_y":       pad_y,
        "scale_x":     None,
        "scale_y":     None,
    }
    logger.debug(
        "resize_image: letterboxed %dx%d → %dx%d (pad x=%d, y=%d, scale=%.4f)",
        src_w, src_h, tw, th, pad_x, pad_y, scale,
    )
    return canvas, spatial_params


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
        all three BGR channels, returning a ``(H, W, 3)`` array compatible
        with every downstream function in the project.

        If ``False`` a true single-channel ``(H, W)`` array is returned.
        **Note:** passing a single-channel array to
        ``qr_enhancement.auto_enhance`` or ``qr_detector.detect_qr_opencv``
        will raise an error; use ``as_bgr=False`` only when you intend to
        handle the conversion yourself.

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

    .. note::
        If this step is enabled and ``auto_enhance`` is called afterwards
        with ``try_blur=True``, Gaussian blur will run twice.  Pass
        ``try_blur=False`` to ``auto_enhance`` to avoid double-blurring.

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
    the luminance channel is adjusted.

    .. warning::
        ``qr_enhancement.enhance_low_light_qr`` also applies CLAHE on the
        LAB L-channel (``clip=2.0``, ``tileGridSize=(8, 8)``).  Running
        both in sequence applies CLAHE twice, over-enhancing contrast in
        already-boosted regions.  When this step is enabled, pass
        ``try_low_light=False`` to ``auto_enhance`` to suppress the second
        CLAHE pass.

    The *target_mean* parameter adds a global linear-scaling pass after
    CLAHE to shift uniformly dark or bright images toward a predictable
    working level before the enhancement step selects its technique.

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.
    target_mean:
        Desired mean pixel value in [0, 255].  Defaults to ``128``.
    clip_limit:
        CLAHE clip limit.  Defaults to ``2.0``, matching
        ``qr_enhancement.enhance_low_light_qr``.

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
    l_eq  = clahe.apply(l_ch)

    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Step 2: global linear scale to reach target_mean
    current_mean = float(np.mean(result))
    if current_mean > 1e-6:   # guard against all-black images
        scale  = target_mean / current_mean
        result = cv2.convertScaleAbs(result, alpha=scale, beta=0)

    logger.debug(
        "normalize_brightness: clip_limit=%.2f  target_mean=%.1f  "
        "actual_mean=%.1f",
        clip_limit, target_mean, float(np.mean(result)),
    )
    return result


# ===========================================================================
# Coordinate remapping
# ===========================================================================

def remap_to_original(
    detection_result: dict,
    prep_result: PreprocessResult,
) -> dict:
    """Transform detector coordinates from preprocessed space to original space.

    Must be called after detection when ``resize_target`` was set during
    preprocessing.  Has no effect (returns a copy) when no resize was
    performed (``prep_result.spatial_params`` is empty).

    The function creates a **deep copy** of *detection_result* and updates
    every ``bbox_tuple``, ``bbox_dict``, and ``corner_points`` entry in
    ``detections`` in-place on the copy.  The original dict is not mutated.

    Parameters
    ----------
    detection_result:
        ``DetectionResult`` dict returned by :func:`qr_detector.detect_qr`
        or ``detect_qr_frame`` in ``live_scan.py``.
    prep_result:
        :class:`PreprocessResult` from the preprocessing step that was
        applied before detection.

    Returns
    -------
    dict
        A deep copy of *detection_result* with all coordinates transformed
        into the original image's pixel space.

    Raises
    ------
    ValueError
        If ``prep_result.spatial_params`` contains an unrecognised
        ``resize_mode``.

    Notes
    -----
    Inverse transform formulas
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    **Letterbox mode** (``keep_aspect_ratio=True``):

    ::

        x_orig = int((x_prep - pad_x) / scale)
        y_orig = int((y_prep - pad_y) / scale)
        w_orig = int(w_prep / scale)
        h_orig = int(h_prep / scale)

    **Stretch mode** (``keep_aspect_ratio=False``):

    ::

        x_orig = int(x_prep / scale_x)
        y_orig = int(y_prep / scale_y)
        w_orig = int(w_prep / scale_x)
        h_orig = int(h_prep / scale_y)

    Corner points are remapped element-wise using the same formulas applied
    to each ``[x, y]`` pair.
    """
    params = prep_result.spatial_params

    # No resize was performed — return a copy with coordinates unchanged.
    if not params:
        return copy.deepcopy(detection_result)

    mode = params.get("resize_mode")
    if mode not in ("letterbox", "stretch"):
        raise ValueError(
            f"remap_to_original: unknown resize_mode '{mode}' in spatial_params."
        )

    result = copy.deepcopy(detection_result)

    for det in result.get("detections", []):
        # ── bbox_tuple: [x, y, w, h] ────────────────────────────────────────
        x_p, y_p, w_p, h_p = det["bbox_tuple"]

        if mode == "letterbox":
            scale = params["scale"]
            pad_x = params["pad_x"]
            pad_y = params["pad_y"]
            x_o = int((x_p - pad_x) / scale)
            y_o = int((y_p - pad_y) / scale)
            w_o = int(w_p / scale)
            h_o = int(h_p / scale)
        else:  # stretch
            sx = params["scale_x"]
            sy = params["scale_y"]
            x_o = int(x_p / sx)
            y_o = int(y_p / sy)
            w_o = int(w_p / sx)
            h_o = int(h_p / sy)

        det["bbox_tuple"] = [x_o, y_o, w_o, h_o]
        det["bbox_dict"]  = {"x": x_o, "y": y_o, "w": w_o, "h": h_o}

        # ── corner_points: [[x, y], ...] ─────────────────────────────────────
        remapped_corners = []
        for cx, cy in det["corner_points"]:
            if mode == "letterbox":
                rx = int((cx - pad_x) / scale)
                ry = int((cy - pad_y) / scale)
            else:
                rx = int(cx / sx)
                ry = int(cy / sy)
            remapped_corners.append([rx, ry])
        det["corner_points"] = remapped_corners

    logger.debug(
        "remap_to_original: remapped %d detection(s) via mode='%s'.",
        len(result.get("detections", [])),
        mode,
    )
    return result


# ===========================================================================
# Pipeline composer
# ===========================================================================

def preprocess_pipeline(
    image: BgrArray,
    config: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    """Apply preprocessing steps in the recommended fixed order.

    Execution order
    ---------------
    1. **Resize** — normalise dimensions before any pixel-level operations.
       Disabled by default.  Populates ``PreprocessResult.spatial_params``
       when active so that :func:`remap_to_original` can invert the transform.
    2. **Grayscale** — convert colour channels (if requested).
    3. **Gaussian denoise** — smooth continuous noise.
    4. **Median denoise** — remove impulse noise after Gaussian pass.
    5. **Brightness normalisation** — standardise luminance level last so
       denoising does not shift the mean consumed by CLAHE.

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.  Must be a valid 3-channel uint8 image.
    config:
        :class:`PreprocessConfig` controlling which steps run.  If ``None``,
        a default config is used (no steps enabled — returns a copy of input).

    Returns
    -------
    PreprocessResult
        Dataclass with ``processed_image``, ``processing_steps``, per-step
        timings, shape information, and ``spatial_params`` (non-empty only
        when resize ran).

    Raises
    ------
    ValueError
        Propagated from individual step functions on invalid input.

    Notes
    -----
    When both ``denoise_gaussian`` and ``denoise_median`` are enabled,
    Gaussian runs first — it reduces background noise, making the median
    filter more effective at isolating residual impulse artefacts.

    When ``normalize_brightness`` is enabled, pass ``try_low_light=False``
    to ``auto_enhance`` downstream to prevent double CLAHE.

    When ``denoise_gaussian`` is enabled, pass ``try_blur=False`` to
    ``auto_enhance`` downstream to prevent double Gaussian blurring.
    """
    if config is None:
        config = PreprocessConfig()

    _validate_bgr_image(image, caller="preprocess_pipeline")

    original_shape    = image.shape
    t_pipeline_start  = time.perf_counter()

    current: BgrArray         = image.copy()
    steps_applied: list[str]  = []
    step_timings: dict[str, float] = {}
    spatial_params: dict      = {}

    # ── Step 1: Resize ───────────────────────────────────────────────────────
    if config.resize_target is not None:
        t0 = time.perf_counter()
        current, spatial_params = resize_image(
            current,
            config.resize_target,
            keep_aspect_ratio=config.keep_aspect_ratio,
        )
        step_timings["resize"] = (time.perf_counter() - t0) * 1000.0
        steps_applied.append("resize")
        logger.debug(
            "preprocess_pipeline [resize]: %s → %s  (%.1f ms)",
            original_shape[:2][::-1],   # (W, H) for readability
            current.shape[:2][::-1],
            step_timings["resize"],
        )

    # ── Step 2: Grayscale ────────────────────────────────────────────────────
    if config.convert_to_gray:
        t0 = time.perf_counter()
        current = convert_grayscale(current, as_bgr=True)
        step_timings["grayscale"] = (time.perf_counter() - t0) * 1000.0
        steps_applied.append("grayscale")
        logger.debug(
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
        logger.debug(
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
        logger.debug(
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
        logger.debug(
            "preprocess_pipeline [normalize_brightness]: "
            "target_mean=%.1f  clip_limit=%.2f  (%.1f ms)",
            config.brightness_target_mean,
            config.brightness_clip_limit,
            step_timings["normalize_brightness"],
        )

    elapsed_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

    if not steps_applied:
        logger.debug(
            "preprocess_pipeline: no steps enabled — returning copy of input "
            "(%.1f ms).",
            elapsed_ms,
        )
    else:
        logger.info(
            "preprocess_pipeline: completed %d step(s) %s in %.1f ms.",
            len(steps_applied),
            steps_applied,
            elapsed_ms,
        )

    return PreprocessResult(
        processed_image  = current,
        processing_steps = steps_applied,
        elapsed_ms       = elapsed_ms,
        original_shape   = original_shape,
        final_shape      = current.shape,
        step_elapsed_ms  = step_timings,
        spatial_params   = spatial_params,
    )


# ===========================================================================
# Private validation helpers
# ===========================================================================

def _validate_bgr_image(image: object, *, caller: str = "unknown") -> None:
    """Raise :class:`ValueError` if *image* is not a valid BGR uint8 array.

    Checks performed
    ----------------
    * Not ``None``.
    * Is a ``numpy.ndarray``.
    * Has exactly 3 dimensions ``(H, W, C)``.
    * Third dimension is 3 channels.
    * ``dtype`` is ``uint8``.
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
    """Raise :class:`ValueError` if *kernel_size* is not a positive odd integer."""
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"{caller}: kernel_size must be a positive odd integer, "
            f"got {kernel_size!r}."
        )


# ===========================================================================
# Convenience entry point
# ===========================================================================

def preprocess_for_qr(
    image: BgrArray,
    *,
    resize_target: Optional[Tuple[int, int]] = None,
    denoise: bool = True,
    normalize: bool = True,
) -> PreprocessResult:
    """Convenience wrapper applying the recommended default preprocessing config.

    Resize is **disabled by default** (``resize_target=None``) to ensure that
    bounding-box coordinates returned by the detector remain valid in the
    original image's pixel space.  Downstream visualisation can then draw
    directly on the original image without any coordinate remapping.

    If you do need to resize (e.g. to accelerate detection on very large
    images), pass ``resize_target=(w, h)`` and call :func:`remap_to_original`
    on the detection result before visualisation.

    Parameters
    ----------
    image:
        Input BGR ``numpy.ndarray``.
    resize_target:
        Target ``(width, height)`` for resizing.  Defaults to ``None``
        (resizing disabled).  When set, you **must** call
        :func:`remap_to_original` on the detection result before drawing
        on the original image.
    denoise:
        If ``True`` (default) applies both Gaussian (k=3) and median (k=3)
        denoising.

        .. note::
            When ``True``, pass ``try_blur=False`` to ``auto_enhance`` to
            prevent double Gaussian blurring.

    normalize:
        If ``True`` (default) normalises brightness via CLAHE + mean scaling.

        .. warning::
            When ``True``, pass ``try_low_light=False`` to ``auto_enhance``
            to prevent double CLAHE.

    Returns
    -------
    PreprocessResult
        Result from :func:`preprocess_pipeline`.
    """
    config = PreprocessConfig(
        resize_target         = resize_target,
        keep_aspect_ratio     = True,
        convert_to_gray       = False,   # keep BGR for qr_enhancement compatibility
        denoise_gaussian      = denoise,
        gaussian_kernel_size  = 3,
        gaussian_sigma        = 0.0,
        denoise_median        = denoise,
        median_kernel_size    = 3,
        normalize_brightness  = normalize,
        brightness_target_mean = 128.0,
        brightness_clip_limit  = 2.0,
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
            "Runs the default preprocessing pipeline (denoise + brightness\n"
            "normalisation, no resize) and saves the result.\n"
            "Resizing is disabled by default to preserve coordinate validity.\n"
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

    # resize_target=None by default — coordinates remain valid.
    result = preprocess_for_qr(raw)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Preprocessing Module — Result")
    print(sep)
    print(f"  Source       : {src_path}")
    print(f"  Original     : {result.original_shape}")
    print(f"  Final shape  : {result.final_shape}")
    print(f"  Steps run    : {result.processing_steps or ['none']}")
    print(f"  Total time   : {result.elapsed_ms:.2f} ms")
    for step, ms in result.step_elapsed_ms.items():
        print(f"    {step:<30} {ms:.2f} ms")
    if result.spatial_params:
        print(f"  Spatial params : {result.spatial_params}")
        print("  ⚠  Resize was active — call remap_to_original() before visualisation.")
    print(sep)

    if dst_path:
        cv2.imwrite(dst_path, result.processed_image)
        print(f"\n✅ Saved preprocessed image → {Path(dst_path).resolve()}\n")
    else:
        stem   = Path(src_path).stem
        suffix = Path(src_path).suffix
        out    = f"preprocessed_{stem}{suffix}"
        cv2.imwrite(out, result.processed_image)
        print(f"\n✅ Saved preprocessed image → {Path(out).resolve()}\n")