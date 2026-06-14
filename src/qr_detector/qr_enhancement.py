"""
qr_enhancement.py
=================
Week 2 – Member 2: QR Enhancement Module
Project: QR Code Tamper Detection using Computer Vision

Purpose
-------
An optional pre-processing layer that improves image quality **before** the
image reaches ``qr_detector.detect_qr()``.  Every function accepts a BGR
``numpy.ndarray`` (the native format used by ``cv2.imread`` and the rest of
the Week 1 pipeline) and returns a BGR ``numpy.ndarray``, so the module slots
in without any changes to existing code.

Enhancement functions
---------------------
``enhance_rotated_qr(image)``
    Attempts multiple rotation angles and returns the candidate that the
    detector is most likely to succeed on (highest Laplacian variance →
    sharpest finder-pattern edges).

``enhance_low_light_qr(image)``
    Applies CLAHE (Contrast-Limited Adaptive Histogram Equalisation) in the
    L channel of LAB colour space, then merges back to BGR.  Significantly
    brightens under-exposed images while avoiding over-saturation.

``enhance_blurred_qr(image)``
    Two-stage pipeline: Gaussian blur for noise suppression followed by an
    unsharp-mask sharpening pass.  Recovers edge definition in motion-blurred
    or out-of-focus captures.

``benchmark_detectors(image_path, enhance_fn)``
    Runs ``detect_qr`` with and without enhancement, measures detection rate
    and wall-clock time, and returns a structured comparison report.

Integration
-----------
Drop-in before ``detect_qr``::

    from src.qr_detector.qr_enhancement import enhance_low_light_qr
    from src.qr_detector.qr_detector import detect_qr

    # optional — only call when needed
    enhanced = enhance_low_light_qr(raw_bgr_array)
    result = detect_qr_from_array(enhanced)   # see example in __main__

Compatibility
-------------
* OpenCV ≥ 4.5   (cv2)
* NumPy ≥ 1.21   (numpy)
* Python ≥ 3.9
* No changes to Week 1 files required.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Re-use Week 1 detection building blocks — no logic duplicated
# ---------------------------------------------------------------------------
from src.qr_detector.qr_detector import (
    detect_qr_opencv,
    detect_qr_pyzbar,
    convert_coordinates,
    load_image as _load_image_from_path,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("qr_enhancement")


# ===========================================================================
# Internal type aliases  (mirror qr_detector.py conventions)
# ===========================================================================
BgrArray = np.ndarray   # shape (H, W, 3), dtype uint8, BGR channel order


# ===========================================================================
# Dataclasses for structured results
# ===========================================================================

@dataclass
class EnhancementResult:
    """Wraps an enhanced image with metadata about what was applied."""

    enhanced_image: BgrArray
    """The processed BGR image ready for QR detection."""

    technique: str
    """Human-readable name of the enhancement applied."""

    params: dict = field(default_factory=dict)
    """Key hyper-parameters used (for reproducibility / logging)."""

    elapsed_ms: float = 0.0
    """Wall-clock time for the enhancement step in milliseconds."""


@dataclass
class BenchmarkReport:
    """Comparison of QR detection with and without enhancement."""

    image_path: str
    """Absolute path of the tested image."""

    technique: str
    """Name of the enhancement technique under test."""

    # --- baseline (no enhancement) ------------------------------------------
    baseline_detected: bool = False
    baseline_count: int = 0
    baseline_detector_used: str = "none"
    baseline_elapsed_ms: float = 0.0

    # --- enhanced -----------------------------------------------------------
    enhanced_detected: bool = False
    enhanced_count: int = 0
    enhanced_detector_used: str = "none"
    enhanced_elapsed_ms: float = 0.0
    enhancement_elapsed_ms: float = 0.0

    # --- derived summary -----------------------------------------------------
    @property
    def improvement(self) -> str:
        """One-line human-readable verdict."""
        if self.enhanced_detected and not self.baseline_detected:
            return "✅ Enhancement ENABLED detection (was failing before)"
        if self.enhanced_count > self.baseline_count:
            return (
                f"✅ Enhancement found more codes "
                f"({self.enhanced_count} vs {self.baseline_count})"
            )
        if not self.enhanced_detected and not self.baseline_detected:
            return "⚠  Neither baseline nor enhanced detected a QR code"
        if self.baseline_detected and not self.enhanced_detected:
            return "❌ Enhancement degraded detection (was working before)"
        return "ℹ  No change in detection outcome"


# ===========================================================================
# Private helpers
# ===========================================================================

def _detect_from_array(image: BgrArray) -> dict:
    """Run the Week 1 OpenCV-first / pyzbar-fallback pipeline on an in-memory
    array (avoids disk I/O).  Returns the same dict structure as
    ``qr_detector.detect_qr``.

    Parameters
    ----------
    image:
        BGR image array.

    Returns
    -------
    dict
        ``DetectionResult`` compatible with ``qr_detector.detect_qr`` output.
    """
    raw_detections: list[tuple[str, np.ndarray]] = []
    detector_used = "none"

    try:
        raw_detections = detect_qr_opencv(image)
        if raw_detections:
            detector_used = "opencv"
    except RuntimeError as exc:
        logger.debug("OpenCV detection failed (%s); trying pyzbar.", exc)

    if not raw_detections:
        try:
            raw_detections = detect_qr_pyzbar(image)
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

    img_h, img_w = image.shape[:2]
    return {
        "detected": len(detections) > 0,
        "count": len(detections),
        "detector_used": detector_used,
        "image_info": {"width": img_w, "height": img_h},
        "detections": detections,
    }


def _laplacian_variance(image: BgrArray) -> float:
    """Return the variance of the Laplacian of *image*.

    Used as a focus / sharpness metric: higher → sharper edges.
    Computed on the grayscale channel.

    Parameters
    ----------
    image:
        BGR image array.

    Returns
    -------
    float
        Laplacian variance (non-negative).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ===========================================================================
# Public Enhancement Functions
# ===========================================================================

def enhance_rotated_qr(
    image: BgrArray,
    *,
    angles: Optional[list[float]] = None,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> EnhancementResult:
    """Attempt multiple rotation angles and return the best candidate image.

    Strategy
    --------
    For each candidate angle the function:

    1. Rotates the image around its centre (without cropping — the canvas
       expands to contain the full rotated image).
    2. Measures the Laplacian variance of the result (proxy for QR finder
       pattern edge sharpness).
    3. Returns the rotation with the **highest** Laplacian variance, which
       correlates strongly with detector success on rotated QR codes.

    If the caller already knows the probable rotation range they can supply a
    custom ``angles`` list.  The default covers the most common cases:
    0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°.

    Parameters
    ----------
    image:
        Input BGR image (potentially containing a rotated QR code).
    angles:
        List of rotation angles in degrees (counter-clockwise).
        Defaults to ``[0, 45, 90, 135, 180, 225, 270, 315]``.
    border_mode:
        OpenCV border extrapolation mode for pixels exposed after rotation.
        ``cv2.BORDER_REPLICATE`` (default) avoids dark/white edge artifacts
        that can confuse QR finders.

    Returns
    -------
    EnhancementResult
        ``.enhanced_image`` is the best-angle BGR array.
        ``.params["best_angle"]`` records which angle won.
        ``.params["scores"]`` maps every angle → its Laplacian variance.

    Raises
    ------
    ValueError
        If *image* is not a 3-channel BGR array.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "enhance_rotated_qr: expected a 3-channel BGR array, "
            f"got shape {getattr(image, 'shape', None)}"
        )

    t0 = time.perf_counter()

    if angles is None:
        angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    best_image: BgrArray = image
    best_score: float = -1.0
    best_angle: float = 0.0
    scores: dict[float, float] = {}

    for angle in angles:
        # Build rotation matrix; expand=True keeps full image visible
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # Compute new canvas size after rotation
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)

        # Adjust translation so the rotated image centres in the new canvas
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0

        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )
        score = _laplacian_variance(rotated)
        scores[angle] = score

        if score > best_score:
            best_score = score
            best_image = rotated
            best_angle = angle

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "enhance_rotated_qr — best angle=%.1f° (score=%.2f)  [%.1f ms]",
        best_angle, best_score, elapsed_ms,
    )

    return EnhancementResult(
        enhanced_image=best_image,
        technique="rotation",
        params={"best_angle": best_angle, "scores": scores, "border_mode": border_mode},
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------

def enhance_low_light_qr(
    image: BgrArray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    gamma: Optional[float] = None,
) -> EnhancementResult:
    """Improve QR detection in under-exposed (dark / low-light) images.

    Algorithm
    ---------
    1. Convert BGR → LAB colour space.  The L channel carries all luminance
       information; A/B carry chrominance and are left untouched.
    2. Apply CLAHE (Contrast-Limited Adaptive Histogram Equalisation) to L.
       CLAHE divides L into small tiles (``tile_grid_size``) and equalises each
       independently, then bilinear-interpolates at tile boundaries.
       ``clip_limit`` caps amplification to prevent noise amplification.
    3. Optionally apply gamma correction before CLAHE if ``gamma`` is given
       (useful for very dark images; ``gamma < 1`` brightens).
    4. Merge the enhanced L back with the original A/B channels.
    5. Convert LAB → BGR and return.

    Why CLAHE in LAB?
        Equalising directly in BGR smears colours and alters hue.
        Operating only on L preserves colour fidelity while maximising
        luminance contrast — exactly what QR module detection needs.

    Parameters
    ----------
    image:
        Input BGR array (dark / low-light).
    clip_limit:
        CLAHE contrast-limit threshold.  Higher → more aggressive
        equalisation but also more noise.  Default 2.0 is conservative.
    tile_grid_size:
        CLAHE tile size ``(cols, rows)``.  Smaller → more local adaptation.
        Default ``(8, 8)`` works well for typical camera resolutions.
    gamma:
        Optional gamma value applied before CLAHE.  ``None`` skips gamma.
        Values in ``(0, 1)`` brighten; values ``> 1`` darken.

    Returns
    -------
    EnhancementResult
        ``.enhanced_image`` is the brightness-corrected BGR array.

    Raises
    ------
    ValueError
        If *image* is not a 3-channel BGR array.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "enhance_low_light_qr: expected a 3-channel BGR array, "
            f"got shape {getattr(image, 'shape', None)}"
        )

    t0 = time.perf_counter()

    work = image.copy()

    # Optional gamma pre-pass (brightens very dark frames before CLAHE)
    if gamma is not None and gamma != 1.0:
        inv_gamma = 1.0 / gamma
        lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        work = cv2.LUT(work, lut)
        logger.debug("enhance_low_light_qr — gamma correction applied (γ=%.2f)", gamma)

    # Convert to LAB and split channels
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Apply CLAHE to the L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_ch)

    # Merge enhanced L with original A, B and convert back to BGR
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "enhance_low_light_qr — CLAHE applied "
        "(clip=%.1f, tile=%s, gamma=%s)  [%.1f ms]",
        clip_limit, tile_grid_size, gamma, elapsed_ms,
    )

    return EnhancementResult(
        enhanced_image=enhanced,
        technique="low_light_clahe",
        params={
            "clip_limit": clip_limit,
            "tile_grid_size": tile_grid_size,
            "gamma": gamma,
        },
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------

def enhance_blurred_qr(
    image: BgrArray,
    *,
    denoise_ksize: int = 3,
    sharpen_strength: float = 1.5,
    sharpen_sigma: float = 1.0,
    threshold_binarise: bool = False,
    threshold_block_size: int = 11,
    threshold_c: int = 2,
) -> EnhancementResult:
    """Recover detail in motion-blurred or out-of-focus QR images.

    Algorithm
    ---------
    1. **Noise suppression** — Gaussian blur with a small kernel removes
       high-frequency sensor noise that would otherwise be amplified by
       sharpening.
    2. **Unsharp mask sharpening** — a blurred copy is subtracted from the
       original, and the difference is added back at ``sharpen_strength``.
       Formula::

           sharpened = original + strength × (original − blurred)
                     ≡ (1 + strength) × original − strength × blurred

       This recovers soft finder-pattern edges without ringing.
    3. **Optional adaptive binarisation** — if ``threshold_binarise=True``,
       the sharpened image is converted to grayscale and binarised via
       ``cv2.adaptiveThreshold`` with Gaussian weighting.  Useful for
       cameras with severe focus blur where colour is unhelpful.

    Parameters
    ----------
    image:
        Input BGR array (blurred / defocused QR).
    denoise_ksize:
        Gaussian kernel size for the denoising pre-pass.
        Must be a positive odd integer.  Larger → stronger smoothing.
        Default ``3`` is mild.
    sharpen_strength:
        Unsharp-mask amplification factor.  ``1.5`` (default) gives a
        moderate boost; raise toward ``3.0`` for heavy blur.
    sharpen_sigma:
        Standard deviation for the blurred copy used in unsharp masking.
        Larger sigma → broader sharpening halo.
    threshold_binarise:
        If ``True``, output is a 3-channel grayscale binary image (not
        colour BGR).  Helps pyzbar on very blurry captures.
    threshold_block_size:
        Adaptive threshold neighbourhood size (must be odd, ≥ 3).
    threshold_c:
        Constant subtracted from the neighbourhood mean in adaptive threshold.

    Returns
    -------
    EnhancementResult
        ``.enhanced_image`` is the sharpened (optionally binarised) BGR array.

    Raises
    ------
    ValueError
        If *image* is not a 3-channel BGR array, or if kernel sizes are
        invalid.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "enhance_blurred_qr: expected a 3-channel BGR array, "
            f"got shape {getattr(image, 'shape', None)}"
        )
    if denoise_ksize < 1 or denoise_ksize % 2 == 0:
        raise ValueError(
            f"denoise_ksize must be a positive odd integer, got {denoise_ksize}"
        )
    if threshold_block_size < 3 or threshold_block_size % 2 == 0:
        raise ValueError(
            f"threshold_block_size must be an odd integer ≥ 3, "
            f"got {threshold_block_size}"
        )

    t0 = time.perf_counter()

    # Step 1 — mild Gaussian denoise
    denoised = cv2.GaussianBlur(image, (denoise_ksize, denoise_ksize), 0)

    # Step 2 — unsharp mask
    blur_for_mask = cv2.GaussianBlur(
        denoised, (0, 0), sigmaX=sharpen_sigma, sigmaY=sharpen_sigma
    )
    sharpened = cv2.addWeighted(
        denoised, 1.0 + sharpen_strength,
        blur_for_mask, -sharpen_strength,
        0,
    )
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Step 3 — optional adaptive binarisation
    if threshold_binarise:
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            threshold_block_size,
            threshold_c,
        )
        # Return as 3-channel so it stays compatible with BGR pipeline
        sharpened = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "enhance_blurred_qr — unsharp mask applied "
        "(denoise_k=%d, strength=%.2f, sigma=%.2f, binarise=%s)  [%.1f ms]",
        denoise_ksize, sharpen_strength, sharpen_sigma,
        threshold_binarise, elapsed_ms,
    )

    return EnhancementResult(
        enhanced_image=sharpened,
        technique="blur_sharpen" + ("_binarised" if threshold_binarise else ""),
        params={
            "denoise_ksize": denoise_ksize,
            "sharpen_strength": sharpen_strength,
            "sharpen_sigma": sharpen_sigma,
            "threshold_binarise": threshold_binarise,
            "threshold_block_size": threshold_block_size,
            "threshold_c": threshold_c,
        },
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------

def enhance_contrast_qr(
    image: BgrArray,
    *,
    alpha: float = 1.4,
    beta: float = 10,
) -> EnhancementResult:
    """Apply basic linear contrast stretching (alpha / beta correction).

    Suitable as a lightweight fallback when CLAHE overhead is undesirable
    (e.g., inside a high-frame-rate live scan loop).

    Formula::

        output = clip(alpha × input + beta, 0, 255)

    Parameters
    ----------
    image:
        Input BGR array.
    alpha:
        Contrast multiplier.  ``1.0`` = no change; ``> 1`` increases
        contrast; ``< 1`` reduces contrast.  Default ``1.4``.
    beta:
        Brightness offset added after scaling.  Positive values brighten.
        Default ``10``.

    Returns
    -------
    EnhancementResult
        ``.enhanced_image`` is the contrast-stretched BGR array.

    Raises
    ------
    ValueError
        If *image* is not a 3-channel BGR array, or if *alpha* ≤ 0.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "enhance_contrast_qr: expected a 3-channel BGR array, "
            f"got shape {getattr(image, 'shape', None)}"
        )
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    t0 = time.perf_counter()

    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "enhance_contrast_qr — linear correction (α=%.2f, β=%d)  [%.1f ms]",
        alpha, beta, elapsed_ms,
    )

    return EnhancementResult(
        enhanced_image=enhanced,
        technique="linear_contrast",
        params={"alpha": alpha, "beta": beta},
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------

def enhance_sharpness_qr(
    image: BgrArray,
    *,
    kernel_type: str = "laplacian",
) -> EnhancementResult:
    """Apply a fixed sharpening convolution kernel.

    Offers two classic sharpening kernels:

    ``"laplacian"``  (default)
        Edge-enhancement kernel derived from the Laplacian operator.
        Aggressively sharpens all edges including QR finder patterns.

    ``"mild"``
        A softer 3×3 kernel that sharpens without the ringing that
        Laplacian can produce on noisy captures.

    Parameters
    ----------
    image:
        Input BGR array.
    kernel_type:
        ``"laplacian"`` or ``"mild"``.

    Returns
    -------
    EnhancementResult
        ``.enhanced_image`` is the sharpened BGR array.

    Raises
    ------
    ValueError
        If *image* is not a 3-channel BGR array, or ``kernel_type`` is
        unrecognised.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "enhance_sharpness_qr: expected a 3-channel BGR array, "
            f"got shape {getattr(image, 'shape', None)}"
        )

    kernels: dict[str, np.ndarray] = {
        "laplacian": np.array(
            [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]], dtype=np.float32
        ),
        "mild": np.array(
            [[-0.5, -0.5, -0.5],
             [-0.5,  5.0, -0.5],
             [-0.5, -0.5, -0.5]], dtype=np.float32
        ) / 1.0,
    }

    if kernel_type not in kernels:
        raise ValueError(
            f"kernel_type must be one of {list(kernels)}, got '{kernel_type}'"
        )

    t0 = time.perf_counter()
    kernel = kernels[kernel_type]
    sharpened = cv2.filter2D(image, -1, kernel)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "enhance_sharpness_qr — kernel=%s  [%.1f ms]", kernel_type, elapsed_ms
    )

    return EnhancementResult(
        enhanced_image=sharpened,
        technique=f"sharpness_{kernel_type}",
        params={"kernel_type": kernel_type},
        elapsed_ms=elapsed_ms,
    )


# ===========================================================================
# Benchmarking Utility
# ===========================================================================

def benchmark_detectors(
    image_path: str,
    enhance_fn: Optional[Callable[[BgrArray], EnhancementResult]] = None,
    *,
    enhance_fn_name: Optional[str] = None,
    runs: int = 1,
    verbose: bool = True,
) -> BenchmarkReport:
    """Compare QR detection accuracy with and without an enhancement function.

    Runs ``detect_qr`` on the raw image (baseline) and, if *enhance_fn* is
    supplied, on the enhanced image.  Timing for both paths is measured so
    the performance cost of enhancement can be assessed.

    Parameters
    ----------
    image_path:
        Path to a JPG or PNG test image.
    enhance_fn:
        An enhancement function with signature
        ``(BgrArray) -> EnhancementResult``.  If ``None``, only the baseline
        is benchmarked and the enhanced columns of the report will reflect
        the same values as the baseline.
    enhance_fn_name:
        Human-readable label used in the report.  Inferred from
        ``enhance_fn.__name__`` if not provided.
    runs:
        Number of times to repeat each detection path (averaged for timing).
        Default ``1``; set higher (e.g. 5) for more stable timing estimates.
    verbose:
        If ``True`` (default), prints a formatted report to stdout.

    Returns
    -------
    BenchmarkReport
        Dataclass with baseline and enhanced metrics.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    ValueError
        If OpenCV cannot decode the file.
    """
    abs_path = str(Path(image_path).resolve())

    # Load image once; share the BGR array across runs
    raw_image: BgrArray = _load_image_from_path(abs_path)

    technique_name = (
        enhance_fn_name
        or (enhance_fn.__name__ if enhance_fn is not None else "none")
    )
    report = BenchmarkReport(image_path=abs_path, technique=technique_name)

    # ── Baseline (no enhancement) ──────────────────────────────────────────
    baseline_times: list[float] = []
    last_baseline: dict = {}
    for _ in range(runs):
        t0 = time.perf_counter()
        last_baseline = _detect_from_array(raw_image)
        baseline_times.append((time.perf_counter() - t0) * 1000.0)

    report.baseline_detected = last_baseline["detected"]
    report.baseline_count = last_baseline["count"]
    report.baseline_detector_used = last_baseline["detector_used"]
    report.baseline_elapsed_ms = sum(baseline_times) / len(baseline_times)

    # ── Enhanced ───────────────────────────────────────────────────────────
    if enhance_fn is not None:
        enhanced_times: list[float] = []
        enhance_times: list[float] = []
        last_enhanced: dict = {}

        for _ in range(runs):
            # Measure enhancement separately
            t_enh0 = time.perf_counter()
            enh_result = enhance_fn(raw_image)
            enhance_times.append((time.perf_counter() - t_enh0) * 1000.0)

            t_det0 = time.perf_counter()
            last_enhanced = _detect_from_array(enh_result.enhanced_image)
            enhanced_times.append((time.perf_counter() - t_det0) * 1000.0)

        report.enhanced_detected = last_enhanced["detected"]
        report.enhanced_count = last_enhanced["count"]
        report.enhanced_detector_used = last_enhanced["detector_used"]
        report.enhanced_elapsed_ms = sum(enhanced_times) / len(enhanced_times)
        report.enhancement_elapsed_ms = sum(enhance_times) / len(enhance_times)
    else:
        # No enhancement function — copy baseline values
        report.enhanced_detected = report.baseline_detected
        report.enhanced_count = report.baseline_count
        report.enhanced_detector_used = report.baseline_detector_used
        report.enhanced_elapsed_ms = report.baseline_elapsed_ms

    # ── Print summary ──────────────────────────────────────────────────────
    if verbose:
        _print_benchmark_report(report, runs)

    return report


def _print_benchmark_report(report: BenchmarkReport, runs: int) -> None:
    """Print a human-readable benchmark report to stdout."""
    sep = "=" * 62
    print(f"\n{sep}")
    print("  QR Enhancement — Benchmark Report")
    print(sep)
    print(f"  Image     : {report.image_path}")
    print(f"  Technique : {report.technique}")
    print(f"  Runs      : {runs}")
    print(sep)
    print(f"\n  {'Metric':<32} {'Baseline':>12} {'Enhanced':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Detected':<32} {str(report.baseline_detected):>12} "
          f"{str(report.enhanced_detected):>12}")
    print(f"  {'QR codes found':<32} {report.baseline_count:>12} "
          f"{report.enhanced_count:>12}")
    print(f"  {'Detector used':<32} {report.baseline_detector_used:>12} "
          f"{report.enhanced_detector_used:>12}")
    print(f"  {'Detection time (ms, avg)':<32} "
          f"{report.baseline_elapsed_ms:>11.1f}  "
          f"{report.enhanced_elapsed_ms:>11.1f}")
    if report.enhancement_elapsed_ms > 0.0:
        print(f"  {'Enhancement overhead (ms)':<32} "
              f"{'—':>12} {report.enhancement_elapsed_ms:>11.1f}")
    print(f"\n  Verdict: {report.improvement}")
    print(f"\n{sep}\n")


# ===========================================================================
# Convenience: apply all enhancements and pick the best
# ===========================================================================

def auto_enhance(
    image: BgrArray,
    *,
    try_rotation: bool = True,
    try_low_light: bool = True,
    try_blur: bool = True,
    try_contrast: bool = True,
) -> EnhancementResult:
    """Apply a sequence of enhancements and return the one that yields the
    best Laplacian variance score (most likely to help QR detection).

    This is a convenient single-call entry point for callers who do not know
    in advance which enhancement their image needs.

    Parameters
    ----------
    image:
        Input BGR array.
    try_rotation:
        Include :func:`enhance_rotated_qr` in the candidates.
    try_low_light:
        Include :func:`enhance_low_light_qr` in the candidates.
    try_blur:
        Include :func:`enhance_blurred_qr` in the candidates.
    try_contrast:
        Include :func:`enhance_contrast_qr` in the candidates.

    Returns
    -------
    EnhancementResult
        The enhancement result with the highest Laplacian variance score.
        If no function improves on the raw image, returns the raw image
        wrapped in an ``EnhancementResult`` with ``technique="none"``.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "auto_enhance: expected a 3-channel BGR array, "
            f"got shape {getattr(image, 'shape', None)}"
        )

    t0 = time.perf_counter()
    baseline_score = _laplacian_variance(image)
    candidates: list[EnhancementResult] = []

    if try_rotation:
        try:
            candidates.append(enhance_rotated_qr(image))
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto_enhance: rotation step failed — %s", exc)

    if try_low_light:
        try:
            candidates.append(enhance_low_light_qr(image))
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto_enhance: low-light step failed — %s", exc)

    if try_blur:
        try:
            candidates.append(enhance_blurred_qr(image))
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto_enhance: blur step failed — %s", exc)

    if try_contrast:
        try:
            candidates.append(enhance_contrast_qr(image))
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto_enhance: contrast step failed — %s", exc)

    best_result: Optional[EnhancementResult] = None
    best_score = baseline_score

    for candidate in candidates:
        score = _laplacian_variance(candidate.enhanced_image)
        if score > best_score:
            best_score = score
            best_result = candidate

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if best_result is None:
        logger.info(
            "auto_enhance — no enhancement improved the image; "
            "returning raw (score=%.2f)  [%.1f ms]",
            baseline_score, elapsed_ms,
        )
        return EnhancementResult(
            enhanced_image=image,
            technique="none",
            params={"baseline_score": baseline_score},
            elapsed_ms=elapsed_ms,
        )

    logger.info(
        "auto_enhance — best technique='%s' (score=%.2f > baseline=%.2f)  "
        "[%.1f ms total]",
        best_result.technique, best_score, baseline_score, elapsed_ms,
    )
    return best_result


# ===========================================================================
# Example / CLI entry-point
# ===========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "\nUsage: python qr_enhancement.py <image_path> [technique]\n"
            "\n"
            "Techniques:\n"
            "  rotation    – best-angle rotation sweep\n"
            "  low_light   – CLAHE low-light enhancement\n"
            "  blur        – unsharp-mask blur recovery\n"
            "  contrast    – linear contrast stretching\n"
            "  sharpness   – convolution sharpening\n"
            "  auto        – try all; pick best (default)\n"
            "  benchmark   – benchmark all techniques against detect_qr\n"
            "\n"
            "Example:\n"
            "  python qr_enhancement.py test_qr.jpg low_light\n"
            "  python qr_enhancement.py dark_qr.png benchmark\n"
        )
        sys.exit(0)

    image_path = sys.argv[1]
    technique  = sys.argv[2].lower() if len(sys.argv) > 2 else "auto"

    # Load raw image
    raw = _load_image_from_path(image_path)

    TECHNIQUE_MAP: dict[str, Callable[[BgrArray], EnhancementResult]] = {
        "rotation":  enhance_rotated_qr,
        "low_light": enhance_low_light_qr,
        "blur":      enhance_blurred_qr,
        "contrast":  enhance_contrast_qr,
        "sharpness": enhance_sharpness_qr,
        "auto":      auto_enhance,
    }

    if technique == "benchmark":
        print("\n── Benchmarking all individual techniques ──\n")
        for name, fn in TECHNIQUE_MAP.items():
            if name == "auto":
                continue
            benchmark_detectors(image_path, fn, enhance_fn_name=name)
        sys.exit(0)

    if technique not in TECHNIQUE_MAP:
        print(f"Unknown technique '{technique}'. Run without args to see options.")
        sys.exit(1)

    result = TECHNIQUE_MAP[technique](raw)

    stem   = Path(image_path).stem
    suffix = Path(image_path).suffix
    out    = f"enhanced_{stem}_{result.technique}{suffix}"
    cv2.imwrite(out, result.enhanced_image)

    print(f"\n✅ Enhancement complete")
    print(f"   Technique : {result.technique}")
    print(f"   Elapsed   : {result.elapsed_ms:.1f} ms")
    print(f"   Params    : {result.params}")
    print(f"   Output    : {out}\n")
