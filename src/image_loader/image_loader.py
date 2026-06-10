"""
image_loader.py
---------------
Week 1 - Member 1: Image Loader Module
Project: QR Code Tamper Detection using Computer Vision

This module handles loading, validating, and preprocessing
JPG and PNG images for the QR Code tamper detection pipeline.

Deliverable: Input Image → Successfully Loaded Image Object
"""

import os
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


# ── Supported formats ──────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_MODES = {"RGB", "RGBA", "L", "P"}   # Pillow colour modes we accept


# ── Custom Exceptions ──────────────────────────────────────────────────────────
class ImageLoaderError(Exception):
    """Base exception for all image-loader errors."""


class InvalidFileError(ImageLoaderError):
    """Raised when the file cannot be opened or decoded as an image."""


class UnsupportedFormatError(ImageLoaderError):
    """Raised when the file extension is not JPG or PNG."""


class FileNotFoundError(ImageLoaderError):       # shadows built-in intentionally
    """Raised when the supplied path does not exist."""


# ── Core helpers ───────────────────────────────────────────────────────────────
def validate_image_file(file_path: str) -> str:
    """
    Validate that *file_path* points to a readable JPG or PNG file.

    Steps
    -----
    1. Check the path actually exists on disk.
    2. Confirm the file extension is .jpg / .jpeg / .png.
    3. Attempt to open with Pillow to catch corrupt / non-image files.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    str
        Canonical (absolute) path to the validated file.

    Raises
    ------
    FileNotFoundError       – path does not exist
    UnsupportedFormatError  – extension is not JPG/PNG
    InvalidFileError        – file is corrupt or not a real image
    """
    # 1. Existence check
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")

    # 2. Extension check
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}'. Only JPG and PNG are accepted."
        )

    # 3. Pillow integrity check
    try:
        with Image.open(file_path) as img:
            img.verify()           # catches truncated / corrupt files
    except UnidentifiedImageError:
        raise InvalidFileError(f"Cannot identify image file: '{file_path}'")
    except Exception as exc:
        raise InvalidFileError(f"Image validation failed for '{file_path}': {exc}")

    return os.path.abspath(file_path)


# ── Pillow loader ──────────────────────────────────────────────────────────────
def load_with_pillow(file_path: str) -> Image.Image:
    """
    Load an image using Pillow and normalise it to RGB.

    Pillow is great for metadata, colour-space conversion, and
    format-aware decoding (e.g., PNG transparency → RGBA → RGB).

    Parameters
    ----------
    file_path : str
        Path to a validated JPG or PNG file.

    Returns
    -------
    PIL.Image.Image
        An RGB image object ready for further processing.

    Raises
    ------
    InvalidFileError  – if Pillow cannot load the file.
    """
    validated_path = validate_image_file(file_path)

    try:
        img = Image.open(validated_path)

        # Convert palette / RGBA / grayscale → RGB so downstream code is uniform
        if img.mode not in ("RGB",):
            img = img.convert("RGB")

        # Force-load pixel data (Image.open is lazy by default)
        img.load()
        return img

    except (ImageLoaderError, UnsupportedFormatError, InvalidFileError):
        raise                          # re-raise our own exceptions unchanged
    except Exception as exc:
        raise InvalidFileError(f"Pillow failed to load '{file_path}': {exc}")


# ── OpenCV loader ─────────────────────────────────────────────────────────────
def load_with_opencv(file_path: str) -> np.ndarray:
    """
    Load an image using OpenCV (cv2).

    OpenCV loads images as BGR NumPy arrays by default; this function
    converts to the more common RGB ordering so results match Pillow.

    Parameters
    ----------
    file_path : str
        Path to a validated JPG or PNG file.

    Returns
    -------
    numpy.ndarray
        An (H, W, 3) uint8 array in **RGB** colour order.

    Raises
    ------
    InvalidFileError  – if OpenCV returns None (unreadable file).
    """
    validated_path = validate_image_file(file_path)

    bgr_image = cv2.imread(validated_path, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise InvalidFileError(
            f"OpenCV could not read the image at '{file_path}'. "
            "The file may be corrupt or an unsupported sub-format."
        )

    # Convert BGR → RGB for consistency with Pillow output
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


# ── Unified public API ────────────────────────────────────────────────────────
def load_image(file_path: str, backend: str = "opencv") -> dict:
    """
    Main entry point for the Image Loader Module.

    Loads a JPG or PNG image and returns a standardised result dictionary
    that the rest of the pipeline can consume without knowing which
    backend was used.

    Parameters
    ----------
    file_path : str
        Path to the image file.
    backend : str, optional
        ``"opencv"`` (default) or ``"pillow"``.

    Returns
    -------
    dict with keys:
        ``path``        – absolute path (str)
        ``backend``     – which library loaded it (str)
        ``width``       – image width in pixels (int)
        ``height``      – image height in pixels (int)
        ``channels``    – number of colour channels (int)
        ``format``      – file extension, lowercase (str)
        ``numpy_array`` – pixel data as np.ndarray (H×W×C, uint8)
        ``pil_image``   – PIL.Image.Image object (or None for opencv-only)
        ``status``      – ``"success"``

    Raises
    ------
    ImageLoaderError (or subclass) on any failure.
    """
    backend = backend.lower()
    if backend not in ("opencv", "pillow"):
        raise ValueError(f"Unknown backend '{backend}'. Choose 'opencv' or 'pillow'.")

    abs_path = validate_image_file(file_path)
    _, ext = os.path.splitext(abs_path)

    pil_img = None
    numpy_array = None

    if backend == "pillow":
        pil_img = load_with_pillow(abs_path)
        numpy_array = np.array(pil_img)
    else:
        numpy_array = load_with_opencv(abs_path)

    height, width = numpy_array.shape[:2]
    channels = 1 if numpy_array.ndim == 2 else numpy_array.shape[2]

    return {
        "path": abs_path,
        "backend": backend,
        "width": width,
        "height": height,
        "channels": channels,
        "format": ext.lower().lstrip("."),
        "numpy_array": numpy_array,
        "pil_image": pil_img,
        "status": "success",
    }


# ── Convenience wrappers ──────────────────────────────────────────────────────
def load_jpg(file_path: str, backend: str = "opencv") -> dict:
    """Wrapper that explicitly states intent: load a JPG file."""
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in (".jpg", ".jpeg"):
        raise UnsupportedFormatError(
            f"load_jpg expects a .jpg/.jpeg file, got '{ext}'"
        )
    return load_image(file_path, backend=backend)


def load_png(file_path: str, backend: str = "opencv") -> dict:
    """Wrapper that explicitly states intent: load a PNG file."""
    _, ext = os.path.splitext(file_path)
    if ext.lower() != ".png":
        raise UnsupportedFormatError(
            f"load_png expects a .png file, got '{ext}'"
        )
    return load_image(file_path, backend=backend)


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_loader.py <image_path> [opencv|pillow]")
        sys.exit(0)

    path    = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "opencv"

    try:
        result = load_image(path, backend=backend)
        print(f"\n✅ Successfully Loaded")
        print(f"   Path     : {result['path']}")
        print(f"   Format   : {result['format'].upper()}")
        print(f"   Size     : {result['width']} × {result['height']} px")
        print(f"   Channels : {result['channels']}")
        print(f"   Backend  : {result['backend']}")
        print(f"   Status   : {result['status']}\n")
    except ImageLoaderError as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)