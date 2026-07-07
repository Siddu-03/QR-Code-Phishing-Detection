"""
utils.py
========
Evaluation Framework — shared, reusable helper functions.

Per the architecture rules, this module holds *functions only* — no
configuration constants (those live in ``config.py``) and no orchestration
logic (that lives in ``evaluate_dataset.py``). Every function here is a
small, dependency-light building block reused by two or more other modules:
structured logging setup, a wall-clock stage-timing context manager, the
category ground-truth loader, JSON/CSV writers, path/file helpers, and
formatting helpers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

from src.evaluation.config import (
    DEFAULT_CATEGORY_LABELS,
    DUPLICATE_HASH_ALGORITHM,
    FALLBACK_CATEGORY_LABEL,
    SUPPORTED_IMAGE_EXTENSIONS,
)


# ===========================================================================
# Ground-truth convention
# ===========================================================================

def load_category_labels(labels_path: str | Path | None) -> dict[str, dict[str, bool]]:
    """Load the category ground-truth mapping, optionally overridden by a JSON file.

    See ``config.DEFAULT_CATEGORY_LABELS`` for the built-in convention:
    every dataset category is assumed to contain images that *do* contain a
    QR code, and only ``phishing``/``overlay_attack`` are treated as
    malicious/tampered by default.

    Parameters
    ----------
    labels_path : str | Path | None
        Path to a JSON file mapping category name -> {"expect_qr": bool,
        "expect_malicious": bool}. When ``None``, the built-in defaults are
        returned unchanged. When provided, entries in the file are merged
        on top of the defaults (file wins on conflicts).

    Returns
    -------
    dict[str, dict[str, bool]]
        Combined mapping.
    """
    labels = {k: dict(v) for k, v in DEFAULT_CATEGORY_LABELS.items()}
    if labels_path is None:
        return labels

    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        overrides = json.load(fh)

    for category, spec in overrides.items():
        labels[category] = {
            "expect_qr": bool(spec.get("expect_qr", True)),
            "expect_malicious": bool(spec.get("expect_malicious", False)),
        }
    return labels


def ground_truth_for(category: str, labels: dict[str, dict[str, bool]]) -> dict[str, bool]:
    """Look up ground truth for *category*, falling back to ``config.FALLBACK_CATEGORY_LABEL``."""
    return labels.get(category, FALLBACK_CATEGORY_LABEL)


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(
    verbose: bool = False, log_file: str | Path | None = None
) -> logging.Logger:
    """Configure and return the root logger for the evaluation run.

    Parameters
    ----------
    verbose : bool
        When True, sets level to DEBUG; otherwise INFO.
    log_file : str | Path, optional
        When provided, also writes logs to this file (typically
        ``results/<dataset>/logs/evaluation.log``) in addition to stderr.

    Returns
    -------
    logging.Logger
        Configured logger named ``"evaluation"``.
    """
    logger = logging.getLogger("evaluation")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ===========================================================================
# Timing
# ===========================================================================

class StageTimer:
    """Accumulates named-stage timings (milliseconds) for a single image.

    Example
    -------
    ::

        timer = StageTimer()
        with timer.measure("load"):
            ...
        with timer.measure("detect"):
            ...
        print(timer.as_dict())
    """

    def __init__(self) -> None:
        self._timings: dict[str, float] = {}

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._timings[stage_name] = self._timings.get(stage_name, 0.0) + elapsed_ms

    def as_dict(self) -> dict[str, float]:
        return dict(self._timings)

    @property
    def total_ms(self) -> float:
        return sum(self._timings.values())


# ===========================================================================
# Serialisation helpers
# ===========================================================================

def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/enums/paths into JSON-serialisable primitives."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value") and hasattr(obj, "name") and not isinstance(obj, (int, str)):
        # Enum-like
        return obj.value
    return obj


def write_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write *data* (already JSON-safe or convertible via :func:`to_jsonable`) to *path*."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, indent=indent, ensure_ascii=False)
    return dest


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a list of flat dictionaries to a CSV file.

    The union of all keys across *rows* is used as the column header,
    preserving first-seen order. Missing keys in a given row are written
    as empty strings.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        dest.write_text("", encoding="utf-8")
        return dest

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_jsonable(v) for k, v in row.items()})
    return dest


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a short human-readable string (e.g. '1m 03s')."""
    seconds = max(0.0, seconds)
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ===========================================================================
# File / path helpers
# ===========================================================================

def is_supported_image(path: Path) -> bool:
    """Return True if *path*'s extension is a supported image type (case-insensitive)."""
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def hash_file(
    path: str | Path, algorithm: str = DUPLICATE_HASH_ALGORITHM, chunk_size: int = 65536
) -> str:
    """Compute a hex digest of *path*'s file bytes, streaming so large files are safe.

    Used by ``duplicate_detector.py`` for exact-content duplicate detection.
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it doesn't exist yet, and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_stem(path: str | Path) -> str:
    """Return a filesystem-safe stem for *path*, for use in generated filenames."""
    return Path(path).stem.replace(" ", "_")