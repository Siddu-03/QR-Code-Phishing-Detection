"""
dataset_loader.py
==================
Evaluation Framework — dataset management.

Handles everything about turning a CLI-supplied ``dataset_root`` argument
into a concrete directory of validated images:

* accepting either a plain folder (``datasets/<folder>``) or a ``.zip``
  archive (``datasets/<name>.zip``), extracting the latter into a managed
  temporary directory that is cleaned up after the run;
* recursive (or one-level) traversal and category assignment from
  subfolder names;
* per-file validation (decodable image, supported extension) with
  corrupted/unsupported files recorded and skipped rather than aborting
  the run;
* dataset-wide statistics printed before evaluation begins.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2

from src.evaluation.config import SUPPORTED_ARCHIVE_EXTENSIONS
from src.evaluation.utils import is_supported_image

logger = logging.getLogger("evaluation.dataset_loader")


@dataclass
class ImageRecord:
    """A single validated image discovered in the dataset."""

    path: str
    category: str
    relative_path: str
    size_bytes: int
    width: int
    height: int


@dataclass
class SkippedFile:
    """A file that was discovered but excluded from the run, with a reason."""

    path: str
    reason: str


@dataclass
class DatasetStats:
    """Summary statistics for a discovered dataset, printed before execution."""

    root: str
    categories: dict[str, int] = field(default_factory=dict)
    total_images: int = 0
    total_bytes: int = 0
    skipped: list[SkippedFile] = field(default_factory=list)

    def pretty_print(self) -> str:
        lines = [
            "=" * 60,
            "  Dataset discovery summary",
            "=" * 60,
            f"  Root directory : {self.root}",
            f"  Categories     : {len(self.categories)}",
        ]
        for category, count in sorted(self.categories.items()):
            lines.append(f"    - {category:<22} {count:>5} image(s)")
        lines.append(f"  Total images   : {self.total_images}")
        lines.append(f"  Total size     : {self.total_bytes / (1024 * 1024):.2f} MB")
        if self.skipped:
            lines.append(f"  Skipped files  : {len(self.skipped)}")
            for s in self.skipped[:10]:
                lines.append(f"    - {s.path}: {s.reason}")
            if len(self.skipped) > 10:
                lines.append(f"    ... and {len(self.skipped) - 10} more")
        lines.append("=" * 60)
        return "\n".join(lines)


# ===========================================================================
# Archive support
# ===========================================================================

def is_archive(path: str | Path) -> bool:
    """Return True if *path* has a supported archive extension (currently ``.zip``)."""
    return Path(path).suffix.lower() in SUPPORTED_ARCHIVE_EXTENSIONS


def extract_archive(archive_path: str | Path) -> Path:
    """Extract *archive_path* into a new managed temporary directory.

    Parameters
    ----------
    archive_path : str | Path
        Path to a ``.zip`` file.

    Returns
    -------
    Path
        The temporary directory the archive was extracted into. Caller is
        responsible for cleanup via :func:`cleanup_temp_dir` (or by using
        :func:`resolve_dataset_source` as a context manager, which handles
        this automatically).

    Raises
    ------
    FileNotFoundError
        If *archive_path* does not exist.
    zipfile.BadZipFile
        If the archive is corrupt / not a valid ZIP.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="qrshield_eval_"))
    logger.info("Extracting archive %s -> %s", archive_path, temp_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(temp_dir)
    return temp_dir


def cleanup_temp_dir(path: str | Path) -> None:
    """Remove a temporary directory tree created by :func:`extract_archive`, ignoring errors."""
    try:
        shutil.rmtree(path, ignore_errors=True)
        logger.debug("Cleaned up temporary directory: %s", path)
    except Exception as exc:  # noqa: BLE001 — cleanup must never crash the run
        logger.warning("Failed to clean up temporary directory %s: %s", path, exc)


def _dataset_name_from_path(path: Path) -> str:
    """Derive a dataset name for the results folder from a folder or archive path."""
    return path.stem if path.is_file() else path.name


@contextmanager
def resolve_dataset_source(input_path: str | Path) -> Iterator[tuple[Path, str]]:
    """Resolve *input_path* (a folder or a ``.zip`` archive) to a usable dataset root.

    Yields ``(dataset_root, dataset_name)`` and guarantees any temporary
    extraction directory created for a ``.zip`` input is cleaned up when
    the ``with`` block exits, success or failure.

    Parameters
    ----------
    input_path : str | Path
        Either a directory (``datasets/<folder>``) or a ``.zip`` archive
        (``datasets/<name>.zip``).

    Yields
    ------
    (Path, str)
        The concrete directory to run discovery against, and the dataset
        name to use for the ``results/<dataset_name>/`` output folder.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    ValueError
        If *input_path* is a file with an unsupported (non-archive) extension.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_dir():
        yield path, _dataset_name_from_path(path)
        return

    if is_archive(path):
        temp_dir = extract_archive(path)
        try:
            yield temp_dir, _dataset_name_from_path(path)
        finally:
            cleanup_temp_dir(temp_dir)
        return

    raise ValueError(
        f"Unsupported dataset input '{path}'. Expected a directory or one of "
        f"{sorted(SUPPORTED_ARCHIVE_EXTENSIONS)}."
    )


# ===========================================================================
# Image discovery
# ===========================================================================

def discover_images(root: str | Path, recursive: bool = True) -> tuple[list[ImageRecord], DatasetStats]:
    """Discover and validate every image under *root*.

    Directory layout is expected to be ``<root>/<category>/*.<ext>``. The
    immediate child directory name of *root* is used as the category label;
    files directly inside *root* (no category subfolder) are assigned the
    category ``"uncategorized"``.

    Parameters
    ----------
    root : str | Path
        Dataset root directory (e.g. ``data/evaluation``, or the temporary
        extraction directory produced by :func:`resolve_dataset_source` for
        a ``.zip`` input).
    recursive : bool
        When True (default), search all nested subdirectories. When False,
        only look one level deep (``root/<category>/*``).

    Returns
    -------
    (list[ImageRecord], DatasetStats)
        The valid, loadable images and a summary statistics object. Corrupt
        or unsupported files are excluded from the returned list and
        recorded in ``DatasetStats.skipped`` instead of raising.

    Raises
    ------
    FileNotFoundError
        If *root* does not exist.
    NotADirectoryError
        If *root* is not a directory.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root_path}")

    pattern = "**/*" if recursive else "*/*"
    candidates = sorted(p for p in root_path.glob(pattern) if p.is_file())

    records: list[ImageRecord] = []
    stats = DatasetStats(root=str(root_path))

    for candidate in candidates:
        if not is_supported_image(candidate):
            stats.skipped.append(SkippedFile(str(candidate), "unsupported extension"))
            continue

        try:
            relative = candidate.relative_to(root_path)
        except ValueError:
            relative = candidate

        parts = relative.parts
        category = parts[0] if len(parts) > 1 else "uncategorized"

        image = cv2.imread(str(candidate))
        if image is None:
            stats.skipped.append(SkippedFile(str(candidate), "corrupted or undecodable"))
            logger.warning("Skipping corrupted/unreadable image: %s", candidate)
            continue

        h, w = image.shape[:2]
        size_bytes = candidate.stat().st_size

        records.append(
            ImageRecord(
                path=str(candidate.resolve()),
                category=category,
                relative_path=str(relative),
                size_bytes=size_bytes,
                width=w,
                height=h,
            )
        )
        stats.categories[category] = stats.categories.get(category, 0) + 1
        stats.total_images += 1
        stats.total_bytes += size_bytes

    if stats.total_images == 0:
        logger.warning("No valid images discovered under %s", root_path)

    return records, stats