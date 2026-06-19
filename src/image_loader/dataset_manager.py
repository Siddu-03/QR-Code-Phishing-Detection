"""
dataset_manager.py

Member 1 - Dataset Management
Branch: feature/dataset-management

Project: Computer Vision-Based Graphic Tamper Detection for QR Code
Phishing Prevention

This module owns everything related to the raw dataset that feeds the
tamper-detection pipeline: discovering images on disk, validating that
they are usable, extracting per-image metadata, computing dataset-wide
statistics, and loading images in batches for downstream stages
(preprocessing, QR enhancement, tamper analysis).

Expected on-disk layout (relative to the configured `data_root`):

    data/
    ├── normal/
    ├── rotated/
    ├── low_light/
    ├── blurry/
    ├── overlay/
    └── barcode/

Each subfolder is treated as a distinct *class label* corresponding to a
category of QR code image condition. Images may be .png, .jpg, .jpeg,
.bmp, .tiff, or .webp.

Public API
----------
DatasetManager
    .scan()                    -> discover files and build the internal index
    .validate_dataset()        -> check integrity of every discovered image
    .extract_metadata(path)    -> per-image metadata dict
    .get_statistics()          -> dataset-wide statistics dict
    .iter_batches(batch_size)  -> generator yielding batches of loaded images
    .load_batch(paths)         -> load a specific list of image paths
    .get_image_paths(label)    -> list of paths for one label (or all)
    .train_val_test_split(...) -> stratified split into train/val/test sets
    .export_manifest(path)     -> write the full index + metadata to JSON/CSV
"""
#nikhil

from __future__ import annotations

import csv
import json
import logging
import os
import random
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV (cv2) is required by dataset_manager.py. "
        "Install it with `pip install opencv-python --break-system-packages`."
    ) from exc


logger = logging.getLogger("dataset_manager")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                           datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

DEFAULT_LABELS = [
    "normal",
    "rotated",
    "low_light",
    "blurry",
    "overlay",
    "barcode",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    """Metadata for a single image discovered in the dataset."""

    path: str
    label: str
    filename: str
    width: int = 0
    height: int = 0
    channels: int = 0
    file_size_bytes: int = 0
    extension: str = ""
    mean_brightness: float = 0.0
    std_brightness: float = 0.0
    aspect_ratio: float = 0.0
    sha256: str = ""
    is_valid: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetStats:
    """Aggregate statistics for the whole dataset (or a subset of it)."""

    total_images: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    per_label_counts: Dict[str, int] = field(default_factory=dict)
    per_label_invalid: Dict[str, int] = field(default_factory=dict)
    mean_width: float = 0.0
    mean_height: float = 0.0
    mean_brightness: float = 0.0
    std_brightness: float = 0.0
    min_resolution: Tuple[int, int] = (0, 0)
    max_resolution: Tuple[int, int] = (0, 0)
    total_size_mb: float = 0.0
    duplicate_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["min_resolution"] = list(self.min_resolution)
        d["max_resolution"] = list(self.max_resolution)
        return d


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DatasetManager:
    """
    Owns discovery, validation, metadata extraction, statistics, and batch
    loading for the QR-tamper-detection image dataset.

    Parameters
    ----------
    data_root : str | Path
        Root directory containing one subfolder per label
        (normal, rotated, low_light, blurry, overlay, barcode).
    labels : Sequence[str], optional
        Override the set of expected label subfolders. Defaults to
        DEFAULT_LABELS. Folders not in this list are ignored; missing
        folders are reported as warnings but do not raise.
    compute_hash : bool, default True
        Whether to compute a SHA-256 hash per image during validation,
        used for duplicate detection. Disable for very large datasets
        if speed matters more than dedup checks.
    seed : int, default 42
        Random seed used for shuffling / splitting, for reproducibility.
    """

    def __init__(
        self,
        data_root: str | Path,
        labels: Optional[Sequence[str]] = None,
        compute_hash: bool = True,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.labels = list(labels) if labels else list(DEFAULT_LABELS)
        self.compute_hash = compute_hash
        self.seed = seed

        self._records: Dict[str, ImageRecord] = {}  # path -> record
        self._scanned = False

        random.seed(self.seed)

    # ------------------------------------------------------------------
    # Discovery / organization
    # ------------------------------------------------------------------

    def scan(self) -> int:
        """
        Walk `data_root`, discover all supported image files under each
        label subfolder, and build the internal index. Does NOT validate
        image content (use `validate_dataset` for that) — this step only
        discovers files and records lightweight filesystem metadata.

        Returns
        -------
        int: total number of image files discovered.
        """
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"data_root '{self.data_root}' does not exist. "
                f"Expected subfolders: {self.labels}"
            )

        self._records.clear()
        found = 0

        for label in self.labels:
            label_dir = self.data_root / label
            if not label_dir.is_dir():
                logger.warning("Expected label folder missing: %s", label_dir)
                continue

            for entry in sorted(label_dir.rglob("*")):
                if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                    rec = ImageRecord(
                        path=str(entry),
                        label=label,
                        filename=entry.name,
                        extension=entry.suffix.lower(),
                        file_size_bytes=entry.stat().st_size,
                    )
                    self._records[str(entry)] = rec
                    found += 1

        self._scanned = True
        logger.info("Scan complete: %d images discovered across %d labels.",
                    found, len(self.labels))
        return found

    def _ensure_scanned(self) -> None:
        if not self._scanned:
            self.scan()

    def reorganize_unsorted(self, unsorted_dir: str | Path, label: str,
                             move: bool = False) -> int:
        """
        Helper for dataset organization: move or copy every supported image
        from `unsorted_dir` into `data_root/label/`. Useful when new raw
        captures land in a staging folder and need to be sorted into the
        canonical structure.

        Parameters
        ----------
        unsorted_dir : str | Path
            Folder containing new, unsorted images.
        label : str
            Target label subfolder (must be one of self.labels).
        move : bool, default False
            If True, move files; otherwise copy them.

        Returns
        -------
        int: number of files relocated.
        """
        import shutil

        if label not in self.labels:
            raise ValueError(f"Unknown label '{label}'. Valid labels: {self.labels}")

        src_dir = Path(unsorted_dir)
        dst_dir = self.data_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for entry in sorted(src_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                target = dst_dir / entry.name
                if target.exists():
                    target = dst_dir / f"{entry.stem}_{count}{entry.suffix}"
                if move:
                    shutil.move(str(entry), str(target))
                else:
                    shutil.copy2(str(entry), str(target))
                count += 1

        logger.info("Reorganized %d files into %s (move=%s)", count, dst_dir, move)
        self._scanned = False  # index is stale, force re-scan on next access
        return count

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def extract_metadata(self, path: str | Path) -> ImageRecord:
        """
        Open a single image and extract full metadata: dimensions,
        channels, brightness statistics, aspect ratio, and a content
        hash. Sets `is_valid` and `error` fields based on whether the
        image could be decoded.

        This does not require `scan()` to have been called first; it can
        be used standalone on any path.
        """
        path = str(path)
        label = Path(path).parent.name
        filename = Path(path).name
        extension = Path(path).suffix.lower()

        record = ImageRecord(
            path=path,
            label=label,
            filename=filename,
            extension=extension,
        )

        try:
            file_size = os.path.getsize(path)
            record.file_size_bytes = file_size
        except OSError as e:
            record.is_valid = False
            record.error = f"Cannot stat file: {e}"
            return record

        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                record.is_valid = False
                record.error = "cv2.imread returned None (corrupt or unsupported file)"
                return record

            if img.ndim == 2:
                h, w = img.shape
                channels = 1
            else:
                h, w, channels = img.shape

            record.width = int(w)
            record.height = int(h)
            record.channels = int(channels)
            record.aspect_ratio = round(w / h, 4) if h else 0.0

            gray = img if channels == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            record.mean_brightness = float(round(np.mean(gray), 4))
            record.std_brightness = float(round(np.std(gray), 4))

            if self.compute_hash:
                with open(path, "rb") as f:
                    record.sha256 = hashlib.sha256(f.read()).hexdigest()

            record.is_valid = True
            record.error = None

        except Exception as e:  # noqa: BLE001 - we want to capture any decode failure
            record.is_valid = False
            record.error = f"Failed to process image: {e}"

        return record

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_dataset(self, fail_fast: bool = False) -> Dict[str, ImageRecord]:
        """
        Validate every image discovered by `scan()`: confirm it decodes
        correctly, compute its metadata, and flag duplicates by content
        hash. Populates `self._records` with full ImageRecord metadata
        (replacing the lightweight scan-only entries).

        Parameters
        ----------
        fail_fast : bool, default False
            If True, raise on the first invalid image. If False (default),
            collect all errors and continue; invalid images are kept in
            the index with `is_valid=False` so they can be inspected or
            filtered out later.

        Returns
        -------
        Dict[str, ImageRecord]: path -> validated record.
        """
        self._ensure_scanned()

        if not self._records:
            logger.warning("No images found to validate. Did you call scan()?")
            return {}

        validated: Dict[str, ImageRecord] = {}
        seen_hashes: Dict[str, str] = {}  # hash -> first path that had it
        invalid_count = 0
        duplicate_count = 0

        for path in list(self._records.keys()):
            record = self.extract_metadata(path)

            if not record.is_valid:
                invalid_count += 1
                logger.warning("Invalid image: %s (%s)", path, record.error)
                if fail_fast:
                    raise ValueError(f"Validation failed for {path}: {record.error}")

            if record.is_valid and self.compute_hash and record.sha256:
                if record.sha256 in seen_hashes:
                    duplicate_count += 1
                    record.error = f"Duplicate of {seen_hashes[record.sha256]}"
                    logger.info("Duplicate detected: %s == %s",
                                path, seen_hashes[record.sha256])
                else:
                    seen_hashes[record.sha256] = path

            validated[path] = record

        self._records = validated
        logger.info(
            "Validation complete: %d valid, %d invalid, %d duplicates (of %d total).",
            len(validated) - invalid_count, invalid_count, duplicate_count,
            len(validated),
        )
        return validated

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> DatasetStats:
        """
        Compute dataset-wide statistics across all currently indexed
        images. If `validate_dataset()` has not been run yet, this will
        run it automatically so statistics reflect real image content
        rather than just file counts.
        """
        if not self._records:
            self._ensure_scanned()
        if self._records and not any(r.width for r in self._records.values()):
            # records exist but look unvalidated (no dimensions populated)
            self.validate_dataset()

        stats = DatasetStats()
        widths, heights, brightness_means = [], [], []
        min_res = (float("inf"), float("inf"))
        max_res = (0, 0)
        total_bytes = 0
        per_label_counts: Dict[str, int] = defaultdict(int)
        per_label_invalid: Dict[str, int] = defaultdict(int)
        seen_hashes: set = set()
        dup_count = 0

        for record in self._records.values():
            per_label_counts[record.label] += 1
            total_bytes += record.file_size_bytes

            if not record.is_valid:
                per_label_invalid[record.label] += 1
                continue

            if record.error and record.error.startswith("Duplicate of"):
                dup_count += 1

            widths.append(record.width)
            heights.append(record.height)
            brightness_means.append(record.mean_brightness)

            if record.width * record.height < min_res[0] * min_res[1]:
                min_res = (record.width, record.height)
            if record.width * record.height > max_res[0] * max_res[1]:
                max_res = (record.width, record.height)

        stats.total_images = len(self._records)
        stats.invalid_images = sum(per_label_invalid.values())
        stats.valid_images = stats.total_images - stats.invalid_images
        stats.per_label_counts = dict(per_label_counts)
        stats.per_label_invalid = dict(per_label_invalid)
        stats.duplicate_count = dup_count
        stats.total_size_mb = round(total_bytes / (1024 * 1024), 3)

        if widths:
            stats.mean_width = round(float(np.mean(widths)), 2)
            stats.mean_height = round(float(np.mean(heights)), 2)
            stats.mean_brightness = round(float(np.mean(brightness_means)), 4)
            stats.std_brightness = round(float(np.std(brightness_means)), 4)
            stats.min_resolution = (int(min_res[0]), int(min_res[1]))
            stats.max_resolution = (int(max_res[0]), int(max_res[1]))

        return stats

    def print_statistics(self) -> None:
        """Pretty-print dataset statistics to stdout / logger."""
        stats = self.get_statistics()
        lines = [
            "=" * 56,
            "DATASET STATISTICS",
            "=" * 56,
            f"Total images      : {stats.total_images}",
            f"Valid images       : {stats.valid_images}",
            f"Invalid images      : {stats.invalid_images}",
            f"Duplicate images    : {stats.duplicate_count}",
            f"Total size (MB)     : {stats.total_size_mb}",
            f"Mean resolution     : {stats.mean_width} x {stats.mean_height}",
            f"Min resolution      : {stats.min_resolution}",
            f"Max resolution      : {stats.max_resolution}",
            f"Mean brightness     : {stats.mean_brightness}",
            f"Brightness std dev  : {stats.std_brightness}",
            "-" * 56,
            "Per-label counts:",
        ]
        for label in self.labels:
            count = stats.per_label_counts.get(label, 0)
            invalid = stats.per_label_invalid.get(label, 0)
            lines.append(f"  {label:<12} : {count:>5} images  ({invalid} invalid)")
        lines.append("=" * 56)
        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_image_paths(self, label: Optional[str] = None,
                         valid_only: bool = True,
                         exclude_duplicates: bool = False) -> List[str]:
        """
        Return image paths, optionally filtered by label.

        Parameters
        ----------
        label : str, optional
            Restrict to a single label. None returns all labels.
        valid_only : bool, default True
            If True, only return paths for images that decoded
            successfully. If the dataset has not yet been validated
            (i.e. only `scan()` has run), every discovered file is
            treated as valid by default, since validity is unknown
            until `validate_dataset()` actually decodes it.
        exclude_duplicates : bool, default False
            If True, drop images flagged as duplicates during
            validation (a record whose `error` starts with
            "Duplicate of"). Duplicates are still valid, decodable
            images — this is a separate concern from validity.
        """
        self._ensure_scanned()
        dataset_validated = any(r.width for r in self._records.values())

        result = []
        for record in self._records.values():
            if label is not None and record.label != label:
                continue
            if valid_only and dataset_validated and not record.is_valid:
                continue
            if exclude_duplicates and record.error and record.error.startswith("Duplicate of"):
                continue
            result.append(record.path)
        return sorted(result)

    def get_records(self, label: Optional[str] = None) -> List[ImageRecord]:
        """Return ImageRecord objects, optionally filtered by label."""
        self._ensure_scanned()
        if label is None:
            return list(self._records.values())
        return [r for r in self._records.values() if r.label == label]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_image(self, path: str, target_size: Optional[Tuple[int, int]] = None,
                    color_mode: str = "bgr") -> np.ndarray:
        """
        Load a single image as a numpy array.

        Parameters
        ----------
        path : str
        target_size : (width, height), optional
            If given, resize the loaded image to this size.
        color_mode : {"bgr", "rgb", "gray"}
        """
        flag = cv2.IMREAD_GRAYSCALE if color_mode == "gray" else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        if color_mode == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        return img

    def load_batch(self, paths: Sequence[str],
                    target_size: Optional[Tuple[int, int]] = None,
                    color_mode: str = "bgr") -> Tuple[np.ndarray, List[str]]:
        """
        Load a specific list of image paths into a single stacked numpy
        array (when all images share the same shape, e.g. via target_size)
        or a list of arrays otherwise.

        Returns
        -------
        (images, loaded_paths)
            images : np.ndarray of shape (N, H, W, C) if target_size was
                     given (uniform shape), else an object-dtype array of
                     individually-shaped arrays.
            loaded_paths : the subset of `paths` that loaded successfully,
                     in the same order as `images`.
        """
        images = []
        loaded_paths = []
        for p in paths:
            try:
                img = self.load_image(p, target_size=target_size, color_mode=color_mode)
                images.append(img)
                loaded_paths.append(p)
            except ValueError as e:
                logger.warning("Skipping unreadable image in batch: %s", e)

        if not images:
            return np.empty((0,)), []

        if target_size is not None:
            stacked = np.stack(images, axis=0)
            return stacked, loaded_paths

        # Mixed shapes — return as an object array rather than forcing a stack.
        arr = np.empty(len(images), dtype=object)
        for i, im in enumerate(images):
            arr[i] = im
        return arr, loaded_paths

    def iter_batches(
        self,
        batch_size: int = 32,
        label: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        color_mode: str = "bgr",
        shuffle: bool = True,
    ) -> Generator[Tuple[np.ndarray, List[str]], None, None]:
        """
        Generator that yields (images, paths) batches for the requested
        label (or the whole dataset if label is None). Intended to be
        consumed directly by the preprocessing / model-training stages
        without loading the entire dataset into memory at once.

        Parameters
        ----------
        batch_size : int
        label : str, optional
            Restrict iteration to a single label/class folder.
        target_size : (width, height), optional
            Resize every image to this shape so batches stack cleanly.
            Defaults to (224, 224), a common CNN input size; pass None to
            keep native resolutions (batches will then be object arrays).
        color_mode : {"bgr", "rgb", "gray"}
        shuffle : bool, default True
            Shuffle file order before batching (uses self.seed).
        """
        paths = self.get_image_paths(label=label, valid_only=True)
        if shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(paths)

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            images, loaded_paths = self.load_batch(
                batch_paths, target_size=target_size, color_mode=color_mode
            )
            if len(loaded_paths) > 0:
                yield images, loaded_paths

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by_label: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Split the validated, deduplicated dataset into train/val/test path
        lists. By default the split is stratified per label so every
        class is proportionally represented in each split.

        Returns
        -------
        {"train": [...], "val": [...], "test": [...]}
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        self._ensure_scanned()
        rng = random.Random(self.seed)

        splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        if stratify_by_label:
            groups = {label: self.get_image_paths(label=label) for label in self.labels}
        else:
            groups = {"_all": self.get_image_paths(label=None)}

        for _, paths in groups.items():
            paths = list(paths)
            rng.shuffle(paths)
            n = len(paths)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            splits["train"].extend(paths[:n_train])
            splits["val"].extend(paths[n_train:n_train + n_val])
            splits["test"].extend(paths[n_train + n_val:])

        logger.info(
            "Split complete: train=%d, val=%d, test=%d",
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
        )
        return splits

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_manifest(self, output_path: str | Path) -> None:
        """
        Write the full dataset index (every ImageRecord) to disk as
        either JSON or CSV, inferred from the output_path extension.
        Useful for handing off a frozen snapshot of the dataset to
        teammates working on preprocessing / tamper-analysis branches.
        """
        self._ensure_scanned()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = [r.to_dict() for r in self._records.values()]

        if output_path.suffix.lower() == ".json":
            with open(output_path, "w") as f:
                json.dump(records, f, indent=2)
        elif output_path.suffix.lower() == ".csv":
            if not records:
                logger.warning("No records to export.")
                return
            fieldnames = list(records[0].keys())
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
        else:
            raise ValueError("output_path must end in .json or .csv")

        logger.info("Manifest exported to %s (%d records).", output_path, len(records))


# ---------------------------------------------------------------------------
# CLI entry point — quick manual sanity check / demo
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Dataset management utility for QR tamper-detection dataset."
    )
    parser.add_argument("data_root", help="Path to the dataset root folder.")
    parser.add_argument("--validate", action="store_true",
                         help="Run full validation (decode + hash every image).")
    parser.add_argument("--manifest", default=None,
                         help="Optional path to export a manifest (.json or .csv).")
    args = parser.parse_args()

    manager = DatasetManager(args.data_root)
    manager.scan()

    if args.validate:
        manager.validate_dataset()

    manager.print_statistics()

    if args.manifest:
        manager.export_manifest(args.manifest)


if __name__ == "__main__":
    _main()
