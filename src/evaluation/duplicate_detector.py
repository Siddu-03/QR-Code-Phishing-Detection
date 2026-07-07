"""
duplicate_detector.py
======================
Evaluation Framework — duplicate image handling.

Datasets assembled from multiple sources (or extracted from ZIPs someone
re-zipped) frequently contain byte-identical images under different
filenames or categories. This module fingerprints each image's file bytes
so ``evaluate_dataset.py`` can run the pipeline once per unique image and
copy the result to every duplicate, instead of re-running the (potentially
expensive) pipeline on content it has already scored.

Detection is exact-content (hash-based), not perceptual/near-duplicate —
this keeps it fast, dependency-free, and unambiguous: a hash match means
the files are byte-for-byte identical, so reusing the first result is
always correct, never an approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.evaluation.dataset_loader import ImageRecord
from src.evaluation.utils import hash_file

logger = logging.getLogger("evaluation.duplicate_detector")


@dataclass
class DuplicateReport:
    """Result of scanning a dataset for exact-content duplicates.

    ``unique_records`` is the list to actually run through the pipeline;
    ``duplicate_map`` maps every duplicate image's path to the path of the
    first (canonical) occurrence with the same content hash, so its result
    can be cloned after the canonical image is processed.
    """

    unique_records: list[ImageRecord]
    duplicate_map: dict[str, str] = field(default_factory=dict)  # duplicate_path -> canonical_path
    hash_by_path: dict[str, str] = field(default_factory=dict)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_map)

    def to_dict(self) -> dict:
        return {
            "unique_image_count": len(self.unique_records),
            "duplicate_count": self.duplicate_count,
            "duplicate_map": self.duplicate_map,
        }


def find_duplicates(records: list[ImageRecord]) -> DuplicateReport:
    """Scan *records* for exact-content duplicates via a streaming file hash.

    Parameters
    ----------
    records : list[ImageRecord]
        Images discovered by ``dataset_loader.discover_images``.

    Returns
    -------
    DuplicateReport
    """
    seen_hash_to_path: dict[str, str] = {}
    duplicate_map: dict[str, str] = {}
    hash_by_path: dict[str, str] = {}
    unique_records: list[ImageRecord] = []

    for record in records:
        try:
            digest = hash_file(record.path)
        except OSError as exc:
            logger.warning("Could not hash %s for duplicate detection: %s", record.path, exc)
            unique_records.append(record)
            continue

        hash_by_path[record.path] = digest
        if digest in seen_hash_to_path:
            canonical_path = seen_hash_to_path[digest]
            duplicate_map[record.path] = canonical_path
            logger.debug("Duplicate detected: %s == %s", record.path, canonical_path)
        else:
            seen_hash_to_path[digest] = record.path
            unique_records.append(record)

    if duplicate_map:
        logger.info(
            "Duplicate detection: %d duplicate(s) found among %d image(s); "
            "%d unique image(s) will be processed.",
            len(duplicate_map), len(records), len(unique_records),
        )

    return DuplicateReport(
        unique_records=unique_records, duplicate_map=duplicate_map, hash_by_path=hash_by_path
    )


def clone_result_for_duplicates(
    canonical_result: dict, duplicate_path: str, category: str, ground_truth: dict[str, bool]
) -> dict:
    """Build a result row for a duplicate image by cloning the canonical image's result.

    Every pipeline-derived field (detection, risk, URL analysis) is copied
    as-is — same content bytes guarantee the same pipeline outcome. The
    identifying fields (``image_path``, ``category``) and ground truth
    (``expect_qr``/``expect_malicious``/``predicted_malicious``) are
    recomputed from *this* image's own category, since ground truth is a
    property of where the duplicate was catalogued, not of the canonical
    copy's category. Per-image timings are zeroed since the pipeline did
    not actually run on this file.
    """
    cloned = dict(canonical_result)
    cloned["image_path"] = duplicate_path
    cloned["category"] = category
    cloned["duplicate_of"] = canonical_result.get("image_path")
    cloned["timings_ms"] = {}
    cloned["total_time_ms"] = 0.0
    cloned["expect_qr"] = ground_truth["expect_qr"]
    cloned["expect_malicious"] = ground_truth["expect_malicious"]
    return cloned