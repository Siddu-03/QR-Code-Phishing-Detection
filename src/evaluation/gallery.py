"""
gallery.py
==========
Evaluation Framework — visual gallery generation.

Copies a bounded sample of evaluated images into
``results/<dataset>/gallery/{detected,failed,high_risk,tampered}/`` so a
human reviewer can quickly eyeball representative successes/failures/risky
detections without digging through the full dataset. Copying (not just
listing paths) is deliberate: the gallery is meant to be browsable on its
own, including after the original dataset — especially a temporary ZIP
extraction — has been cleaned up.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from src.evaluation.config import GALLERY_HIGH_RISK_LEVELS, GALLERY_MAX_IMAGES_PER_CATEGORY, GALLERY_TAMPERED_LEVELS
from src.evaluation.utils import ensure_dir, safe_stem

logger = logging.getLogger("evaluation.gallery")


def _copy_capped(rows: list[dict[str, Any]], dest_dir: Path, cap: int | None) -> int:
    dest_dir = ensure_dir(dest_dir)
    copied = 0
    for row in rows:
        if cap is not None and copied >= cap:
            break
        src = Path(row["image_path"])
        if not src.exists():
            continue  # e.g. a ZIP-extracted temp file already cleaned up
        dest_name = f"{safe_stem(src)}_{row.get('category', 'uncategorized')}{src.suffix}"
        try:
            shutil.copy2(src, dest_dir / dest_name)
            copied += 1
        except OSError as exc:
            logger.warning("Could not copy %s into gallery: %s", src, exc)
    return copied


def copy_failed_images(results: list[dict[str, Any]], failed_images_dir: str | Path) -> int:
    """Copy every failed image into ``results/<dataset>/failed_images/`` (uncapped).

    Distinct from ``gallery/failed/`` (a bounded, browsable sample built by
    :func:`build_galleries`): this is the complete, uncapped set of images
    that failed the pipeline, intended for debugging rather than quick
    visual review.
    """
    failed_rows = [r for r in results if not r.get("success")]
    return _copy_capped(failed_rows, Path(failed_images_dir), cap=None)


def build_galleries(results: list[dict[str, Any]], gallery_root: str | Path) -> dict[str, int]:
    """Populate the four gallery subfolders from a run's result rows.

    Parameters
    ----------
    results : list[dict]
        Per-image result rows as produced by ``evaluate_dataset.process_single_image``.
    gallery_root : str | Path
        The ``results/<dataset>/gallery/`` directory (created if needed).

    Returns
    -------
    dict[str, int]
        Number of images copied into each of ``detected``/``failed``/
        ``high_risk``/``tampered``.
    """
    gallery_root = Path(gallery_root)

    detected_rows = [r for r in results if r.get("success") and r.get("detected")]
    failed_rows = [r for r in results if not r.get("success")]
    high_risk_rows = [
        r for r in results if r.get("success") and r.get("risk_level") in GALLERY_HIGH_RISK_LEVELS
    ]
    tampered_rows = [
        r for r in results if r.get("success") and r.get("risk_level") in GALLERY_TAMPERED_LEVELS
    ]

    counts = {
        "detected": _copy_capped(detected_rows, gallery_root / "detected", GALLERY_MAX_IMAGES_PER_CATEGORY),
        "failed": _copy_capped(failed_rows, gallery_root / "failed", GALLERY_MAX_IMAGES_PER_CATEGORY),
        "high_risk": _copy_capped(high_risk_rows, gallery_root / "high_risk", GALLERY_MAX_IMAGES_PER_CATEGORY),
        "tampered": _copy_capped(tampered_rows, gallery_root / "tampered", GALLERY_MAX_IMAGES_PER_CATEGORY),
    }
    logger.info("Gallery populated: %s", counts)
    return counts