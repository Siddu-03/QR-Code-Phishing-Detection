"""
checkpoint.py
=============
Evaluation Framework — checkpoint and resume system.

Large evaluation runs can take a long time; this module lets a run be
interrupted (crash, Ctrl-C, machine reboot) and resumed without re-running
the pipeline on images that already completed. A checkpoint is a single
JSON file (``results/<dataset>/checkpoints/resume.json``) containing the
dataset root, every completed result row, and a set of already-processed
image paths.

Design choice: the checkpoint stores full result rows (not just "done"
markers), so resuming a run doesn't just skip work — it also restores the
in-memory ``results`` list exactly as if the run had never stopped, and
the final CSV/JSON/report outputs are identical either way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.utils import write_json

logger = logging.getLogger("evaluation.checkpoint")


@dataclass
class Checkpoint:
    """In-memory representation of a resumable evaluation run."""

    dataset_root: str
    completed_paths: set[str] = field(default_factory=set)
    results: list[dict[str, Any]] = field(default_factory=list)

    def mark_done(self, result: dict[str, Any]) -> None:
        """Record one completed image result."""
        self.completed_paths.add(result["image_path"])
        self.results.append(result)

    def is_done(self, image_path: str) -> bool:
        return image_path in self.completed_paths

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "completed_paths": sorted(self.completed_paths),
            "results": self.results,
        }


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> Path:
    """Persist *checkpoint* to *path* as JSON. Safe to call repeatedly (overwrites)."""
    return write_json(checkpoint.to_dict(), path)


def load_checkpoint(path: str | Path) -> Checkpoint | None:
    """Load a previously saved checkpoint, or return ``None`` if *path* doesn't exist.

    A malformed checkpoint file is logged and treated as "no checkpoint"
    (fresh start) rather than raising, since a corrupt resume file should
    never block a re-run of the evaluation.
    """
    import json

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None

    try:
        with checkpoint_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return Checkpoint(
            dataset_root=data["dataset_root"],
            completed_paths=set(data.get("completed_paths", [])),
            results=list(data.get("results", [])),
        )
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        logger.warning("Could not load checkpoint at %s (%s) — starting fresh.", checkpoint_path, exc)
        return None


def should_checkpoint(completed_count: int, interval: int) -> bool:
    """Return True if a checkpoint should be written after *completed_count* completions."""
    return interval > 0 and completed_count % interval == 0