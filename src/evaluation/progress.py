"""
progress.py
===========
Evaluation Framework — terminal progress reporting.

A minimal, dependency-free progress bar with ETA, images/sec, and elapsed
time, printed to stderr so it doesn't interleave with piped stdout output.
Moved out of ``evaluate_dataset.py`` so the orchestrator doesn't own
presentation-layer console logic directly (single-responsibility).
"""

from __future__ import annotations

import sys
import time

from src.evaluation.config import PROGRESS_BAR_WIDTH, PROGRESS_REFRESH_INTERVAL_SECONDS
from src.evaluation.utils import format_duration


class ProgressReporter:
    """Dependency-free progress bar with ETA, images/sec, and elapsed time.

    Redraws are throttled to ``config.PROGRESS_REFRESH_INTERVAL_SECONDS`` so
    very fast runs (e.g. duplicate-skipped images) don't flood the terminal;
    the final ``update()`` call and :meth:`finish` always draw immediately.
    """

    def __init__(self, total: int) -> None:
        self.total = max(total, 1)
        self.done = 0
        self.start = time.perf_counter()
        self._last_draw = 0.0

    def update(self, category: str) -> None:
        self.done += 1
        now = time.perf_counter()
        is_last = self.done >= self.total
        if not is_last and (now - self._last_draw) < PROGRESS_REFRESH_INTERVAL_SECONDS:
            return
        self._last_draw = now

        elapsed = now - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else 0.0
        pct = 100.0 * self.done / self.total
        filled = int(PROGRESS_BAR_WIDTH * self.done / self.total)
        bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
        sys.stderr.write(
            f"\r  [{bar}] {pct:5.1f}%  ({self.done}/{self.total})  "
            f"category={category:<20}  {rate:5.1f} img/s  "
            f"elapsed {format_duration(elapsed)}  ETA {format_duration(remaining)}   "
        )
        sys.stderr.flush()

    def finish(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()