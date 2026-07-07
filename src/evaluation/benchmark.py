"""
benchmark.py
============
Week 4+ — Evaluation Framework: performance benchmarking.

Aggregates per-image stage timings (produced by
``evaluate_dataset.process_single_image``) into per-category and overall
throughput/performance statistics: average/min/max/std-dev processing time,
fastest/slowest image, average FPS, pipeline throughput, and success/failure
rates.

Consumes plain result dictionaries (as produced by ``evaluate_dataset.py``)
rather than a specific class, so it has no import-time dependency on the
rest of the pipeline and stays independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.evaluation.metrics import NumericStats, compute_numeric_stats


@dataclass
class CategoryBenchmark:
    """Benchmark summary for a single dataset category."""

    category: str
    image_count: int
    success_count: int
    failure_count: int
    total_time_stats: NumericStats
    stage_stats: dict[str, NumericStats]
    fastest_image: str | None
    fastest_time_ms: float
    slowest_image: str | None
    slowest_time_ms: float

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.image_count) if self.image_count else 0.0

    @property
    def failure_rate(self) -> float:
        return (self.failure_count / self.image_count) if self.image_count else 0.0

    @property
    def average_fps(self) -> float:
        mean_ms = self.total_time_stats.mean
        return (1000.0 / mean_ms) if mean_ms > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "image_count": self.image_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "average_fps": round(self.average_fps, 3),
            "total_time_ms": self.total_time_stats.to_dict(),
            "stage_time_ms": {k: v.to_dict() for k, v in self.stage_stats.items()},
            "fastest_image": self.fastest_image,
            "fastest_time_ms": round(self.fastest_time_ms, 3),
            "slowest_image": self.slowest_image,
            "slowest_time_ms": round(self.slowest_time_ms, 3),
        }


@dataclass
class BenchmarkReport:
    """Full benchmark report: overall + per-category breakdown."""

    overall: CategoryBenchmark
    per_category: dict[str, CategoryBenchmark] = field(default_factory=dict)
    wall_clock_seconds: float = 0.0
    worker_count: int = 1

    @property
    def pipeline_throughput_ips(self) -> float:
        """Images processed per second across the whole run (wall-clock based)."""
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.overall.image_count / self.wall_clock_seconds

    def to_dict(self) -> dict:
        return {
            "overall": self.overall.to_dict(),
            "per_category": {k: v.to_dict() for k, v in self.per_category.items()},
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "worker_count": self.worker_count,
            "pipeline_throughput_images_per_sec": round(self.pipeline_throughput_ips, 3),
        }


STAGE_NAMES = (
    "load",
    "preprocess",
    "detect",
    "url_analyze",
    "risk_assess",
    "report",
)


def _benchmark_for_subset(category: str, rows: list[dict]) -> CategoryBenchmark:
    total_times = [r["total_time_ms"] for r in rows if r.get("success")]
    stage_stats: dict[str, NumericStats] = {}
    for stage in STAGE_NAMES:
        values = [
            r["timings_ms"][stage]
            for r in rows
            if r.get("success") and stage in r.get("timings_ms", {})
        ]
        stage_stats[stage] = compute_numeric_stats(values)

    success_rows = [r for r in rows if r.get("success")]
    fastest = min(success_rows, key=lambda r: r["total_time_ms"], default=None)
    slowest = max(success_rows, key=lambda r: r["total_time_ms"], default=None)

    return CategoryBenchmark(
        category=category,
        image_count=len(rows),
        success_count=len(success_rows),
        failure_count=len(rows) - len(success_rows),
        total_time_stats=compute_numeric_stats(total_times),
        stage_stats=stage_stats,
        fastest_image=fastest["image_path"] if fastest else None,
        fastest_time_ms=fastest["total_time_ms"] if fastest else 0.0,
        slowest_image=slowest["image_path"] if slowest else None,
        slowest_time_ms=slowest["total_time_ms"] if slowest else 0.0,
    )


def build_benchmark_report(
    results: list[dict], wall_clock_seconds: float, worker_count: int
) -> BenchmarkReport:
    """Build a full :class:`BenchmarkReport` from a flat list of per-image result dicts.

    Parameters
    ----------
    results : list[dict]
        Per-image results as produced by
        ``evaluate_dataset.process_single_image``. Each dict is expected to
        have at least: ``image_path``, ``category``, ``success``,
        ``total_time_ms``, ``timings_ms`` (dict keyed by stage name).
    wall_clock_seconds : float
        Actual wall-clock duration of the whole run (used for throughput,
        which differs from the sum of per-image times under parallelism).
    worker_count : int
        Number of worker processes/threads used for the run.

    Returns
    -------
    BenchmarkReport
    """
    by_category: dict[str, list[dict]] = {}
    for row in results:
        by_category.setdefault(row["category"], []).append(row)

    per_category = {
        category: _benchmark_for_subset(category, rows)
        for category, rows in sorted(by_category.items())
    }
    overall = _benchmark_for_subset("overall", results)

    return BenchmarkReport(
        overall=overall,
        per_category=per_category,
        wall_clock_seconds=wall_clock_seconds,
        worker_count=worker_count,
    )