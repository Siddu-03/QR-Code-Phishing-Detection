"""
generate_report.py
==================
Week 4+ — Evaluation Framework: research-ready markdown report generation.

Assembles a single markdown document from the outputs of ``metrics.py``,
``benchmark.py``, and ``confusion_matrix.py`` — tables, benchmark summaries,
category-wise statistics, confusion matrices, and an overall conclusions
section. Optionally embeds links to the PNG charts produced by
``plots.py`` (relative paths, so the report is portable alongside its
``plots/`` folder).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.benchmark import BenchmarkReport
from src.evaluation.metrics import ConfusionMatrix
from src.evaluation.metrics import BinaryClassificationMetrics


def _url_metrics_table(url_metrics_by_category: dict[str, BinaryClassificationMetrics]) -> str:
    """Render per-category URL Analyzer classification metrics as a markdown table.

    Evaluation Framework Update — mirrors :func:`_metrics_table`'s shape but
    adds the FPR/FNR columns this update introduced on
    ``BinaryClassificationMetrics``.
    """
    header = "| Category | Accuracy | Precision | Recall | F1 | FPR | FNR | TP | FP | TN | FN |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for category, m in sorted(url_metrics_by_category.items()):
        rows.append(
            f"| {category} | {m.accuracy:.3f} | {m.precision:.3f} | {m.recall:.3f} "
            f"| {m.f1_score:.3f} | {m.false_positive_rate:.3f} | {m.false_negative_rate:.3f} "
            f"| {m.true_positives} | {m.false_positives} | {m.true_negatives} | {m.false_negatives} |"
        )
    return "\n".join(rows)


def _metrics_table(metrics_by_category: dict[str, BinaryClassificationMetrics]) -> str:
    header = "| Category | Detection Rate | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for category, m in sorted(metrics_by_category.items()):
        rows.append(
            f"| {category} | {m.detection_rate:.3f} | {m.precision:.3f} | {m.recall:.3f} "
            f"| {m.f1_score:.3f} | {m.accuracy:.3f} | {m.true_positives} | {m.false_positives} "
            f"| {m.true_negatives} | {m.false_negatives} |"
        )
    return "\n".join(rows)


def _benchmark_table(benchmark: BenchmarkReport) -> str:
    header = "| Category | Images | Success Rate | Avg Time (ms) | Min (ms) | Max (ms) | Std Dev | Avg FPS |"
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for category, cb in sorted(benchmark.per_category.items()):
        s = cb.total_time_stats
        rows.append(
            f"| {category} | {cb.image_count} | {cb.success_rate:.1%} | {s.mean:.2f} "
            f"| {s.minimum:.2f} | {s.maximum:.2f} | {s.std_dev:.2f} | {cb.average_fps:.2f} |"
        )
    return "\n".join(rows)


def _conclusions(
    detection_metrics: BinaryClassificationMetrics,
    risk_metrics: BinaryClassificationMetrics,
    benchmark: BenchmarkReport,
) -> str:
    lines = []
    lines.append(
        f"- **QR detection** achieved an overall detection rate (recall) of "
        f"**{detection_metrics.detection_rate:.1%}**, precision of "
        f"**{detection_metrics.precision:.1%}**, and F1 score of "
        f"**{detection_metrics.f1_score:.3f}** across all categories."
    )
    lines.append(
        f"- **Risk / malicious classification** achieved recall of "
        f"**{risk_metrics.recall:.1%}** and precision of **{risk_metrics.precision:.1%}** "
        f"(F1 = **{risk_metrics.f1_score:.3f}**) against the phishing/overlay-attack ground truth."
    )
    lines.append(
        f"- The pipeline processed images at an average of "
        f"**{benchmark.overall.average_fps:.2f} FPS** per worker, with an overall "
        f"end-to-end throughput of **{benchmark.pipeline_throughput_ips:.2f} images/sec** "
        f"using {benchmark.worker_count} worker(s)."
    )
    lines.append(
        f"- Pipeline success rate was **{benchmark.overall.success_rate:.1%}** "
        f"({benchmark.overall.failure_count} failure(s) out of {benchmark.overall.image_count})."
    )
    weakest = min(
        benchmark.per_category.items(),
        key=lambda kv: kv[1].success_rate,
        default=(None, None),
    )
    if weakest[0] is not None:
        lines.append(
            f"- The **{weakest[0]}** category had the lowest success/robustness "
            f"({weakest[1].success_rate:.1%}), suggesting it as a priority for future "
            f"preprocessing or model improvements."
        )
    return "\n".join(lines)


def generate_markdown_report(
    *,
    dataset_root: str,
    detection_metrics_by_category: dict[str, BinaryClassificationMetrics],
    detection_metrics_overall: BinaryClassificationMetrics,
    risk_metrics_by_category: dict[str, BinaryClassificationMetrics],
    risk_metrics_overall: BinaryClassificationMetrics,
    benchmark: BenchmarkReport,
    detection_cm: ConfusionMatrix,
    risk_cm: ConfusionMatrix,
    plot_paths: list[Path] | None = None,
    engine_versions: dict[str, str] | None = None,
    url_metrics_by_category: dict[str, BinaryClassificationMetrics] | None = None,
    url_metrics_overall: BinaryClassificationMetrics | None = None,
    url_summary_stats: dict[str, float] | None = None,
    url_plot_paths: list[Path] | None = None,
) -> str:
    """Render the full evaluation report as a markdown string.

    Parameters
    ----------
    dataset_root : str
        Path to the evaluated dataset, for the report header.
    detection_metrics_by_category, risk_metrics_by_category : dict
        Per-category :class:`BinaryClassificationMetrics` for QR detection
        and risk/malicious classification respectively.
    detection_metrics_overall, risk_metrics_overall : BinaryClassificationMetrics
        Dataset-wide metrics for each task.
    benchmark : BenchmarkReport
        Performance/throughput benchmark data.
    detection_cm, risk_cm : ConfusionMatrix
        Overall confusion matrices for each task.
    plot_paths : list[Path], optional
        Paths to generated PNG charts; embedded as relative markdown images
        if provided.
    engine_versions : dict[str, str], optional
        e.g. {"risk_engine": "1.0.0"} for reproducibility notes.
    url_metrics_by_category, url_metrics_overall : dict | BinaryClassificationMetrics, optional
        **Evaluation Framework Update.** URL Analyzer malicious-vs-benign
        classification metrics. Leave as ``None`` (the default) when the
        URL Analyzer module is not available — doing so reproduces the
        exact report this function produced before this update, with no
        "URL Analyzer" section and unchanged section numbering.
    url_summary_stats : dict[str, float], optional
        **Evaluation Framework Update.** Expects keys ``average_risk_score``,
        ``average_analysis_time_ms``, ``success_rate``; rendered as a bullet
        list if provided.
    url_plot_paths : list[Path], optional
        **Evaluation Framework Update.** URL Analyzer chart paths, embedded
        alongside the main chart section if provided.

    Returns
    -------
    str
        Complete markdown document.
    """
    generated_at = datetime.now(tz=timezone.utc).isoformat()
    plot_paths = plot_paths or []
    url_plot_paths = url_plot_paths or []
    engine_versions = engine_versions or {}
    url_available = url_metrics_overall is not None

    section_counter = [0]

    def next_section(title: str) -> str:
        section_counter[0] += 1
        return f"## {section_counter[0]}. {title}"

    sections = [
        "# QR Shield — Evaluation Report",
        "",
        f"*Generated: {generated_at}*  ",
        f"*Dataset: `{dataset_root}`*  ",
        (
            f"*Engine version(s): "
            f"{', '.join(f'{k}={v}' for k, v in engine_versions.items()) or 'n/a'}*"
        ),
        "",
        next_section("Dataset Overview"),
        "",
        f"- Total images evaluated: **{benchmark.overall.image_count}**",
        f"- Categories: **{len(benchmark.per_category)}**",
        f"- Successful pipeline runs: **{benchmark.overall.success_count}**",
        f"- Failed pipeline runs: **{benchmark.overall.failure_count}**",
        "",
        next_section("QR Detection Metrics"),
        "",
        f"Overall — Detection Rate: **{detection_metrics_overall.detection_rate:.3f}**, "
        f"Precision: **{detection_metrics_overall.precision:.3f}**, "
        f"Recall: **{detection_metrics_overall.recall:.3f}**, "
        f"F1: **{detection_metrics_overall.f1_score:.3f}**, "
        f"Accuracy: **{detection_metrics_overall.accuracy:.3f}**",
        "",
        _metrics_table(detection_metrics_by_category),
        "",
        detection_cm.to_markdown_table(),
        "",
        next_section("Risk / Malicious-Classification Metrics"),
        "",
        f"Overall — Precision: **{risk_metrics_overall.precision:.3f}**, "
        f"Recall: **{risk_metrics_overall.recall:.3f}**, "
        f"F1: **{risk_metrics_overall.f1_score:.3f}**, "
        f"Accuracy: **{risk_metrics_overall.accuracy:.3f}**",
        "",
        _metrics_table(risk_metrics_by_category),
        "",
        risk_cm.to_markdown_table(),
        "",
        next_section("Benchmark — Processing Time & Throughput"),
        "",
        f"- Wall-clock run time: **{benchmark.wall_clock_seconds:.2f}s** "
        f"using **{benchmark.worker_count}** worker(s)",
        f"- Overall pipeline throughput: **{benchmark.pipeline_throughput_ips:.2f} images/sec**",
        f"- Overall average FPS (per-image): **{benchmark.overall.average_fps:.2f}**",
        f"- Fastest image: `{benchmark.overall.fastest_image}` "
        f"({benchmark.overall.fastest_time_ms:.2f} ms)",
        f"- Slowest image: `{benchmark.overall.slowest_image}` "
        f"({benchmark.overall.slowest_time_ms:.2f} ms)",
        "",
        _benchmark_table(benchmark),
        "",
    ]

    if url_available:
        sections.append(next_section("URL Analyzer Metrics *(Evaluation Framework Update)*"))
        sections.append("")
        sections.append(
            f"Overall — Accuracy: **{url_metrics_overall.accuracy:.3f}**, "
            f"Precision: **{url_metrics_overall.precision:.3f}**, "
            f"Recall: **{url_metrics_overall.recall:.3f}**, "
            f"F1: **{url_metrics_overall.f1_score:.3f}**, "
            f"FPR: **{url_metrics_overall.false_positive_rate:.3f}**, "
            f"FNR: **{url_metrics_overall.false_negative_rate:.3f}**"
        )
        sections.append("")
        if url_summary_stats:
            sections.append(
                f"- Average URL risk score: **{url_summary_stats.get('average_risk_score', 0.0):.3f}**"
            )
            sections.append(
                f"- Average URL analysis time: "
                f"**{url_summary_stats.get('average_analysis_time_ms', 0.0):.2f} ms**"
            )
            sections.append(
                f"- URL Analyzer success rate (of decoded QR codes): "
                f"**{url_summary_stats.get('success_rate', 0.0):.1%}**"
            )
            sections.append("")
        if url_metrics_by_category:
            sections.append(_url_metrics_table(url_metrics_by_category))
            sections.append("")

    if plot_paths or url_plot_paths:
        sections.append(next_section("Charts"))
        sections.append("")
        for p in [*plot_paths, *url_plot_paths]:
            sections.append(f"![{p.stem}](plots/{p.name})")
            sections.append("")

    sections.append(next_section("Conclusions"))
    sections.append("")
    sections.append(
        _conclusions(detection_metrics_overall, risk_metrics_overall, benchmark)
    )
    if url_available:
        sections.append(
            f"- **URL Analyzer** *(Evaluation Framework Update)* achieved recall of "
            f"**{url_metrics_overall.recall:.1%}** and a false positive rate of "
            f"**{url_metrics_overall.false_positive_rate:.1%}** against the same "
            f"phishing/overlay-attack ground truth used for risk classification."
        )
    sections.append("")

    return "\n".join(sections)


def write_report(markdown: str, path: str | Path) -> Path:
    """Write the markdown report string to *path*, creating parent dirs as needed."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown, encoding="utf-8")
    return dest