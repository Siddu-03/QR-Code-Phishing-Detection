"""
html_report.py
===============
Evaluation Framework — interactive HTML dashboard (``Evaluation_Report.html``).

Renders the same underlying data as ``generate_report.py``'s markdown
report — detection/risk/URL metrics, confusion matrices, benchmark
statistics, and system info — as a single self-contained HTML file with
embedded chart images and sortable-by-eye summary tables. No metric or
benchmark computation happens in this file: it only accepts already-computed
data structures (the same ones passed to ``generate_report.py``) and
renders them, keeping all the actual number-crunching in one place
(``metrics.py``/``benchmark.py``) rather than duplicated across two report
formats.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.benchmark import BenchmarkReport
from src.evaluation.metrics import BinaryClassificationMetrics, ConfusionMatrix
from src.evaluation.system_info import SystemInfo

_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
       background: #0f172a; color: #e2e8f0; }
h1 { font-size: 1.8rem; margin-bottom: 0.2rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #334155; padding-bottom: 0.4rem; }
.meta { color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }
.cards { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.card { background: #1e293b; border-radius: 10px; padding: 1rem 1.4rem; min-width: 160px; }
.card .label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }
.card .value { font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
th, td { border: 1px solid #334155; padding: 0.5rem 0.8rem; text-align: left; }
th { background: #1e293b; }
tr:nth-child(even) { background: #16213a; }
.charts { display: flex; flex-wrap: wrap; gap: 1.2rem; margin-top: 1rem; }
.charts figure { margin: 0; background: #1e293b; padding: 0.8rem; border-radius: 10px; }
.charts img { max-width: 420px; display: block; border-radius: 6px; }
.charts figcaption { color: #94a3b8; font-size: 0.85rem; margin-top: 0.4rem; text-align: center; }
"""


def _stat_card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'


def _metrics_table_html(title: str, metrics_by_category: dict[str, BinaryClassificationMetrics]) -> str:
    rows = ""
    for category, m in sorted(metrics_by_category.items()):
        rows += (
            f"<tr><td>{category}</td><td>{m.detection_rate:.3f}</td><td>{m.precision:.3f}</td>"
            f"<td>{m.recall:.3f}</td><td>{m.f1_score:.3f}</td><td>{m.accuracy:.3f}</td>"
            f"<td>{m.true_positives}</td><td>{m.false_positives}</td>"
            f"<td>{m.true_negatives}</td><td>{m.false_negatives}</td></tr>"
        )
    return (
        f"<h3>{title}</h3><table><thead><tr>"
        "<th>Category</th><th>Detection Rate</th><th>Precision</th><th>Recall</th><th>F1</th>"
        "<th>Accuracy</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _confusion_matrix_html(cm: ConfusionMatrix) -> str:
    m = cm.metrics
    return (
        f"<h3>Confusion Matrix — {cm.label}</h3>"
        "<table><tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>"
        f"<tr><th>Actual Positive</th><td>{m.true_positives}</td><td>{m.false_negatives}</td></tr>"
        f"<tr><th>Actual Negative</th><td>{m.false_positives}</td><td>{m.true_negatives}</td></tr></table>"
    )


def _benchmark_table_html(benchmark: BenchmarkReport) -> str:
    rows = ""
    for category, cb in sorted(benchmark.per_category.items()):
        s = cb.total_time_stats
        rows += (
            f"<tr><td>{category}</td><td>{cb.image_count}</td><td>{cb.success_rate:.1%}</td>"
            f"<td>{s.mean:.2f}</td><td>{s.minimum:.2f}</td><td>{s.maximum:.2f}</td>"
            f"<td>{s.std_dev:.2f}</td><td>{cb.average_fps:.2f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Category</th><th>Images</th><th>Success Rate</th>"
        "<th>Avg Time (ms)</th><th>Min (ms)</th><th>Max (ms)</th><th>Std Dev</th>"
        f"<th>Avg FPS</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def generate_html_report(
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
    system_info: SystemInfo | None = None,
    url_metrics_by_category: dict[str, BinaryClassificationMetrics] | None = None,
    url_metrics_overall: BinaryClassificationMetrics | None = None,
    url_summary_stats: dict[str, float] | None = None,
    url_plot_paths: list[Path] | None = None,
) -> str:
    """Render the full evaluation report as a self-contained HTML string.

    Mirrors ``generate_report.generate_markdown_report``'s parameters
    exactly, so ``evaluate_dataset.py`` computes every metric/benchmark
    object once and passes the same objects to both renderers — no
    recomputation, no duplicated aggregation logic.

    Parameters are documented in ``generate_report.generate_markdown_report``;
    behaviour is identical here, only the output format differs.

    Returns
    -------
    str
        Complete HTML document, ready to write to ``Evaluation_Report.html``.
    """
    plot_paths = plot_paths or []
    url_plot_paths = url_plot_paths or []
    generated_at = datetime.now(tz=timezone.utc).isoformat()
    url_available = url_metrics_overall is not None

    cards = "".join(
        [
            _stat_card("Images Evaluated", str(benchmark.overall.image_count)),
            _stat_card("Categories", str(len(benchmark.per_category))),
            _stat_card("Success Rate", f"{benchmark.overall.success_rate:.1%}"),
            _stat_card("Detection Rate", f"{detection_metrics_overall.detection_rate:.1%}"),
            _stat_card("Risk F1", f"{risk_metrics_overall.f1_score:.3f}"),
            _stat_card("Throughput", f"{benchmark.pipeline_throughput_ips:.2f} img/s"),
        ]
    )

    system_html = ""
    if system_info is not None:
        s = system_info
        ram = f"{s.total_ram_mb:.0f} MB" if s.total_ram_mb is not None else "unknown"
        system_html = (
            "<h2>System Information</h2>"
            "<div class='cards'>"
            + _stat_card("OS", f"{s.os_name} {s.os_release}")
            + _stat_card("CPU Cores", str(s.cpu_count or "unknown"))
            + _stat_card("RAM", ram)
            + _stat_card("Python", s.python_version)
            + _stat_card("OpenCV", s.opencv_version or "n/a")
            + "</div>"
        )

    url_html = ""
    if url_available:
        stats_html = ""
        if url_summary_stats:
            stats_html = (
                "<div class='cards'>"
                + _stat_card("Avg URL Risk", f"{url_summary_stats.get('average_risk_score', 0.0):.3f}")
                + _stat_card(
                    "Avg Analysis Time", f"{url_summary_stats.get('average_analysis_time_ms', 0.0):.2f} ms"
                )
                + _stat_card("Success Rate", f"{url_summary_stats.get('success_rate', 0.0):.1%}")
                + "</div>"
            )
        url_table = _metrics_table_html("URL Analyzer by Category", url_metrics_by_category or {})
        url_html = (
            "<h2>URL Analyzer Metrics <em>(Evaluation Framework Update)</em></h2>"
            f"{stats_html}{url_table}"
        )

    all_plots = [*plot_paths, *url_plot_paths]
    charts_html = ""
    if all_plots:
        figures = "".join(
            f"<figure><img src='charts/{p.name}' alt='{p.stem}'>"
            f"<figcaption>{p.stem.replace('_', ' ').title()}</figcaption></figure>"
            for p in all_plots
        )
        charts_html = f"<h2>Charts</h2><div class='charts'>{figures}</div>"

    weakest = min(
        benchmark.per_category.items(), key=lambda kv: kv[1].success_rate, default=(None, None)
    )
    weakest_html = ""
    if weakest[0] is not None:
        weakest_html = (
            f"<li>The <strong>{weakest[0]}</strong> category had the lowest "
            f"success/robustness ({weakest[1].success_rate:.1%}).</li>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QR Shield — Evaluation Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>QR Shield — Evaluation Report</h1>
<div class="meta">Generated: {generated_at} &middot; Dataset: <code>{dataset_root}</code></div>

<div class="cards">{cards}</div>

{system_html}

<h2>QR Detection Metrics</h2>
{_metrics_table_html("Detection by Category", detection_metrics_by_category)}
{_confusion_matrix_html(detection_cm)}

<h2>Risk / Malicious-Classification Metrics</h2>
{_metrics_table_html("Risk Classification by Category", risk_metrics_by_category)}
{_confusion_matrix_html(risk_cm)}

<h2>Benchmark — Processing Time &amp; Throughput</h2>
<div class="cards">
{_stat_card("Wall Clock", f"{benchmark.wall_clock_seconds:.2f}s")}
{_stat_card("Workers", str(benchmark.worker_count))}
{_stat_card("Avg FPS", f"{benchmark.overall.average_fps:.2f}")}
</div>
{_benchmark_table_html(benchmark)}

{url_html}

{charts_html}

<h2>Conclusions</h2>
<ul>
<li>Overall detection F1: <strong>{detection_metrics_overall.f1_score:.3f}</strong>,
    risk classification F1: <strong>{risk_metrics_overall.f1_score:.3f}</strong>.</li>
{weakest_html}
</ul>

</body>
</html>
"""


def write_html_report(html: str, path: str | Path) -> Path:
    """Write the HTML report string to *path*, creating parent dirs as needed."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    return dest