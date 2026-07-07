"""
plots.py
========
Week 4+ — Evaluation Framework: research-report chart generation.

Renders the graph set requested for the QR Shield evaluation suite using
matplotlib's non-interactive ``Agg`` backend (safe for headless/CI use).
Every function takes already-computed data structures (from ``metrics.py``,
``benchmark.py``, ``confusion_matrix.py``) and a destination directory, and
returns the list of file paths it wrote — it never computes metrics itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.benchmark import BenchmarkReport
from src.evaluation.config import CHART_DPI, CHART_EXPORT_FORMATS
from src.evaluation.metrics import ConfusionMatrix

logger = logging.getLogger("evaluation.plots")


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Save *fig* under *name* in every format listed in ``config.CHART_EXPORT_FORMATS``.

    Returns the path to the PNG (used for markdown/HTML embedding); other
    configured formats (``svg``, ``pdf``) are written alongside it silently.
    A run with the default config (PNG only) behaves exactly as before this
    became configurable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    for fmt in CHART_EXPORT_FORMATS:
        dest = out_dir / f"{name}.{fmt}"
        fig.savefig(dest, bbox_inches="tight", dpi=CHART_DPI)
        logger.info("Saved plot: %s", dest)
    plt.close(fig)
    return png_path


def plot_detection_rate_per_category(
    benchmark: BenchmarkReport,
    detection_rate_by_category: dict[str, float],
    out_dir: Path,
) -> Path:
    """Bar chart of QR detection rate per category."""
    categories = list(detection_rate_by_category.keys())
    rates = [detection_rate_by_category[c] * 100 for c in categories]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(categories, rates, color="#3b82f6")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("QR Detection Rate per Category")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=35)
    for i, r in enumerate(rates):
        ax.text(i, r + 1, f"{r:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "detection_rate_per_category")


def plot_average_processing_time(benchmark: BenchmarkReport, out_dir: Path) -> Path:
    """Bar chart of average total processing time (ms) per category."""
    categories = list(benchmark.per_category.keys())
    means = [benchmark.per_category[c].total_time_stats.mean for c in categories]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(categories, means, color="#f97316")
    ax.set_ylabel("Average Processing Time (ms)")
    ax.set_title("Average Processing Time per Category")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return _save(fig, out_dir, "average_processing_time")


def plot_confusion_matrix(cm: ConfusionMatrix, out_dir: Path) -> Path:
    """Heatmap rendering of a 2x2 confusion matrix."""
    import numpy as np

    m = cm.metrics
    matrix = np.array([[m.true_positives, m.false_negatives], [m.false_positives, m.true_negatives]])

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticklabels(["Actual Positive", "Actual Negative"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title(f"Confusion Matrix — {cm.label}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, out_dir, f"confusion_matrix_{cm.label.lower().replace(' ', '_')}")


def plot_risk_level_distribution(risk_level_counts: dict[str, int], out_dir: Path) -> Path:
    """Pie chart of predicted risk-level distribution (SAFE/SUSPICIOUS/HIGH_RISK)."""
    labels = list(risk_level_counts.keys())
    values = list(risk_level_counts.values())
    colors = {"SAFE": "#22c55e", "SUSPICIOUS": "#eab308", "HIGH_RISK": "#ef4444"}

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=[colors.get(lbl, "#94a3b8") for lbl in labels],
    )
    ax.set_title("Risk Level Distribution")
    fig.tight_layout()
    return _save(fig, out_dir, "risk_level_distribution")


def plot_confidence_distribution(confidences: list[float], out_dir: Path) -> Path:
    """Histogram of risk-assessment confidence scores."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(confidences, bins=20, color="#8b5cf6", edgecolor="white")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Image Count")
    ax.set_title("Confidence Distribution")
    fig.tight_layout()
    return _save(fig, out_dir, "confidence_distribution")


def plot_processing_time_histogram(total_times_ms: list[float], out_dir: Path) -> Path:
    """Histogram of total per-image processing time (ms)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(total_times_ms, bins=30, color="#0ea5e9", edgecolor="white")
    ax.set_xlabel("Total Processing Time (ms)")
    ax.set_ylabel("Image Count")
    ax.set_title("Processing Time Histogram")
    fig.tight_layout()
    return _save(fig, out_dir, "processing_time_histogram")


def plot_detection_success_pie(success_count: int, failure_count: int, out_dir: Path) -> Path:
    """Pie chart of overall pipeline success vs failure."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [success_count, failure_count],
        labels=["Success", "Failure"],
        autopct="%1.1f%%",
        colors=["#22c55e", "#ef4444"],
    )
    ax.set_title("Detection Success Rate")
    fig.tight_layout()
    return _save(fig, out_dir, "detection_success_pie")


def plot_false_positive_vs_negative(fp_by_category: dict[str, int], fn_by_category: dict[str, int], out_dir: Path) -> Path:
    """Grouped bar chart comparing false positives vs false negatives per category."""
    import numpy as np

    categories = sorted(set(fp_by_category) | set(fn_by_category))
    fp_values = [fp_by_category.get(c, 0) for c in categories]
    fn_values = [fn_by_category.get(c, 0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, fp_values, width, label="False Positives", color="#f43f5e")
    ax.bar(x + width / 2, fn_values, width, label="False Negatives", color="#6366f1")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("False Positives vs False Negatives per Category")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "false_positive_vs_negative")


def plot_url_risk_distribution(risk_scores: list[float], out_dir: Path) -> Path:
    """Histogram of normalised (0-1) URL Analyzer risk scores.

    Evaluation Framework Update — only called when URL Analyzer results
    are available for at least one image; see ``evaluate_dataset.py``.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(risk_scores, bins=20, color="#dc2626", edgecolor="white")
    ax.set_xlabel("URL Risk Score (0-1)")
    ax.set_ylabel("Image Count")
    ax.set_title("URL Risk Distribution")
    fig.tight_layout()
    return _save(fig, out_dir, "url_risk_distribution")


def plot_https_vs_http(https_count: int, http_count: int, out_dir: Path) -> Path:
    """Pie chart of HTTPS vs HTTP decoded URLs.

    Evaluation Framework Update.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [https_count, http_count],
        labels=["HTTPS", "HTTP"],
        autopct="%1.1f%%",
        colors=["#22c55e", "#f97316"],
    )
    ax.set_title("HTTPS vs HTTP")
    fig.tight_layout()
    return _save(fig, out_dir, "https_vs_http")


def plot_shortener_distribution(shortener_count: int, non_shortener_count: int, out_dir: Path) -> Path:
    """Pie chart of URL-shortener vs direct-link prevalence.

    Evaluation Framework Update.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [shortener_count, non_shortener_count],
        labels=["Shortener", "Direct Link"],
        autopct="%1.1f%%",
        colors=["#eab308", "#3b82f6"],
    )
    ax.set_title("URL Shortener Distribution")
    fig.tight_layout()
    return _save(fig, out_dir, "shortener_distribution")


def plot_suspicious_keyword_frequency(keyword_counts: dict[str, int], out_dir: Path) -> Path:
    """Horizontal bar chart of suspicious-keyword hit frequency.

    Evaluation Framework Update.
    """
    items = sorted(keyword_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    labels = [k for k, _ in items]
    counts = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels))))
    ax.barh(labels, counts, color="#a855f7")
    ax.set_xlabel("Occurrences")
    ax.set_title("Suspicious Keyword Frequency")
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, out_dir, "suspicious_keyword_frequency")


def plot_suspicious_tld_frequency(tld_counts: dict[str, int], out_dir: Path) -> Path:
    """Bar chart of suspicious top-level-domain frequency.

    Evaluation Framework Update.
    """
    items = sorted(tld_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
    labels = [k for k, _ in items]
    counts = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, counts, color="#0891b2")
    ax.set_ylabel("Occurrences")
    ax.set_title("Suspicious TLD Frequency")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return _save(fig, out_dir, "suspicious_tld_frequency")


def plot_visual_risk_vs_url_risk(visual_risk_scores: list[float], url_risk_scores: list[float], out_dir: Path) -> Path:
    """Scatter plot comparing pipeline (visual/tamper) risk score vs URL risk score.

    Evaluation Framework Update. Both axes expected on a comparable 0-100
    scale (the existing ``RiskResult.score`` scale); URL scores below 1.0
    are assumed already normalised 0-1 and are rescaled to 0-100 for the
    comparison, matching ``RiskResult.score``'s convention.
    """
    import numpy as np

    url_scores_rescaled = [s * 100.0 if s <= 1.0 else s for s in url_risk_scores]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(visual_risk_scores, url_scores_rescaled, alpha=0.6, color="#6366f1", edgecolor="white")
    lims = [0, 100]
    ax.plot(lims, lims, linestyle="--", color="#94a3b8", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Visual/Tamper Risk Score (RiskEngine)")
    ax.set_ylabel("URL Risk Score")
    ax.set_title("Combined Visual Risk vs URL Risk")
    fig.tight_layout()
    return _save(fig, out_dir, "visual_risk_vs_url_risk")


def generate_url_analyzer_plots(
    risk_scores: list[float],
    https_count: int,
    http_count: int,
    shortener_count: int,
    non_shortener_count: int,
    keyword_counts: dict[str, int],
    tld_counts: dict[str, int],
    visual_risk_scores: list[float],
    url_risk_scores_for_scatter: list[float],
    out_dir: str | Path,
) -> list[Path]:
    """Generate the URL Analyzer chart set (Evaluation Framework Update).

    Called only when URL Analyzer results are available for at least one
    image in the run — see ``evaluate_dataset.py``. Each chart failing
    (e.g. empty data) is logged and skipped rather than aborting the batch,
    matching :func:`generate_all_plots`'s behaviour.
    """
    out_dir = Path(out_dir)
    paths: list[Path] = []
    jobs = [
        lambda: plot_url_risk_distribution(risk_scores, out_dir),
        lambda: plot_https_vs_http(https_count, http_count, out_dir),
        lambda: plot_shortener_distribution(shortener_count, non_shortener_count, out_dir),
        lambda: plot_suspicious_keyword_frequency(keyword_counts, out_dir),
        lambda: plot_suspicious_tld_frequency(tld_counts, out_dir),
        lambda: plot_visual_risk_vs_url_risk(visual_risk_scores, url_risk_scores_for_scatter, out_dir),
    ]
    for job in jobs:
        try:
            paths.append(job())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping a URL-analyzer plot due to error: %s", exc)
    return paths


def generate_all_plots(
    benchmark: BenchmarkReport,
    detection_rate_by_category: dict[str, float],
    detection_cm: ConfusionMatrix,
    risk_cm: ConfusionMatrix,
    risk_level_counts: dict[str, int],
    confidences: list[float],
    total_times_ms: list[float],
    fp_by_category: dict[str, int],
    fn_by_category: dict[str, int],
    out_dir: str | Path,
) -> list[Path]:
    """Generate the full requested chart set in one call.

    Any single chart failing (e.g. empty data) is logged and skipped rather
    than aborting the rest of the batch.
    """
    out_dir = Path(out_dir)
    paths: list[Path] = []
    jobs = [
        lambda: plot_detection_rate_per_category(benchmark, detection_rate_by_category, out_dir),
        lambda: plot_average_processing_time(benchmark, out_dir),
        lambda: plot_confusion_matrix(detection_cm, out_dir),
        lambda: plot_confusion_matrix(risk_cm, out_dir),
        lambda: plot_risk_level_distribution(risk_level_counts, out_dir),
        lambda: plot_confidence_distribution(confidences, out_dir),
        lambda: plot_processing_time_histogram(total_times_ms, out_dir),
        lambda: plot_detection_success_pie(
            benchmark.overall.success_count, benchmark.overall.failure_count, out_dir
        ),
        lambda: plot_false_positive_vs_negative(fp_by_category, fn_by_category, out_dir),
    ]
    for job in jobs:
        try:
            paths.append(job())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping a plot due to error: %s", exc)
    return paths