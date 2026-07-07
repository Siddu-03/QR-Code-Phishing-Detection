"""
metrics.py
==========
Week 4+ — Evaluation Framework: classification and detection metrics.

Provides pure, side-effect-free functions that turn lists of
(prediction, ground_truth) pairs into the standard binary-classification
metrics used by the rest of the evaluation suite. Used for both:

* **QR detection** — predicted = detector found a QR code; ground truth =
  category is expected to contain one (see ``utils.DEFAULT_CATEGORY_LABELS``).
* **Risk / malicious classification** — predicted = risk engine flagged the
  image as SUSPICIOUS or HIGH_RISK; ground truth = category is expected to
  be malicious/tampered.

No plotting or I/O happens here; see ``plots.py`` and ``confusion_matrix.py``
for presentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BinaryClassificationMetrics:
    """Standard binary classification metrics computed from a confusion matrix."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def total(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

    @property
    def accuracy(self) -> float:
        return _safe_div(
            self.true_positives + self.true_negatives, self.total
        )

    @property
    def precision(self) -> float:
        return _safe_div(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Recall, a.k.a. detection rate / sensitivity / true positive rate."""
        return _safe_div(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def detection_rate(self) -> float:
        """Alias of :attr:`recall` — the fraction of positives correctly found."""
        return self.recall

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return _safe_div(2 * p * r, p + r)

    @property
    def specificity(self) -> float:
        return _safe_div(self.true_negatives, self.true_negatives + self.false_positives)

    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN) — equivalently ``1 - specificity``.

        Added for URL Analyzer benchmarking (Evaluation Framework Update),
        but generally applicable to any binary task computed via
        :func:`compute_binary_metrics`.
        """
        return _safe_div(self.false_positives, self.false_positives + self.true_negatives)

    @property
    def false_negative_rate(self) -> float:
        """FNR = FN / (FN + TP) — equivalently ``1 - recall``.

        Added for URL Analyzer benchmarking (Evaluation Framework Update),
        but generally applicable to any binary task computed via
        :func:`compute_binary_metrics`.
        """
        return _safe_div(self.false_negatives, self.false_negatives + self.true_positives)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "detection_rate": round(self.detection_rate, 4),
            "f1_score": round(self.f1_score, 4),
            "specificity": round(self.specificity, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
        }


def _safe_div(numerator: float, denominator: float) -> float:
    """Return 0.0 instead of raising ZeroDivisionError / producing NaN."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_binary_metrics(
    predictions: list[bool], ground_truths: list[bool]
) -> BinaryClassificationMetrics:
    """Compute TP/FP/TN/FN and derived metrics from parallel boolean lists.

    Parameters
    ----------
    predictions : list[bool]
        Model/pipeline predicted positive (True) or negative (False) per item.
    ground_truths : list[bool]
        Ground-truth positive/negative label per item, same order and length.

    Returns
    -------
    BinaryClassificationMetrics

    Raises
    ------
    ValueError
        If the two lists have different lengths.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truths "
            f"({len(ground_truths)}) must be the same length."
        )

    tp = fp = tn = fn = 0
    for pred, actual in zip(predictions, ground_truths):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and not actual:
            tn += 1
        else:
            fn += 1

    return BinaryClassificationMetrics(
        true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn
    )


@dataclass
class NumericStats:
    """Descriptive statistics for a list of numeric measurements (e.g. timings)."""

    count: int
    mean: float
    minimum: float
    maximum: float
    std_dev: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "std_dev": round(self.std_dev, 4),
        }


def compute_numeric_stats(values: list[float]) -> NumericStats:
    """Compute count/mean/min/max/population-std-dev for *values*.

    Returns all-zero stats for an empty list rather than raising.
    """
    if not values:
        return NumericStats(count=0, mean=0.0, minimum=0.0, maximum=0.0, std_dev=0.0)

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return NumericStats(
        count=n,
        mean=mean,
        minimum=min(values),
        maximum=max(values),
        std_dev=math.sqrt(variance),
    )


# ===========================================================================
# Confusion matrices
# ===========================================================================
# Merged in from the former confusion_matrix.py: a confusion matrix is just
# a presentation view over a BinaryClassificationMetrics, so it lives next
# to the metric it presents rather than in its own file, per the
# architecture's "category metrics" responsibility for this module.

@dataclass
class ConfusionMatrix:
    """A labelled 2x2 confusion matrix for a binary classification task.

    Layout
    ------
    ::

                        Predicted Positive   Predicted Negative
        Actual Positive         TP                   FN
        Actual Negative         FP                   TN
    """

    label: str
    metrics: BinaryClassificationMetrics

    def as_rows(self) -> list[list[str]]:
        """Return the matrix as a list of string rows, suitable for a markdown table."""
        m = self.metrics
        return [
            ["", "Predicted Positive", "Predicted Negative"],
            ["Actual Positive", str(m.true_positives), str(m.false_negatives)],
            ["Actual Negative", str(m.false_positives), str(m.true_negatives)],
        ]

    def to_markdown_table(self) -> str:
        rows = self.as_rows()
        header = f"| {' | '.join(rows[0])} |"
        sep = f"| {' | '.join('---' for _ in rows[0])} |"
        body = "\n".join(f"| {' | '.join(r)} |" for r in rows[1:])
        return f"**{self.label}**\n\n{header}\n{sep}\n{body}"

    def to_dict(self) -> dict:
        return {"label": self.label, **self.metrics.to_dict()}


def build_confusion_matrix(
    label: str, predictions: list[bool], ground_truths: list[bool]
) -> ConfusionMatrix:
    """Build a :class:`ConfusionMatrix` named *label* from parallel boolean lists."""
    metrics = compute_binary_metrics(predictions, ground_truths)
    return ConfusionMatrix(label=label, metrics=metrics)