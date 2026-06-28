"""
scoring.py
==========
Week 3 – Member 2: Risk Assessment Engine
Project: Computer Vision-Based Graphic Tamper Detection for QR Code Phishing Prevention

Overview
--------
The Scoring Engine converts raw, heterogeneous detection signals — QR
detector confidence today; overlay detection, edge inconsistencies, contour
mismatches, finder-pattern damage, anomaly counts, and tamper-model
confidence in the near future — into a single, bounded numerical risk
score in the closed interval ``[0, 100]``.

This module performs **scoring only**.  It does not classify risk levels,
does not produce recommendations, and does not make decisions.  Those
responsibilities belong to ``rule_engine.py`` (decision logic) and the
not-yet-implemented ``risk_engine.py`` (pipeline orchestration), which
consumes both this module and ``rule_engine.py``.

Design philosophy
------------------
* **Configurable** — every weight lives in :class:`ScoringConfig`. There
  are no hardcoded numeric weights inside scoring logic.
* **Registry-based / extensible** — each scoring factor is a small,
  independent :class:`ScoreFactor` implementation registered into a
  :class:`FactorRegistry`. Adding a new signal (e.g. a future ML tamper
  classifier) means writing one new ``ScoreFactor`` subclass and
  registering it — no existing code is touched.
* **Explainable** — :meth:`ScoringEngine.compute_score` returns a
  :class:`ScoreBreakdown` that records every factor's raw input, its
  normalised contribution, the weight applied, and a plain-language
  explanation suitable for audit logs and research evaluation.
* **Independent** — this module has *zero* imports from ``rule_engine``,
  ``risk_engine``, or ``tamper_detector``. It depends only on the standard
  library and operates on plain Python primitives (floats, bools, ints)
  so that callers can populate a :class:`ScoringInputs` instance directly
  from ``qr_detector.py`` output, a future ``tamper_detector.py`` output,
  or hand-constructed test fixtures.
* **Future-proof** — :class:`ScoringInputs` carries a ``metadata`` /
  ``extra_signals`` extension slot mirroring the pattern already used in
  ``RiskResult.metadata``, so new upstream signals can flow through before
  a dedicated ``ScoreFactor`` is written for them.

Pipeline position
------------------
::

    qr_detector.py  ──┐
    tamper_detector.py├──►  scoring.py  ──►  risk_engine.py  ──►
                       ┘    rule_engine.py  ──►  RiskResult
                            (future)

``scoring.py`` produces a numeric score on ``[0, 100]``.  ``rule_engine.py``
already expects exactly this contract — see ``RuleEngine.evaluate(score, ...)``
in ``rule_engine.py``, which validates ``score`` against ``[0, 100]``.
``risk_engine.py`` (not yet implemented) is expected to sit between the two,
calling :func:`compute_score` and feeding ``ScoreBreakdown.total_score``
directly into ``RuleEngine.evaluate``.

Usage
-----
::

    from src.risk_assessment.scoring import (
        ScoringConfig, ScoringEngine, ScoringInputs,
    )

    engine = ScoringEngine(config=ScoringConfig())

    inputs = ScoringInputs(
        detection_confidence=0.92,
        overlay_detected=True,
        overlay_confidence=0.81,
        edge_inconsistency_score=0.40,
        contour_mismatch_score=0.15,
        finder_pattern_damage_score=0.05,
        anomaly_count=2,
        tamper_confidence=0.63,
    )

    breakdown = engine.compute_score(inputs)

    print(breakdown.total_score)        # 0–100 float
    print(breakdown.explanation)        # human-readable narrative
    for fs in breakdown.factor_scores:
        print(fs.factor_name, fs.contribution, fs.weight)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
SCORE_MIN: float = 0.0
SCORE_MAX: float = 100.0


# ===========================================================================
# Scoring Inputs
# ===========================================================================

@dataclass
class ScoringInputs:
    """Normalised container for every signal the Scoring Engine can consume.

    All confidence / intensity fields are expected on ``[0.0, 1.0]`` (a
    detector or model's own confidence scale); the engine performs the
    mapping onto the ``[0, 100]`` score space, not the caller.  Fields that
    a particular pipeline run cannot populate yet (e.g. ``tamper_confidence``
    before the tamper-analysis module exists) are left at their default and
    are simply skipped by factors that require them — see
    :meth:`ScoreFactor.is_applicable`.

    Attributes
    ----------
    detection_confidence : float | None
        Confidence reported by ``qr_detector.py`` for the QR detection
        itself, on ``[0.0, 1.0]``. ``None`` if unavailable (e.g. the
        current OpenCV / pyzbar backends do not report confidence —
        see ``qr_detector.py``'s ``"confidence": None`` field). Low
        detection confidence does not by itself imply tampering, but it
        reduces certainty in every downstream signal, so it is modelled
        as a *mild* risk contributor and a confidence dampener.
    overlay_detected : bool
        Whether a foreign graphic overlay (e.g. a sticker placed on top of
        a legitimate QR code) was detected. Reserved for the tamper
        analysis module.
    overlay_confidence : float
        Confidence in ``overlay_detected``, on ``[0.0, 1.0]``. Ignored
        when ``overlay_detected`` is ``False``. Default ``0.0``.
    edge_inconsistency_score : float
        Normalised measure of edge irregularity around the QR code
        boundary, on ``[0.0, 1.0]``, where ``0.0`` is a clean, uniform
        edge and ``1.0`` is maximally inconsistent. Reserved for the
        tamper analysis module. Default ``0.0``.
    contour_mismatch_score : float
        Normalised measure of how much the detected contour deviates from
        an expected QR code contour shape, on ``[0.0, 1.0]``. Reserved for
        the tamper analysis module. Default ``0.0``.
    finder_pattern_damage_score : float
        Normalised measure of damage / occlusion to the QR code's finder
        patterns (the three corner squares), on ``[0.0, 1.0]``. Reserved
        for the tamper analysis module. Default ``0.0``.
    anomaly_count : int
        Total count of discrete visual anomalies detected across all
        upstream analysis stages. Used as a corroborating signal: a single
        weak indicator is less concerning than several independent weak
        indicators co-occurring. Default ``0``.
    tamper_confidence : float | None
        Aggregate confidence from a future tamper-analysis model that the
        QR code graphic has been tampered with, on ``[0.0, 1.0]``.
        ``None`` if no tamper model has run yet. Default ``None``.
    ai_confidence : float | None
        Reserved slot for a future general-purpose AI / ML confidence
        score (e.g. a learned classifier distinct from
        ``tamper_confidence``), on ``[0.0, 1.0]``. ``None`` if unavailable.
        Default ``None``.
    extra_signals : dict[str, float]
        Open extension slot for signals that do not yet have a dedicated
        field or :class:`ScoreFactor`. Keys are signal names; values are
        normalised to ``[0.0, 1.0]`` by the caller. A custom
        :class:`ScoreFactor` can read from this dict via
        ``inputs.extra_signals.get("my_signal")``. Unrecognised keys are
        inert (ignored) unless a factor is registered to read them — this
        is the primary extension point for ML confidence values that have
        not yet earned a first-class field. Default ``{}``.
    metadata : dict[str, Any]
        Open annotation slot (image id, model version, etc.) carried
        through for audit / research logging. Not consumed by any factor.
        Default ``{}``.

    Raises
    ------
    ValueError
        If any ``[0.0, 1.0]``-bounded field is outside that range, or if
        ``anomaly_count`` is negative.
    """

    detection_confidence:         float | None = None
    overlay_detected:             bool = False
    overlay_confidence:           float = 0.0
    edge_inconsistency_score:     float = 0.0
    contour_mismatch_score:       float = 0.0
    finder_pattern_damage_score:  float = 0.0
    anomaly_count:                int = 0
    tamper_confidence:            float | None = None
    ai_confidence:                float | None = None
    extra_signals:                dict[str, float] = field(default_factory=dict)
    metadata:                     dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that every bounded field falls within its declared range."""
        unit_fields: dict[str, float | None] = {
            "detection_confidence":        self.detection_confidence,
            "overlay_confidence":          self.overlay_confidence,
            "edge_inconsistency_score":    self.edge_inconsistency_score,
            "contour_mismatch_score":      self.contour_mismatch_score,
            "finder_pattern_damage_score": self.finder_pattern_damage_score,
            "tamper_confidence":           self.tamper_confidence,
            "ai_confidence":               self.ai_confidence,
        }
        for name, value in unit_fields.items():
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be a float or None, got {type(value).__name__!r}."
                )
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {value!r}."
                )

        if not isinstance(self.anomaly_count, int) or self.anomaly_count < 0:
            raise ValueError(
                f"anomaly_count must be a non-negative int, "
                f"got {self.anomaly_count!r}."
            )

        for key, value in self.extra_signals.items():
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"extra_signals[{key!r}] must be a float in [0.0, 1.0], "
                    f"got {value!r}."
                )


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class ScoringConfig:
    """Weight configuration for every registered :class:`ScoreFactor`.

    All weights are non-negative floats representing each factor's maximum
    possible contribution to the final ``[0, 100]`` score, *before*
    normalisation. Internally, :class:`ScoringEngine` normalises the sum of
    *applicable* weights (factors whose required inputs are present) back
    onto ``[0, 100]`` — see :meth:`ScoringEngine.compute_score` — so the
    relative ratios between weights matter more than their absolute values.

    Parameters
    ----------
    detection_confidence_weight : float
        Maximum contribution from low QR-detection confidence.
        Default ``5.0``.
    overlay_weight : float
        Maximum contribution from a detected graphic overlay.
        Default ``25.0``.
    edge_inconsistency_weight : float
        Maximum contribution from edge-inconsistency analysis.
        Default ``15.0``.
    contour_mismatch_weight : float
        Maximum contribution from contour-mismatch analysis.
        Default ``15.0``.
    finder_pattern_damage_weight : float
        Maximum contribution from finder-pattern damage analysis.
        Default ``20.0``.
    anomaly_count_weight : float
        Maximum contribution from the corroborating anomaly count.
        Default ``10.0``.
    anomaly_count_saturation : int
        Anomaly count at which :class:`AnomalyCountFactor` reaches its
        maximum contribution. Counts at or above this value saturate to
        ``1.0`` normalised contribution. Default ``5``.
    tamper_confidence_weight : float
        Maximum contribution from a future tamper-analysis model's
        confidence. Default ``35.0``.
    ai_confidence_weight : float
        Maximum contribution from a future general-purpose AI confidence
        score. Default ``20.0``.
    extra_signal_weights : dict[str, float]
        Maximum contribution per named key in
        ``ScoringInputs.extra_signals``. A key with no entry here
        contributes nothing even if present in ``extra_signals`` — this
        keeps unknown signals inert by default until a researcher
        explicitly assigns them weight. Default ``{}``.

    Raises
    ------
    ValueError
        If any weight is negative, or if ``anomaly_count_saturation`` is
        not a positive integer.

    Notes
    -----
    Weights need not sum to ``100``. :class:`ScoringEngine` always
    re-normalises onto ``[0, 100]`` using the sum of weights for factors
    that are actually applicable to a given :class:`ScoringInputs`
    instance, so partial pipelines (e.g. QR detection only, before tamper
    analysis exists) still produce a meaningful score on the full scale
    rather than one silently capped by the missing factors' unused weight.
    """

    detection_confidence_weight:  float = 5.0
    overlay_weight:                float = 25.0
    edge_inconsistency_weight:     float = 15.0
    contour_mismatch_weight:       float = 15.0
    finder_pattern_damage_weight:  float = 20.0
    anomaly_count_weight:          float = 10.0
    anomaly_count_saturation:      int   = 5
    tamper_confidence_weight:      float = 35.0
    ai_confidence_weight:          float = 20.0
    extra_signal_weights:          dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weight_fields: dict[str, float] = {
            "detection_confidence_weight": self.detection_confidence_weight,
            "overlay_weight":              self.overlay_weight,
            "edge_inconsistency_weight":   self.edge_inconsistency_weight,
            "contour_mismatch_weight":     self.contour_mismatch_weight,
            "finder_pattern_damage_weight": self.finder_pattern_damage_weight,
            "anomaly_count_weight":        self.anomaly_count_weight,
            "tamper_confidence_weight":    self.tamper_confidence_weight,
            "ai_confidence_weight":        self.ai_confidence_weight,
        }
        for name, value in weight_fields.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value!r}.")

        if self.anomaly_count_saturation <= 0:
            raise ValueError(
                f"anomaly_count_saturation must be a positive int, "
                f"got {self.anomaly_count_saturation!r}."
            )

        for key, value in self.extra_signal_weights.items():
            if value < 0.0:
                raise ValueError(
                    f"extra_signal_weights[{key!r}] must be non-negative, "
                    f"got {value!r}."
                )

    @property
    def as_dict(self) -> dict[str, Any]:
        """Serialise the configuration for inclusion in audit trails."""
        return {
            "detection_confidence_weight":  self.detection_confidence_weight,
            "overlay_weight":                self.overlay_weight,
            "edge_inconsistency_weight":     self.edge_inconsistency_weight,
            "contour_mismatch_weight":       self.contour_mismatch_weight,
            "finder_pattern_damage_weight":  self.finder_pattern_damage_weight,
            "anomaly_count_weight":          self.anomaly_count_weight,
            "anomaly_count_saturation":      self.anomaly_count_saturation,
            "tamper_confidence_weight":      self.tamper_confidence_weight,
            "ai_confidence_weight":          self.ai_confidence_weight,
            "extra_signal_weights":          dict(self.extra_signal_weights),
        }


# ===========================================================================
# Factor Score (per-factor output record)
# ===========================================================================

@dataclass(frozen=True)
class FactorScore:
    """Result of evaluating a single :class:`ScoreFactor`.

    Attributes
    ----------
    factor_name : str
        Unique identifier of the factor, matching :attr:`ScoreFactor.name`.
    raw_value : float | bool | int | None
        The raw, un-normalised input value the factor read from
        :class:`ScoringInputs` (e.g. ``0.81`` confidence, ``True``
        overlay flag, ``3`` anomaly count). Kept for debugging and audit
        trails; ``None`` if the factor was inapplicable.
    normalised_value : float
        The factor's raw value mapped onto ``[0.0, 1.0]``, where ``1.0``
        represents maximum risk contribution from this factor alone.
        ``0.0`` if the factor was inapplicable.
    weight : float
        The configured maximum-contribution weight for this factor, as
        read from :class:`ScoringConfig` at evaluation time.
    contribution : float
        ``normalised_value * weight`` — the factor's contribution to the
        score *before* the final re-normalisation step described in
        :meth:`ScoringEngine.compute_score`. Always ``0.0`` for
        inapplicable factors.
    applicable : bool
        Whether the required input data was present for this factor to
        run. Inapplicable factors contribute ``0.0`` and are excluded
        from the re-normalisation denominator (they do not silently
        deflate the score just because upstream data is missing).
    explanation : str
        One-sentence, human-readable description of this factor's
        evaluation, suitable for direct inclusion in audit logs and
        research reports.
    """

    factor_name:       str
    raw_value:          float | bool | int | None
    normalised_value:   float
    weight:             float
    contribution:        float
    applicable:          bool
    explanation:         str

    def to_dict(self) -> dict[str, Any]:
        """Serialise this factor score to a JSON-compatible dictionary."""
        return {
            "factor_name":       self.factor_name,
            "raw_value":          self.raw_value,
            "normalised_value":   round(self.normalised_value, 6),
            "weight":             self.weight,
            "contribution":        round(self.contribution, 6),
            "applicable":          self.applicable,
            "explanation":         self.explanation,
        }


# ===========================================================================
# Score Breakdown (top-level output record)
# ===========================================================================

@dataclass(frozen=True)
class ScoreBreakdown:
    """Complete, explainable output of :meth:`ScoringEngine.compute_score`.

    This is the sole public output contract of ``scoring.py``. It is
    intentionally free of any classification or decision fields — those
    belong to ``rule_engine.RuleEngineResult``, produced downstream from
    ``total_score``.

    Attributes
    ----------
    total_score : float
        The final composite risk score, clamped to ``[0, 100]``. This is
        the value intended to be passed as the ``score`` argument to
        ``RuleEngine.evaluate(score=...)``.
    factor_scores : tuple[FactorScore, ...]
        Per-factor breakdown for every registered factor, including
        inapplicable ones (with ``applicable=False``), so that callers
        can see the *complete* registry state at evaluation time.
    applicable_weight_total : float
        Sum of weights for factors that were applicable. Used as the
        re-normalisation denominator; exposed for transparency and unit
        testing.
    explanation : str
        Multi-sentence narrative summarising which factors drove the
        score, suitable for audit logs and research publications.
    config_snapshot : dict[str, Any]
        Serialised :class:`ScoringConfig` active at evaluation time, for
        full reproducibility.

    Notes
    -----
    ``total_score`` deliberately does **not** map to ``RiskResult.score``
    directly — ``RiskResult.score`` is constrained to ``[0.0, 1.0]`` while
    this engine's contract is ``[0, 100]`` (matching ``rule_engine.py``'s
    expected input range). A future ``risk_engine.py`` is responsible for
    dividing by 100 if and when it constructs a ``RiskResult`` directly;
    when going through ``RuleEngine.evaluate``, no conversion is needed.
    """

    total_score:                float
    factor_scores:              tuple[FactorScore, ...]
    applicable_weight_total:    float
    explanation:                 str
    config_snapshot:             dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialise this breakdown to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Safe to pass to ``json.dumps`` or return as a FastAPI response
            body without a custom encoder.
        """
        return {
            "total_score":             round(self.total_score, 6),
            "factor_scores":           [fs.to_dict() for fs in self.factor_scores],
            "applicable_weight_total": round(self.applicable_weight_total, 6),
            "explanation":              self.explanation,
            "config_snapshot":          dict(self.config_snapshot),
        }

    @property
    def applicable_factor_scores(self) -> tuple[FactorScore, ...]:
        """Return only the factors that were applicable (had input data)."""
        return tuple(fs for fs in self.factor_scores if fs.applicable)


# ===========================================================================
# Abstract Factor Interface
# ===========================================================================

class ScoreFactor(ABC):
    """Abstract base class for a single, independent scoring signal.

    Each concrete factor encapsulates exactly one piece of evidence (e.g.
    "is there a graphic overlay?") and knows how to: (1) decide whether it
    has enough data to run (:meth:`is_applicable`), (2) extract and
    normalise its raw input onto ``[0.0, 1.0]`` (:meth:`normalise`), and
    (3) explain itself in plain language (:meth:`explain`). The
    :class:`ScoringEngine` handles weighting, aggregation, and
    re-normalisation centrally — factors never see weights or other
    factors' state, which keeps each factor trivially unit-testable in
    isolation.

    Subclass contract
    ------------------
    Implement :attr:`name`, :meth:`is_applicable`, :meth:`raw_value`, and
    :meth:`normalise`. Optionally override :meth:`explain` for a more
    specific narrative; the default implementation produces a generic but
    serviceable sentence from ``raw_value`` and ``normalise``.

    Do not raise from any method except for genuinely exceptional,
    programmer-error conditions — a factor that cannot apply should return
    ``False`` from :meth:`is_applicable`, not raise.
    """

    #: Short, unique, snake_case identifier used in :class:`FactorScore`
    #: and logs. Must be overridden by every concrete subclass.
    name: ClassVar[str] = "base_factor"

    @abstractmethod
    def is_applicable(self, inputs: ScoringInputs) -> bool:
        """Return whether this factor has sufficient data to evaluate.

        Parameters
        ----------
        inputs : ScoringInputs
            The full set of inputs for the current scoring run.

        Returns
        -------
        bool
            ``True`` if :meth:`raw_value` and :meth:`normalise` can be
            safely called; ``False`` if required upstream data (e.g. a
            tamper-analysis confidence that does not exist yet) is
            absent.
        """

    @abstractmethod
    def raw_value(self, inputs: ScoringInputs) -> float | bool | int | None:
        """Extract this factor's raw, un-normalised value from *inputs*.

        Only called when :meth:`is_applicable` returns ``True``.

        Parameters
        ----------
        inputs : ScoringInputs
            The full set of inputs for the current scoring run.

        Returns
        -------
        float | bool | int | None
            The raw value as read from *inputs*, kept for audit purposes.
        """

    @abstractmethod
    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        """Map this factor's raw value onto ``[0.0, 1.0]`` risk contribution.

        Only called when :meth:`is_applicable` returns ``True``.

        Parameters
        ----------
        inputs : ScoringInputs
            The full set of inputs for the current scoring run.
        config : ScoringConfig
            Active configuration, in case normalisation depends on a
            configured parameter (e.g. saturation point).

        Returns
        -------
        float
            A value in ``[0.0, 1.0]`` where ``1.0`` represents this
            factor's maximum possible risk contribution.
        """

    def weight(self, config: ScoringConfig) -> float:
        """Return this factor's configured maximum-contribution weight.

        Parameters
        ----------
        config : ScoringConfig
            Active configuration.

        Returns
        -------
        float
            Non-negative weight value. Default implementation raises
            :class:`NotImplementedError`; concrete factors must override
            this to read their corresponding ``ScoringConfig`` field.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement weight()."
        )

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        """Produce a one-sentence, human-readable explanation.

        Default implementation; subclasses are encouraged to override
        with a more specific, signal-appropriate sentence.

        Parameters
        ----------
        inputs : ScoringInputs
            The full set of inputs for the current scoring run.
        raw : float | bool | int | None
            This factor's raw value, as returned by :meth:`raw_value`.
        normalised : float
            This factor's normalised contribution, as returned by
            :meth:`normalise`.

        Returns
        -------
        str
            A complete, capitalised sentence.
        """
        return (
            f"{self.name}: raw={raw!r}, normalised risk contribution="
            f"{normalised:.2f}."
        )


# ===========================================================================
# Concrete Factor Implementations
# ===========================================================================

class DetectionConfidenceFactor(ScoreFactor):
    """Mild risk contribution from low QR-detection confidence.

    Rationale
    ---------
    Detection confidence is not itself evidence of tampering — a clean,
    unmodified QR code can still be detected with low confidence due to
    poor lighting or motion blur. However, low confidence reduces the
    overall reliability of *every* downstream measurement (a blurry or
    partially-occluded QR code makes edge, contour, and finder-pattern
    analysis less trustworthy too). This factor therefore contributes a
    small, capped amount of risk proportional to ``1 - confidence``,
    reflecting increased *uncertainty* rather than confirmed tampering.
    """

    name: ClassVar[str] = "detection_confidence"

    def is_applicable(self, inputs: ScoringInputs) -> bool:
        return inputs.detection_confidence is not None

    def raw_value(self, inputs: ScoringInputs) -> float | None:
        return inputs.detection_confidence

    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        confidence = inputs.detection_confidence
        assert confidence is not None  # guaranteed by is_applicable
        return max(0.0, min(1.0, 1.0 - float(confidence)))

    def weight(self, config: ScoringConfig) -> float:
        return config.detection_confidence_weight

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        return (
            f"QR detection confidence was {raw:.2f}, contributing a mild "
            f"uncertainty-based risk factor of {normalised:.2f}."
        )


class OverlayFactor(ScoreFactor):
    """Risk contribution from a detected foreign graphic overlay.

    Rationale
    ---------
    A graphic overlay (e.g. a sticker placed on top of a legitimate QR
    code redirecting scanners to a malicious URL) is one of the most
    direct physical phishing vectors this project targets. When detected,
    the factor's contribution scales with the overlay detector's own
    confidence rather than firing at a fixed value, so a tentative overlay
    detection contributes proportionally less than a confident one.
    """

    name: ClassVar[str] = "overlay_detected"

    def is_applicable(self, inputs: ScoringInputs) -> bool:
        # Always applicable: absence of an overlay is itself meaningful
        # information (zero contribution), not missing data.
        return True

    def raw_value(self, inputs: ScoringInputs) -> bool:
        return inputs.overlay_detected

    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        if not inputs.overlay_detected:
            return 0.0
        return max(0.0, min(1.0, float(inputs.overlay_confidence)))

    def weight(self, config: ScoringConfig) -> float:
        return config.overlay_weight

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        if not raw:
            return "No graphic overlay was detected."
        return (
            f"A graphic overlay was detected with confidence {normalised:.2f}, "
            f"a strong indicator of physical tampering."
        )


class EdgeInconsistencyFactor(ScoreFactor):
    """Risk contribution from QR-boundary edge irregularity.

    Rationale
    ---------
    Tampering that involves printing or affixing a replacement QR code
    typically disturbs the clean, sharp edge of the original code's
    boundary. This factor passes the upstream normalised score through
    directly, since ``edge_inconsistency_score`` is already defined on
    ``[0.0, 1.0]`` by the tamper-analysis module's contract.
    """

    name: ClassVar[str] = "edge_inconsistency"

    def is_applicable(self, inputs: ScoringInputs) -> bool:
        return True

    def raw_value(self, inputs: ScoringInputs) -> float:
        return inputs.edge_inconsistency_score

    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        return max(0.0, min(1.0, float(inputs.edge_inconsistency_score)))

    def weight(self, config: ScoringConfig) -> float:
        return config.edge_inconsistency_weight

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        return f"Edge inconsistency score of {normalised:.2f} around the QR boundary."


class ContourMismatchFactor(ScoreFactor):
    """Risk contribution from QR contour-shape mismatch.

    Rationale
    ---------
    A QR code's outer contour should closely match the expected square
    (or near-square, accounting for perspective) silhouette. Significant
    deviation suggests the printed/displayed code has been altered,
    folded, or partially replaced.
    """

    name: ClassVar[str] = "contour_mismatch"

    def is_applicable(self, inputs: ScoringInputs) -> bool:
        return True

    def raw_value(self, inputs: ScoringInputs) -> float:
        return inputs.contour_mismatch_score

    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        return max(0.0, min(1.0, float(inputs.contour_mismatch_score)))

    def weight(self, config: ScoringConfig) -> float:
        return config.contour_mismatch_weight

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        return (
            f"Contour mismatch score of {normalised:.2f} relative to "
            f"expected QR geometry."
        )


class FinderPatternDamageFactor(ScoreFactor):
    """Risk contribution from finder-pattern damage or occlusion.

    Rationale
    ---------
    The three finder patterns (corner squares) are structurally essential
    to QR decoding and are a common target for crude tampering — covering
    or damaging them while leaving the data payload intact is a known
    phishing technique. This factor is weighted relatively heavily by
    default because finder-pattern damage is hard to produce accidentally.
    """

    name: ClassVar[str] = "finder_pattern_damage"

    def is_applicable(self, inputs: ScoringInputs) -> bool:
        return True

    def raw_value(self, inputs: ScoringInputs) -> float:
        return inputs.finder_pattern_damage_score

    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        return max(0.0, min(1.0, float(inputs.finder_pattern_damage_score)))

    def weight(self, config: ScoringConfig) -> float:
        return config.finder_pattern_damage_weight

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        return f"Finder pattern damage score of {normalised:.2f}."


class AnomalyCountFactor(ScoreFactor):
    """Corroborating risk contribution from the total count of anomalies.

    Rationale
    ---------
    A single weak indicator is often noise; several independent weak
    indicators co-occurring is meaningfully more suspicious. This factor
    rewards corroboration without double-counting the individual
    anomalies' own dedicated factors — it operates purely on the *count*,
    saturating at ``config.anomaly_count_saturation`` so that an
    unbounded anomaly count cannot dominate the score.
    """

    name: ClassVar[str] = "anomaly_count"

    def is_applicable(self, inputs: ScoringInputs) -> bool:
        return True

    def raw_value(self, inputs: ScoringInputs) -> int:
        return inputs.anomaly_count

    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        saturation = max(1, config.anomaly_count_saturation)
        return max(0.0, min(1.0, inputs.anomaly_count / saturation))

    def weight(self, config: ScoringConfig) -> float:
        return config.anomaly_count_weight

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        return (
            f"{raw} distinct anomal{'y' if raw == 1 else 'ies'} detected "
            f"(normalised corroboration score {normalised:.2f})."
        )


class TamperConfidenceFactor(ScoreFactor):
    """Heavily-weighted risk contribution from a tamper-analysis model.

    Rationale
    ---------
    Reserved for the future tamper-analysis module's aggregate confidence
    that the QR graphic has been tampered with. Weighted as the single
    largest contributor by default, since it is expected to be a holistic
    judgement that already incorporates lower-level signals such as edge,
    contour, and finder-pattern damage.
    """

    name: ClassVar[str] = "tamper_confidence"

    def is_applicable(self, inputs: ScoringInputs) -> bool:
        return inputs.tamper_confidence is not None

    def raw_value(self, inputs: ScoringInputs) -> float | None:
        return inputs.tamper_confidence

    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        confidence = inputs.tamper_confidence
        assert confidence is not None  # guaranteed by is_applicable
        return max(0.0, min(1.0, float(confidence)))

    def weight(self, config: ScoringConfig) -> float:
        return config.tamper_confidence_weight

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        return f"Tamper-analysis model confidence of {normalised:.2f}."


class AiConfidenceFactor(ScoreFactor):
    """Risk contribution from a future general-purpose AI confidence score.

    Rationale
    ---------
    Reserved for a learned classifier distinct from the dedicated tamper
    model (for example, a model trained on phishing-URL patterns once the
    decoded QR payload is analysed). Kept separate from
    :class:`TamperConfidenceFactor` so the two can be weighted and audited
    independently once both exist.
    """

    name: ClassVar[str] = "ai_confidence"

    def is_applicable(self, inputs: ScoringInputs) -> bool:
        return inputs.ai_confidence is not None

    def raw_value(self, inputs: ScoringInputs) -> float | None:
        return inputs.ai_confidence

    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        confidence = inputs.ai_confidence
        assert confidence is not None  # guaranteed by is_applicable
        return max(0.0, min(1.0, float(confidence)))

    def weight(self, config: ScoringConfig) -> float:
        return config.ai_confidence_weight

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        return f"General-purpose AI confidence of {normalised:.2f}."


class ExtraSignalFactor(ScoreFactor):
    """Generic factor for a single named entry in ``ScoringInputs.extra_signals``.

    Rationale
    ---------
    This is the primary extension point for signals that have not yet
    earned a dedicated field and :class:`ScoreFactor` subclass — useful
    during rapid research iteration. A new signal can be wired in purely
    through configuration (adding a key to
    ``ScoringConfig.extra_signal_weights`` and populating
    ``ScoringInputs.extra_signals``), with no new Python code required
    until the signal's normalisation logic needs to be more than a direct
    pass-through.

    Parameters
    ----------
    signal_key : str
        The key this instance reads from ``ScoringInputs.extra_signals``.
    """

    def __init__(self, signal_key: str) -> None:
        self._signal_key = signal_key

    @property
    def name(self) -> str:  # type: ignore[override]
        """Unique identifier derived from the wrapped signal key."""
        return f"extra:{self._signal_key}"

    def is_applicable(self, inputs: ScoringInputs) -> bool:
        return self._signal_key in inputs.extra_signals

    def raw_value(self, inputs: ScoringInputs) -> float:
        return inputs.extra_signals[self._signal_key]

    def normalise(self, inputs: ScoringInputs, config: ScoringConfig) -> float:
        return max(0.0, min(1.0, float(inputs.extra_signals[self._signal_key])))

    def weight(self, config: ScoringConfig) -> float:
        return config.extra_signal_weights.get(self._signal_key, 0.0)

    def explain(
        self,
        inputs: ScoringInputs,
        raw: float | bool | int | None,
        normalised: float,
    ) -> str:
        return f"Extra signal '{self._signal_key}' contributed {normalised:.2f}."


# ===========================================================================
# Factor Registry
# ===========================================================================

class FactorRegistry:
    """Ordered collection of :class:`ScoreFactor` instances.

    Decouples the set of active factors from :class:`ScoringEngine`,
    allowing callers to add, remove, or reorder factors (for research
    experiments, ablation studies, or future ML factors) without
    modifying engine code.

    Parameters
    ----------
    factors : list[ScoreFactor], optional
        Initial factors to register, in evaluation order. Defaults to the
        standard factor set returned by
        :meth:`FactorRegistry.default_factors`.
    """

    def __init__(self, factors: list[ScoreFactor] | None = None) -> None:
        self._factors: list[ScoreFactor] = (
            list(factors) if factors is not None else self.default_factors()
        )

    @staticmethod
    def default_factors() -> list[ScoreFactor]:
        """Return the standard set of factors for the current pipeline stage.

        Returns
        -------
        list[ScoreFactor]
            Includes every factor currently defined in this module.
            ``ExtraSignalFactor`` instances are added dynamically by
            :class:`ScoringEngine` based on
            ``ScoringConfig.extra_signal_weights`` keys, not here, since
            the registry has no access to a config at construction time.
        """
        return [
            DetectionConfidenceFactor(),
            OverlayFactor(),
            EdgeInconsistencyFactor(),
            ContourMismatchFactor(),
            FinderPatternDamageFactor(),
            AnomalyCountFactor(),
            TamperConfidenceFactor(),
            AiConfidenceFactor(),
        ]

    def register(self, factor: ScoreFactor) -> None:
        """Append a new factor to the registry.

        Parameters
        ----------
        factor : ScoreFactor
            The factor instance to add. Evaluated after all currently
            registered factors.

        Raises
        ------
        ValueError
            If a factor with the same ``name`` is already registered.
        """
        existing_names = {f.name for f in self._factors}
        if factor.name in existing_names:
            raise ValueError(
                f"A factor named {factor.name!r} is already registered."
            )
        self._factors.append(factor)
        logger.debug("Registered scoring factor: %s", factor.name)

    def unregister(self, name: str) -> None:
        """Remove a factor by name.

        Parameters
        ----------
        name : str
            The ``name`` of the factor to remove.

        Raises
        ------
        KeyError
            If no factor with that name is registered.
        """
        before = len(self._factors)
        self._factors = [f for f in self._factors if f.name != name]
        if len(self._factors) == before:
            raise KeyError(f"No registered factor named {name!r}.")
        logger.debug("Unregistered scoring factor: %s", name)

    @property
    def factors(self) -> tuple[ScoreFactor, ...]:
        """Read-only view of the currently registered factors, in order."""
        return tuple(self._factors)


# ===========================================================================
# Scoring Engine
# ===========================================================================

class ScoringEngine:
    """Computes a numerical ``[0, 100]`` risk score from detection signals.

    The engine is intentionally thin: it owns no scoring logic itself.
    It iterates the active :class:`FactorRegistry`, asks each factor
    whether it applies, normalises and weights applicable factors, and
    re-normalises the weighted sum onto ``[0, 100]``. All actual scoring
    semantics live in the individual :class:`ScoreFactor` implementations,
    which keeps this class trivial to reason about and test.

    Re-normalisation rationale
    ---------------------------
    Several inputs (tamper analysis, AI confidence) do not exist yet in
    the current pipeline stage. If the engine simply summed
    ``contribution`` across *all* registered factors and divided by the
    sum of *all* configured weights, a score computed today (QR detection
    only) would be artificially capped well below 100 even for maximally
    suspicious QR-only signals, because weight reserved for not-yet-built
    factors would always go unused. Instead, the denominator is the sum of
    weights for factors that *are* applicable to this particular
    :class:`ScoringInputs` instance, so the score always spans the full
    ``[0, 100]`` range using whatever evidence is actually available.

    Parameters
    ----------
    config : ScoringConfig, optional
        Weight configuration. Defaults to ``ScoringConfig()``.
    registry : FactorRegistry, optional
        Active factor set. Defaults to a registry built from
        :meth:`FactorRegistry.default_factors`, extended with one
        :class:`ExtraSignalFactor` per key in
        ``config.extra_signal_weights``.
    """

    def __init__(
        self,
        config:   ScoringConfig | None = None,
        registry: FactorRegistry | None = None,
    ) -> None:
        self._config = config if config is not None else ScoringConfig()

        if registry is not None:
            self._registry = registry
        else:
            self._registry = FactorRegistry()
            for signal_key in self._config.extra_signal_weights:
                self._registry.register(ExtraSignalFactor(signal_key))

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    @property
    def config(self) -> ScoringConfig:
        """Read-only access to the active configuration."""
        return self._config

    @property
    def registry(self) -> FactorRegistry:
        """Read-only access to the active factor registry."""
        return self._registry

    def compute_score(self, inputs: ScoringInputs) -> ScoreBreakdown:
        """Compute a full, explainable risk score from *inputs*.

        Parameters
        ----------
        inputs : ScoringInputs
            The detection / analysis signals to score.

        Returns
        -------
        ScoreBreakdown
            The total score, per-factor breakdown, and explanation.
            ``total_score`` is always clamped to ``[0, 100]``.

        Raises
        ------
        TypeError
            If *inputs* is not a :class:`ScoringInputs` instance.

        Examples
        --------
        No tamper signals available yet, QR detection only::

            inputs = ScoringInputs(detection_confidence=0.95)
            breakdown = engine.compute_score(inputs)
            assert 0.0 <= breakdown.total_score <= 100.0

        Strong overlay + finder-pattern damage::

            inputs = ScoringInputs(
                overlay_detected=True,
                overlay_confidence=0.9,
                finder_pattern_damage_score=0.8,
            )
            breakdown = engine.compute_score(inputs)
            assert breakdown.total_score > 50.0
        """
        if not isinstance(inputs, ScoringInputs):
            raise TypeError(
                f"inputs must be a ScoringInputs instance, "
                f"got {type(inputs).__name__!r}."
            )

        factor_scores: list[FactorScore] = []
        weighted_sum = 0.0
        applicable_weight_total = 0.0

        for factor in self._registry.factors:
            applicable = factor.is_applicable(inputs)

            if not applicable:
                factor_scores.append(
                    FactorScore(
                        factor_name=factor.name,
                        raw_value=None,
                        normalised_value=0.0,
                        weight=factor.weight(self._config),
                        contribution=0.0,
                        applicable=False,
                        explanation=f"{factor.name}: not applicable (no input data).",
                    )
                )
                continue

            raw = factor.raw_value(inputs)
            normalised = max(0.0, min(1.0, factor.normalise(inputs, self._config)))
            weight = max(0.0, factor.weight(self._config))
            contribution = normalised * weight

            weighted_sum += contribution
            applicable_weight_total += weight

            factor_scores.append(
                FactorScore(
                    factor_name=factor.name,
                    raw_value=raw,
                    normalised_value=normalised,
                    weight=weight,
                    contribution=contribution,
                    applicable=True,
                    explanation=factor.explain(inputs, raw, normalised),
                )
            )

        if applicable_weight_total > 0.0:
            total_score = (weighted_sum / applicable_weight_total) * SCORE_MAX
        else:
            # No applicable factors at all (e.g. an entirely empty
            # ScoringInputs with every optional field at its zero-risk
            # default). A score of 0 is the only defensible output —
            # there is no evidence of risk to weigh.
            total_score = SCORE_MIN
            logger.warning(
                "ScoringEngine.compute_score: no applicable factors; "
                "defaulting to minimum score."
            )

        total_score = max(SCORE_MIN, min(SCORE_MAX, total_score))

        explanation = self._build_explanation(factor_scores, total_score)

        logger.info(
            "ScoringEngine.compute_score — total_score=%.2f, "
            "applicable_factors=%d/%d",
            total_score,
            sum(1 for fs in factor_scores if fs.applicable),
            len(factor_scores),
        )

        return ScoreBreakdown(
            total_score=total_score,
            factor_scores=tuple(factor_scores),
            applicable_weight_total=applicable_weight_total,
            explanation=explanation,
            config_snapshot=self._config.as_dict,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_explanation(
        factor_scores: list[FactorScore],
        total_score:   float,
    ) -> str:
        """Build a multi-sentence narrative summarising the score breakdown.

        Parameters
        ----------
        factor_scores : list[FactorScore]
            All factor scores, applicable and inapplicable.
        total_score : float
            The final computed score.

        Returns
        -------
        str
            A narrative beginning with the total score, followed by the
            top contributing applicable factors in descending order of
            contribution.
        """
        applicable = [fs for fs in factor_scores if fs.applicable]
        contributing = sorted(
            (fs for fs in applicable if fs.contribution > 0.0),
            key=lambda fs: fs.contribution,
            reverse=True,
        )

        if not contributing:
            return (
                f"Total risk score: {total_score:.2f}/100. "
                f"No risk-contributing factors were present in the "
                f"available data."
            )

        lead_sentences = [fs.explanation for fs in contributing[:3]]
        narrative = " ".join(lead_sentences)

        return f"Total risk score: {total_score:.2f}/100. {narrative}"


# ===========================================================================
# Module-level convenience factory
# ===========================================================================

def create_default_engine() -> ScoringEngine:
    """Instantiate a :class:`ScoringEngine` with default configuration.

    Convenience function for ``risk_engine.py`` and testing harnesses that
    do not need custom weight tuning.

    Returns
    -------
    ScoringEngine
        Engine using ``ScoringConfig()`` defaults and the standard factor
        registry from :meth:`FactorRegistry.default_factors`.
    """
    return ScoringEngine(config=ScoringConfig())


def compute_score(
    inputs: ScoringInputs,
    config: ScoringConfig | None = None,
) -> ScoreBreakdown:
    """Module-level convenience wrapper around :class:`ScoringEngine`.

    Useful for simple call sites (and ``risk_engine.py``) that do not need
    to retain an engine instance across calls.

    Parameters
    ----------
    inputs : ScoringInputs
        The detection / analysis signals to score.
    config : ScoringConfig, optional
        Weight configuration. Defaults to ``ScoringConfig()``.

    Returns
    -------
    ScoreBreakdown
        The total score, per-factor breakdown, and explanation.
    """
    engine = ScoringEngine(config=config)
    return engine.compute_score(inputs)


# ===========================================================================
# Demo / development entry-point
# ===========================================================================

def _demo() -> None:  # pragma: no cover
    """Demonstrate the Scoring Engine with representative test cases.

    Run directly::

        python src/risk_assessment/scoring.py
    """
    import json

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    engine = create_default_engine()

    test_cases: list[dict[str, Any]] = [
        {
            "label":  "Clean detection, no tamper signals available yet",
            "inputs": ScoringInputs(detection_confidence=0.97),
        },
        {
            "label":  "Low detection confidence only",
            "inputs": ScoringInputs(detection_confidence=0.40),
        },
        {
            "label":  "Confident overlay detection",
            "inputs": ScoringInputs(
                detection_confidence=0.90,
                overlay_detected=True,
                overlay_confidence=0.92,
            ),
        },
        {
            "label":  "Multiple weak signals corroborating each other",
            "inputs": ScoringInputs(
                detection_confidence=0.85,
                edge_inconsistency_score=0.35,
                contour_mismatch_score=0.30,
                finder_pattern_damage_score=0.25,
                anomaly_count=3,
            ),
        },
        {
            "label":  "Full pipeline, high tamper confidence",
            "inputs": ScoringInputs(
                detection_confidence=0.93,
                overlay_detected=True,
                overlay_confidence=0.80,
                edge_inconsistency_score=0.60,
                contour_mismatch_score=0.55,
                finder_pattern_damage_score=0.70,
                anomaly_count=4,
                tamper_confidence=0.88,
                ai_confidence=0.75,
            ),
        },
        {
            "label":  "Empty inputs (no evidence at all)",
            "inputs": ScoringInputs(),
        },
    ]

    separator = "=" * 70

    for case in test_cases:
        breakdown = engine.compute_score(case["inputs"])
        print(f"\n{separator}")
        print(f"  CASE : {case['label']}")
        print(f"  Total score: {breakdown.total_score:.2f}/100")
        print(f"  Explanation: {breakdown.explanation}")
        print("  Factor breakdown (applicable only):")
        for fs in breakdown.applicable_factor_scores:
            print(
                f"    - {fs.factor_name:<24} raw={fs.raw_value!r:<8} "
                f"norm={fs.normalised_value:.2f}  weight={fs.weight:.1f}  "
                f"contribution={fs.contribution:.2f}"
            )
        print("  Serialised (to_dict, truncated):")
        as_dict = breakdown.to_dict()
        print(
            json.dumps(
                {
                    "total_score": as_dict["total_score"],
                    "applicable_weight_total": as_dict["applicable_weight_total"],
                },
                indent=4,
            )
        )

    print(f"\n{separator}")
    print("  Demo complete.")
    print(separator)


if __name__ == "__main__":
    _demo()