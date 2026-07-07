"""
rule_engine.py
==============
Week 3 – Member 2: Risk Assessment Engine
Project: Computer Vision-Based Graphic Tamper Detection for QR Code Phishing Prevention

Overview
--------
The Rule Engine is the decision core of the Risk Assessment pipeline.
It accepts a numerical risk score and a dictionary of anomaly indicators
produced upstream (e.g. by ``scoring.py``) and converts them into a
structured, human-readable cybersecurity decision.

Design philosophy
-----------------
* **Configurable** — all threshold and weight parameters are injected at
  construction time via :class:`RuleEngineConfig`; no hardcoded magic
  numbers exist in decision logic.
* **Explainable** — every decision carries a ``reasons`` list and a
  ``decision_explanation`` string that can be displayed to end-users,
  security analysts, or surfaced verbatim in research publications.
* **Extensible** — the :class:`RuleSet` abstraction allows new rule
  families (weighted rules, ML-based rules, domain-specific overrides)
  to be registered without modifying existing logic.
* **Independent** — this module has *zero* imports from ``tamper_detector``
  or ``scoring``; it operates purely on the numeric and boolean values
  passed in by the caller (``risk_engine.py``).  Its only intra-project
  dependency is ``risk_result.RiskLevel``, the shared, dependency-free
  enum that both this module and ``risk_result.py`` use as the single
  source of truth for risk classification (see the Week 4 compatibility
  note near the top of the implementation below).

Pipeline position
-----------------
::

    scoring.py  ──►  risk_engine.py  ──►  rule_engine.py  ──►  RiskResult
                                                           └──►  report_generator.py
                                                           └──►  FastAPI backend
                                                           └──►  Flutter mobile app

Output structure
----------------
:meth:`RuleEngine.evaluate` returns a :class:`RuleEngineResult` dataclass
that is directly consumable by ``risk_engine.py`` to construct a
``RiskResult`` object::

    RuleEngineResult(
        risk_level        = RiskLevel.HIGH_RISK,
        recommendation    = "Do not proceed. ...",
        reasons           = ["Score 85 exceeds HIGH_RISK threshold (61)", ...],
        decision_explanation = "Risk score of 85/100 triggers HIGH_RISK ...",
        applied_rules     = ["threshold_classifier", "anomaly_override"],
        score             = 85.0,
        threshold_config  = {...},
    )

Usage
-----
::

    from src.risk_assessment.rule_engine import RuleEngine, RuleEngineConfig

    config = RuleEngineConfig(safe_max=30, suspicious_max=60)
    engine = RuleEngine(config=config)

    result = engine.evaluate(
        score=72.5,
        anomaly_indicators={
            "url_mismatch":    True,
            "domain_spoofing": False,
            "logo_tampering":  True,
            "qr_overlay":      False,
        },
    )

    print(result.risk_level)            # RiskLevel.HIGH_RISK
    print(result.recommendation)        # "Do not proceed …"
    print(result.decision_explanation)  # Full narrative string
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.risk_assessment.risk_result import RiskLevel

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk Level Enumeration
# ---------------------------------------------------------------------------
# NOTE (Week 4 audit): RiskLevel previously had a second, independently
# defined copy in this module with values identical to
# ``risk_result.RiskLevel``.  Two distinct ``Enum`` classes sharing the same
# string values are *not* interchangeable via ``is`` / class identity (only
# via ``==`` against the ``.value``), which forced ``risk_engine.py`` to
# rebuild a ``risk_result.RiskLevel`` from ``rule_result.risk_level.value``
# on every call.  Importing the single canonical enum here removes that
# duplication, eliminates the value-based reconstruction downstream, and
# guarantees the two modules can never drift out of sync (e.g. if a new
# risk tier is ever added to one but not the other).  This module remains
# independent of ``tamper_detector`` and ``scoring`` — ``risk_result`` is a
# lightweight, dependency-free data-contract module, not a computation
# module, so importing it does not reintroduce the coupling this module's
# design philosophy warns against.


# ===========================================================================
# Configuration Dataclass
# ===========================================================================

@dataclass
class RuleEngineConfig:
    """Threshold and weight configuration for the Rule Engine.

    All numeric boundaries are inclusive on the lower bound and exclusive
    on the upper bound, i.e. ``[0, safe_max]``, ``(safe_max, suspicious_max]``,
    ``(suspicious_max, 100]``.

    Parameters
    ----------
    safe_max : float
        Maximum score (inclusive) that maps to :attr:`RiskLevel.SAFE`.
        Default ``30``.
    suspicious_max : float
        Maximum score (inclusive) that maps to :attr:`RiskLevel.SUSPICIOUS`.
        Scores above this threshold map to :attr:`RiskLevel.HIGH_RISK`.
        Default ``60``.
    anomaly_override_enabled : bool
        When ``True``, critical anomaly indicators can escalate the risk
        level beyond what the numeric score alone would produce.  This
        implements a defence-in-depth principle: a low score does not
        guarantee safety if a high-confidence indicator fires.
        Default ``True``.
    critical_anomaly_keys : list[str]
        Names of anomaly indicator keys whose presence (``True``) should
        trigger an immediate escalation to :attr:`RiskLevel.HIGH_RISK`
        when ``anomaly_override_enabled`` is ``True``.
        Default ``["url_mismatch", "domain_spoofing", "qr_overlay"]``.
    score_weight : float
        Weight of the numeric score in future weighted-rule extensions.
        Currently informational; reserved for ML integration.
        Default ``1.0``.

    Raises
    ------
    ValueError
        If ``safe_max >= suspicious_max``, or if either value is outside
        ``[0, 100]``.
    """

    safe_max:                  float      = 30.0
    suspicious_max:            float      = 60.0
    anomaly_override_enabled:  bool       = True
    critical_anomaly_keys:     list[str]  = field(
        default_factory=lambda: ["url_mismatch", "domain_spoofing", "qr_overlay"]
    )
    score_weight:              float      = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.safe_max < self.suspicious_max <= 100.0):
            raise ValueError(
                f"Invalid threshold configuration: safe_max={self.safe_max}, "
                f"suspicious_max={self.suspicious_max}. "
                "Required: 0 ≤ safe_max < suspicious_max ≤ 100."
            )
        if not (0.0 < self.score_weight <= 10.0):
            raise ValueError(
                f"score_weight must be in (0.0, 10.0], got {self.score_weight}."
            )

    @property
    def as_dict(self) -> dict[str, Any]:
        """Serialise the config for inclusion in audit trails and reports."""
        return {
            "safe_max":                 self.safe_max,
            "suspicious_max":           self.suspicious_max,
            "high_risk_min":            self.suspicious_max + 1,
            "anomaly_override_enabled": self.anomaly_override_enabled,
            "critical_anomaly_keys":    list(self.critical_anomaly_keys),
            "score_weight":             self.score_weight,
        }


# ===========================================================================
# Rule Engine Result Dataclass
# ===========================================================================

@dataclass
class RuleEngineResult:
    """Structured output produced by :class:`RuleEngine`.

    This dataclass is intentionally flat and JSON-serialisable so that
    ``risk_engine.py`` can map it directly onto ``RiskResult`` fields,
    and downstream consumers (FastAPI, Flutter, report_generator) can
    consume it without additional transformation.

    Attributes
    ----------
    risk_level : RiskLevel
        The final categorical risk classification.
    recommendation : str
        A single, actionable sentence directed at the end-user or
        security system that consumed this result.
    reasons : list[str]
        An ordered list of human-readable strings describing each
        condition that contributed to the risk decision.  Each item is
        a complete sentence beginning with a capital letter.
    decision_explanation : str
        A multi-sentence narrative that explains the full decision
        context — suitable for audit logs, research publications, and
        the report_generator.
    applied_rules : list[str]
        Identifiers of the rule handlers that participated in producing
        this result.  Useful for debugging and reproducibility.
    score : float
        The raw numeric score that was evaluated (0–100).
    threshold_config : dict[str, Any]
        Snapshot of the :class:`RuleEngineConfig` at evaluation time,
        embedded for full auditability.
    """

    risk_level:           RiskLevel
    recommendation:       str
    reasons:              list[str]
    decision_explanation: str
    applied_rules:        list[str]
    score:                float
    threshold_config:     dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for JSON serialisation.

        Returns
        -------
        dict[str, Any]
            All fields with ``risk_level`` serialised as its string value.
        """
        return {
            "risk_level":           self.risk_level.value,
            "recommendation":       self.recommendation,
            "reasons":              list(self.reasons),
            "decision_explanation": self.decision_explanation,
            "applied_rules":        list(self.applied_rules),
            "score":                self.score,
            "threshold_config":     dict(self.threshold_config),
        }


# ===========================================================================
# Abstract Rule Interface
# ===========================================================================

class BaseRule(ABC):
    """Abstract base class for all rule implementations.

    Every rule receives the full evaluation context and may append to
    ``reasons``, modify ``risk_level``, and record itself in
    ``applied_rules``.  Rules are applied sequentially by
    :class:`RuleSet`; later rules can observe and override decisions
    made by earlier ones.

    Subclass contract
    -----------------
    Implement :meth:`apply`.  Do not raise exceptions — catch internal
    errors and append a diagnostic reason instead, to preserve pipeline
    continuity.

    Attributes
    ----------
    rule_id : str
        Short, unique identifier used in ``applied_rules`` lists and logs.
    """

    rule_id: str = "base_rule"

    @abstractmethod
    def apply(
        self,
        score:              float,
        anomaly_indicators: dict[str, bool],
        current_level:      RiskLevel,
        reasons:            list[str],
        config:             RuleEngineConfig,
    ) -> RiskLevel:
        """Evaluate this rule and return the (possibly updated) risk level.

        Parameters
        ----------
        score : float
            Numeric risk score in ``[0, 100]``.
        anomaly_indicators : dict[str, bool]
            Boolean flags for each anomaly type detected upstream.
        current_level : RiskLevel
            The risk level produced by all previously applied rules.
        reasons : list[str]
            Mutable list; append a reason string for every condition this
            rule triggers.
        config : RuleEngineConfig
            Active threshold configuration.

        Returns
        -------
        RiskLevel
            The updated (or unchanged) risk level after applying this rule.
        """


# ===========================================================================
# Concrete Rule Implementations
# ===========================================================================

class ThresholdClassifierRule(BaseRule):
    """Classifies risk based purely on numeric score thresholds.

    This is the primary rule and runs first.  It establishes the
    baseline :class:`RiskLevel` using the boundaries defined in
    :class:`RuleEngineConfig`.

    Rule logic
    ----------
    * ``0 ≤ score ≤ safe_max``          → :attr:`RiskLevel.SAFE`
    * ``safe_max < score ≤ suspicious_max`` → :attr:`RiskLevel.SUSPICIOUS`
    * ``score > suspicious_max``         → :attr:`RiskLevel.HIGH_RISK`
    """

    rule_id: str = "threshold_classifier"

    def apply(
        self,
        score:              float,
        anomaly_indicators: dict[str, bool],
        current_level:      RiskLevel,
        reasons:            list[str],
        config:             RuleEngineConfig,
    ) -> RiskLevel:
        """Apply threshold-based classification."""
        if score <= config.safe_max:
            level = RiskLevel.SAFE
            reasons.append(
                f"Risk score {score:.1f} is within the SAFE threshold "
                f"(≤ {config.safe_max:.0f})."
            )
        elif score <= config.suspicious_max:
            level = RiskLevel.SUSPICIOUS
            reasons.append(
                f"Risk score {score:.1f} falls within the SUSPICIOUS range "
                f"({config.safe_max:.0f}–{config.suspicious_max:.0f})."
            )
        else:
            level = RiskLevel.HIGH_RISK
            reasons.append(
                f"Risk score {score:.1f} exceeds the HIGH_RISK threshold "
                f"(> {config.suspicious_max:.0f})."
            )

        logger.debug(
            "[%s] score=%.1f → %s", self.rule_id, score, level.value
        )
        return level


class AnomalyOverrideRule(BaseRule):
    """Escalates risk level when critical anomaly indicators are present.

    Even if the numeric score is low, certain binary anomaly signals
    (e.g. confirmed URL mismatch, domain spoofing) are treated as
    high-confidence evidence of malicious intent.  This rule implements
    a defence-in-depth escalation: score-based classification is
    necessary but not sufficient for a SAFE verdict.

    The escalation only ever increases the risk level — it will never
    downgrade a HIGH_RISK decision produced by a previous rule.

    This rule is a no-op when
    :attr:`RuleEngineConfig.anomaly_override_enabled` is ``False``.
    """

    rule_id: str = "anomaly_override"

    def apply(
        self,
        score:              float,
        anomaly_indicators: dict[str, bool],
        current_level:      RiskLevel,
        reasons:            list[str],
        config:             RuleEngineConfig,
    ) -> RiskLevel:
        """Escalate to HIGH_RISK if any critical anomaly indicator is active."""
        if not config.anomaly_override_enabled:
            logger.debug("[%s] override disabled — skipping.", self.rule_id)
            return current_level

        triggered: list[str] = [
            key
            for key in config.critical_anomaly_keys
            if anomaly_indicators.get(key, False)
        ]

        if not triggered:
            logger.debug(
                "[%s] No critical anomaly indicators active.", self.rule_id
            )
            return current_level

        indicator_str = ", ".join(triggered)
        reasons.append(
            f"Critical anomaly indicator(s) detected: [{indicator_str}]. "
            "Risk level escalated to HIGH_RISK regardless of numeric score."
        )
        logger.warning(
            "[%s] Critical indicators [%s] triggered HIGH_RISK escalation "
            "(score=%.1f).",
            self.rule_id, indicator_str, score,
        )
        return RiskLevel.HIGH_RISK


class AnomalyContextRule(BaseRule):
    """Annotates the decision with non-critical anomaly context.

    This rule does not change the risk level.  It appends informational
    reason strings for anomaly indicators that are present but not
    classified as critical, providing richer context for analysts and
    the ``report_generator``.
    """

    rule_id: str = "anomaly_context"

    def apply(
        self,
        score:              float,
        anomaly_indicators: dict[str, bool],
        current_level:      RiskLevel,
        reasons:            list[str],
        config:             RuleEngineConfig,
    ) -> RiskLevel:
        """Record non-critical anomaly indicators as contextual reasons."""
        non_critical_active = [
            key
            for key, active in anomaly_indicators.items()
            if active and key not in config.critical_anomaly_keys
        ]

        if non_critical_active:
            indicator_str = ", ".join(non_critical_active)
            reasons.append(
                f"Additional anomaly indicator(s) noted (non-critical): "
                f"[{indicator_str}]."
            )
            logger.debug(
                "[%s] Non-critical indicators noted: [%s].",
                self.rule_id, indicator_str,
            )

        inactive_critical = [
            key
            for key in config.critical_anomaly_keys
            if not anomaly_indicators.get(key, False)
        ]
        if inactive_critical:
            logger.debug(
                "[%s] Critical indicators not triggered: [%s].",
                self.rule_id, ", ".join(inactive_critical),
            )

        return current_level


# ===========================================================================
# Rule Set — Ordered Rule Execution Registry
# ===========================================================================

class RuleSet:
    """An ordered, mutable registry of :class:`BaseRule` instances.

    Rules are applied in insertion order.  The :class:`RuleEngine` builds
    a default ``RuleSet`` but callers may inject a custom one (e.g. for
    testing or ML-extended pipelines).

    Parameters
    ----------
    rules : list[BaseRule], optional
        Initial list of rules.  If omitted, the set starts empty.

    Example — registering a custom rule::

        class MLScoringRule(BaseRule):
            rule_id = "ml_scorer"
            def apply(self, score, anomaly_indicators, current_level,
                      reasons, config) -> RiskLevel:
                # ... ML inference ...
                return current_level

        rule_set = RuleSet()
        rule_set.register(ThresholdClassifierRule())
        rule_set.register(MLScoringRule())
    """

    def __init__(self, rules: list[BaseRule] | None = None) -> None:
        self._rules: list[BaseRule] = list(rules) if rules else []

    def register(self, rule: BaseRule) -> None:
        """Append a rule to the end of the execution chain.

        Parameters
        ----------
        rule : BaseRule
            Any concrete subclass of :class:`BaseRule`.
        """
        self._rules.append(rule)
        logger.debug("Registered rule: %s", rule.rule_id)

    def apply_all(
        self,
        score:              float,
        anomaly_indicators: dict[str, bool],
        reasons:            list[str],
        config:             RuleEngineConfig,
    ) -> tuple[RiskLevel, list[str]]:
        """Execute all registered rules in order.

        Parameters
        ----------
        score : float
            Numeric risk score ``[0, 100]``.
        anomaly_indicators : dict[str, bool]
            Upstream anomaly flags.
        reasons : list[str]
            Mutable reason accumulator shared across all rules.
        config : RuleEngineConfig
            Active configuration instance.

        Returns
        -------
        final_level : RiskLevel
            The risk level after all rules have been applied.
        applied_rule_ids : list[str]
            Ordered list of rule identifiers that participated.
        """
        current_level: RiskLevel = RiskLevel.SAFE
        applied_ids:   list[str] = []

        for rule in self._rules:
            try:
                current_level = rule.apply(
                    score=score,
                    anomaly_indicators=anomaly_indicators,
                    current_level=current_level,
                    reasons=reasons,
                    config=config,
                )
                applied_ids.append(rule.rule_id)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Rule '%s' raised an unexpected exception: %s — skipping.",
                    rule.rule_id, exc,
                )
                reasons.append(
                    f"Rule '{rule.rule_id}' encountered an internal error "
                    "and was skipped."
                )

        return current_level, applied_ids


# ===========================================================================
# Recommendation and Explanation Factories
# ===========================================================================

# Recommendation templates keyed by RiskLevel.
# These are module-level constants — editable without touching rule logic.
_RECOMMENDATIONS: dict[RiskLevel, str] = {
    RiskLevel.SAFE: (
        "The QR code appears legitimate. Proceed with standard caution "
        "and verify the destination URL in your browser before entering "
        "any credentials."
    ),
    RiskLevel.SUSPICIOUS: (
        "Exercise caution. Anomalies were detected that may indicate "
        "tampering or phishing activity. Do not enter personal information "
        "until the destination URL has been independently verified."
    ),
    RiskLevel.HIGH_RISK: (
        "Do not proceed. This QR code exhibits strong indicators of "
        "phishing or graphic tampering. Block this code, avoid visiting "
        "the embedded URL, and report this incident to your security team."
    ),
}


def _build_recommendation(level: RiskLevel) -> str:
    """Return the standard recommendation string for *level*.

    Falls back to a generic safe message if the level is unrecognised.
    """
    return _RECOMMENDATIONS.get(
        level,
        "Unable to determine a recommendation. Treat this QR code as "
        "HIGH_RISK and do not proceed.",
    )


def _build_explanation(
    score:              float,
    level:              RiskLevel,
    reasons:            list[str],
    anomaly_indicators: dict[str, bool],
    config:             RuleEngineConfig,
) -> str:
    """Compose a structured narrative decision explanation.

    The explanation is designed to be directly embeddable in research
    publications, audit logs, and user-facing security reports without
    further editing.

    Parameters
    ----------
    score : float
        The evaluated numeric risk score.
    level : RiskLevel
        The final risk classification.
    reasons : list[str]
        Ordered list of reasons produced by all applied rules.
    anomaly_indicators : dict[str, bool]
        Full anomaly indicator dictionary for context.
    config : RuleEngineConfig
        Active configuration snapshot.

    Returns
    -------
    str
        Multi-sentence explanation string.
    """
    active_anomalies   = [k for k, v in anomaly_indicators.items() if v]
    inactive_anomalies = [k for k, v in anomaly_indicators.items() if not v]

    active_str   = (
        f"[{', '.join(active_anomalies)}]"   if active_anomalies   else "none"
    )
    inactive_str = (
        f"[{', '.join(inactive_anomalies)}]" if inactive_anomalies else "none"
    )

    reasons_narrative = " ".join(reasons) if reasons else "No specific reasons recorded."

    explanation = (
        f"Risk Assessment Decision — {level.value}\n"
        f"\n"
        f"Evaluated risk score: {score:.1f} / 100.\n"
        f"Classification thresholds: SAFE ≤ {config.safe_max:.0f}, "
        f"SUSPICIOUS ≤ {config.suspicious_max:.0f}, "
        f"HIGH_RISK > {config.suspicious_max:.0f}.\n"
        f"\n"
        f"Active anomaly indicators:   {active_str}.\n"
        f"Inactive anomaly indicators: {inactive_str}.\n"
        f"\n"
        f"Decision rationale: {reasons_narrative}\n"
        f"\n"
        f"Final classification: {level.value}."
    )
    return explanation


# ===========================================================================
# Rule Engine — Public API
# ===========================================================================

class RuleEngine:
    """Converts anomaly indicators and a numeric risk score into a
    structured cybersecurity decision.

    The engine owns a :class:`RuleSet` and a :class:`RuleEngineConfig`.
    Both are injectable at construction time for testability and for
    future ML pipeline integration.

    Default rule chain (applied in order)
    --------------------------------------
    1. :class:`ThresholdClassifierRule` — baseline score classification.
    2. :class:`AnomalyOverrideRule`     — critical indicator escalation.
    3. :class:`AnomalyContextRule`      — non-critical context annotation.

    Parameters
    ----------
    config : RuleEngineConfig, optional
        Threshold and weight configuration.  Defaults to
        :class:`RuleEngineConfig` with standard thresholds
        (SAFE ≤ 30, SUSPICIOUS ≤ 60, HIGH_RISK > 60).
    rule_set : RuleSet, optional
        Custom ordered rule chain.  When omitted, the default chain
        described above is constructed automatically.

    Example
    -------
    ::

        engine = RuleEngine()
        result = engine.evaluate(
            score=45.0,
            anomaly_indicators={"url_mismatch": False, "logo_tampering": True},
        )
        print(result.risk_level)         # RiskLevel.SUSPICIOUS
        print(result.recommendation)
        for r in result.reasons:
            print(" -", r)
    """

    def __init__(
        self,
        config:   RuleEngineConfig | None = None,
        rule_set: RuleSet          | None = None,
    ) -> None:
        self._config:   RuleEngineConfig = config   or RuleEngineConfig()
        self._rule_set: RuleSet          = rule_set or self._build_default_rule_set()
        logger.info(
            "RuleEngine initialised — safe_max=%.0f, suspicious_max=%.0f, "
            "anomaly_override=%s",
            self._config.safe_max,
            self._config.suspicious_max,
            self._config.anomaly_override_enabled,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_default_rule_set() -> RuleSet:
        """Construct and return the default ordered rule chain."""
        rs = RuleSet()
        rs.register(ThresholdClassifierRule())
        rs.register(AnomalyOverrideRule())
        rs.register(AnomalyContextRule())
        return rs

    @staticmethod
    def _validate_score(score: float) -> None:
        """Validate that *score* is within the expected [0, 100] range.

        Parameters
        ----------
        score : float
            The risk score to validate.

        Raises
        ------
        ValueError
            If *score* is outside ``[0, 100]``.
        TypeError
            If *score* is not numeric.
        """
        if not isinstance(score, (int, float)):
            raise TypeError(
                f"score must be a numeric type (int or float), got {type(score).__name__}."
            )
        if not (0.0 <= float(score) <= 100.0):
            raise ValueError(
                f"score must be in [0, 100], got {score}."
            )

    @staticmethod
    def _normalise_indicators(
        raw: dict[str, Any],
    ) -> dict[str, bool]:
        """Coerce all indicator values to ``bool``.

        Upstream modules may pass numeric confidence scores (e.g. ``0.87``)
        or string flags.  This normalisation ensures the rule logic always
        operates on clean boolean values.

        Parameters
        ----------
        raw : dict[str, Any]
            Raw anomaly indicator dictionary from ``scoring.py`` or
            ``risk_engine.py``.

        Returns
        -------
        dict[str, bool]
            Dictionary with every value coerced to ``bool``.
        """
        normalised: dict[str, bool] = {}
        for key, value in raw.items():
            coerced = bool(value)
            if not isinstance(value, bool):
                logger.debug(
                    "Indicator '%s' coerced from %r (%s) → %s.",
                    key, value, type(value).__name__, coerced,
                )
            normalised[key] = coerced
        return normalised

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    @property
    def config(self) -> RuleEngineConfig:
        """Read-only access to the active configuration."""
        return self._config

    def evaluate(
        self,
        score:              float,
        anomaly_indicators: dict[str, Any] | None = None,
    ) -> RuleEngineResult:
        """Evaluate a risk score and anomaly context, returning a full decision.

        This is the primary public method.  ``risk_engine.py`` calls this
        once per QR code detection event and maps the result onto a
        ``RiskResult`` instance.

        Parameters
        ----------
        score : float
            Numeric risk score produced by ``scoring.py``, expected to
            be in the closed interval ``[0, 100]``.
        anomaly_indicators : dict[str, Any], optional
            Mapping of anomaly type names to their activation status.
            Values are coerced to ``bool`` internally, so callers may
            pass confidence floats (truthy/falsy) if needed.
            Defaults to an empty dict (no anomalies reported).

        Returns
        -------
        RuleEngineResult
            Complete decision record including risk level, recommendation,
            reasons, explanation, applied rules, score, and config snapshot.

        Raises
        ------
        TypeError
            If *score* is not numeric.
        ValueError
            If *score* is outside ``[0, 100]``.

        Examples
        --------
        Safe QR code::

            result = engine.evaluate(score=15.0)
            assert result.risk_level == RiskLevel.SAFE

        Suspicious by score only::

            result = engine.evaluate(score=45.0, anomaly_indicators={})
            assert result.risk_level == RiskLevel.SUSPICIOUS

        HIGH_RISK via anomaly escalation despite low score::

            result = engine.evaluate(
                score=20.0,
                anomaly_indicators={"url_mismatch": True},
            )
            assert result.risk_level == RiskLevel.HIGH_RISK
        """
        # 1. Input validation and normalisation --------------------------------
        self._validate_score(score)
        score = float(score)

        if anomaly_indicators is None:
            anomaly_indicators = {}

        clean_indicators: dict[str, bool] = self._normalise_indicators(
            anomaly_indicators
        )

        logger.info(
            "RuleEngine.evaluate — score=%.1f, indicators=%s",
            score,
            {k: v for k, v in clean_indicators.items() if v} or "none",
        )

        # 2. Execute rule chain ------------------------------------------------
        reasons:      list[str] = []
        final_level, applied_ids = self._rule_set.apply_all(
            score=score,
            anomaly_indicators=clean_indicators,
            reasons=reasons,
            config=self._config,
        )

        # 3. Build outputs -----------------------------------------------------
        recommendation = _build_recommendation(final_level)
        explanation    = _build_explanation(
            score=score,
            level=final_level,
            reasons=reasons,
            anomaly_indicators=clean_indicators,
            config=self._config,
        )

        result = RuleEngineResult(
            risk_level=final_level,
            recommendation=recommendation,
            reasons=reasons,
            decision_explanation=explanation,
            applied_rules=applied_ids,
            score=score,
            threshold_config=self._config.as_dict,
        )

        logger.info(
            "RuleEngine decision — level=%s, rules_applied=%s, "
            "reasons_count=%d",
            result.risk_level.value,
            result.applied_rules,
            len(result.reasons),
        )

        return result


# ===========================================================================
# Module-level convenience factory
# ===========================================================================

def create_default_engine() -> RuleEngine:
    """Instantiate a :class:`RuleEngine` with default configuration.

    Convenience function for ``risk_engine.py`` and testing harnesses
    that do not need custom threshold tuning.

    Returns
    -------
    RuleEngine
        Engine with standard thresholds: SAFE ≤ 30, SUSPICIOUS ≤ 60,
        HIGH_RISK > 60, anomaly override enabled.
    """
    return RuleEngine(config=RuleEngineConfig())


# ===========================================================================
# Demo / development entry-point
# ===========================================================================

def _demo() -> None:  # pragma: no cover
    """Demonstrate the Rule Engine with representative test cases.

    Run directly::

        python src/risk_assessment/rule_engine.py
    """
    import json

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    engine = create_default_engine()

    test_cases: list[dict[str, Any]] = [
        {
            "label":    "Clearly safe QR code",
            "score":    10.0,
            "indicators": {"url_mismatch": False, "logo_tampering": False},
        },
        {
            "label":    "Borderline safe (edge of threshold)",
            "score":    30.0,
            "indicators": {},
        },
        {
            "label":    "Suspicious by score",
            "score":    45.0,
            "indicators": {"logo_tampering": True},
        },
        {
            "label":    "HIGH_RISK by score alone",
            "score":    78.5,
            "indicators": {"url_mismatch": False},
        },
        {
            "label":    "LOW score but critical anomaly override",
            "score":    20.0,
            "indicators": {"url_mismatch": True, "domain_spoofing": True},
        },
        {
            "label":    "Maximum score",
            "score":    100.0,
            "indicators": {
                "url_mismatch":    True,
                "domain_spoofing": True,
                "logo_tampering":  True,
                "qr_overlay":      True,
            },
        },
    ]

    separator = "=" * 70

    for case in test_cases:
        result = engine.evaluate(
            score=case["score"],
            anomaly_indicators=case["indicators"],
        )
        print(f"\n{separator}")
        print(f"  CASE : {case['label']}")
        print(f"  Score: {case['score']}   →   {result.risk_level.value}")
        print(f"  Recommendation: {result.recommendation}")
        print("  Reasons:")
        for reason in result.reasons:
            print(f"    - {reason}")
        print("  Serialised (to_dict):")
        print(
            json.dumps(
                {k: v for k, v in result.to_dict().items()
                 if k != "decision_explanation"},
                indent=4,
            )
        )

    print(f"\n{separator}")
    print("  Demo complete.")
    print(separator)


if __name__ == "__main__":
    _demo()