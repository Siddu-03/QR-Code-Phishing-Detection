"""
risk_engine.py
==============
Week 3 – Member 2: Risk Assessment Engine
Project: Computer Vision-Based Graphic Tamper Detection for QR Code Phishing Prevention

Overview
--------
The Risk Assessment Engine is the central orchestrator of the risk evaluation
pipeline.  It accepts QR detection results produced by ``qr_detector.py``,
drives the :class:`~src.risk_assessment.scoring.ScoringEngine` and the
:class:`~src.risk_assessment.rule_engine.RuleEngine`, and returns a fully
populated :class:`~src.risk_assessment.risk_result.RiskResult` as the final,
authoritative pipeline output.

This module is intentionally **thin** — it contains no scoring logic and no
classification logic. Its sole responsibility is orchestration:

1. Validate inputs.
2. Extract relevant signals from the QR detection result.
3. Accept future tamper-analysis data through a clearly documented interface
   (currently no-ops; see :ref:`future-interface`).
4. Invoke :class:`ScoringEngine` → obtain :class:`ScoreBreakdown`.
5. Invoke :class:`RuleEngine` → obtain :class:`RuleEngineResult`.
6. Construct and return a :class:`RiskResult`.
7. Record wall-clock processing time.
8. Log the complete assessment process.
9. Handle all errors gracefully.

.. _future-interface:

Future interface — Tamper Analysis
-----------------------------------
Once ``src/tamper_analysis/tamper_result.py`` and companion computer-vision
modules (overlay detection, edge analysis, contour analysis, finder-pattern
analysis) are implemented, they should be integrated as follows:

1. **Add a keyword argument** ``tamper_result: TamperResult | None = None``
   to :meth:`RiskEngine.assess` (and its public convenience wrapper
   :func:`assess`).
2. **Populate** ``ScoringInputs.tamper_confidence``,
   ``ScoringInputs.overlay_detected``, ``ScoringInputs.overlay_confidence``,
   ``ScoringInputs.edge_inconsistency_score``,
   ``ScoringInputs.contour_mismatch_score``, and
   ``ScoringInputs.finder_pattern_damage_score`` from the ``TamperResult``
   object inside :meth:`RiskEngine._build_scoring_inputs`.
3. **Propagate** ``tamper_result.anomaly_count`` (or equivalent) to
   ``ScoringInputs.anomaly_count``.
4. No other files in the current codebase require modification at that point.

See the ``# FUTURE-TAMPER`` comment blocks throughout this file for the
exact extension points.

Pipeline position
-----------------
::

    qr_detector.py   ──► risk_engine.py ──► scoring.py   ──┐
    (future)                                 rule_engine.py ──┤──► RiskResult
    tamper_analysis/ ──►                                   ──┘

    risk_engine.py ──► FastAPI backend ──► Flutter mobile app
                   ──► Reporting module
                   ──► Research evaluation framework

Design principles
-----------------
*  **Orchestration only** — no scoring logic, no threshold logic.
*  **Fail-safe** — every public method catches all exceptions and returns a
   HIGH_RISK ``RiskResult`` rather than propagating a crash to callers.
*  **Configurable** — :class:`RiskEngineConfig` carries the single knob the
   orchestration layer needs; all scoring and rule weights live in their
   respective modules.
*  **Stateless sessions** — :class:`RiskEngine` holds no per-assessment state;
   it can be safely shared across threads once constructed.
*  **REST-ready** — :meth:`RiskEngine.assess` returns a :class:`RiskResult`
   whose :meth:`~RiskResult.to_dict` is directly serialisable by FastAPI.
*  **Research-ready** — structured log records at every pipeline stage; full
   score breakdown is embedded in ``RiskResult.metadata`` for evaluation
   scripts.

Compatibility
-------------
*  Python 3.11+.
*  Depends only on ``scoring.py``, ``rule_engine.py``, and ``risk_result.py``
   — all three are fully implemented as of Week 3.
*  Zero imports from ``tamper_analysis`` or any other unfinished module.
*  ``qr_detector.py`` output is consumed as a plain ``dict``; no import of
   ``qr_detector`` is required.

Usage
-----
Minimal (QR detection only)::

    from src.risk_assessment.risk_engine import RiskEngine

    engine = RiskEngine()
    result = engine.assess(detection_result=detect_qr("scan.png"))
    print(result.summary())

Custom configuration::

    from src.risk_assessment.risk_engine import RiskEngine, RiskEngineConfig
    from src.risk_assessment.scoring import ScoringConfig
    from src.risk_assessment.rule_engine import RuleEngineConfig

    engine = RiskEngine(
        config=RiskEngineConfig(
            scoring_config=ScoringConfig(overlay_weight=30.0),
            rule_config=RuleEngineConfig(safe_max=25, suspicious_max=55),
            engine_version="1.1.0",
        )
    )
    result = engine.assess(detection_result=detect_qr("scan.png"))

Module-level convenience function::

    from src.risk_assessment.risk_engine import assess

    result = assess(detection_result=detect_qr("scan.png"))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.risk_assessment.risk_result import RiskLevel, RiskResult
from src.risk_assessment.rule_engine import (
    RuleEngine,
    RuleEngineConfig,
    RuleEngineResult,
)
from src.risk_assessment.scoring import (
    ScoringConfig,
    ScoringEngine,
    ScoringInputs,
    ScoreBreakdown,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
_ENGINE_VERSION: str = "1.0.0"
_SCORE_TO_UNIT_DIVISOR: float = 100.0   # scoring.py → [0,100]; RiskResult → [0,1]


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class RiskEngineConfig:
    """Configuration for the Risk Assessment Engine orchestrator.

    The engine itself needs only two knobs: which ``ScoringConfig`` to pass
    to the :class:`ScoringEngine` and which ``RuleEngineConfig`` to pass to
    the :class:`RuleEngine`.  All other tunables live in those downstream
    configs.

    Parameters
    ----------
    scoring_config : ScoringConfig, optional
        Weight configuration forwarded to :class:`ScoringEngine`.
        Defaults to ``ScoringConfig()`` (standard weights).
    rule_config : RuleEngineConfig, optional
        Threshold configuration forwarded to :class:`RuleEngine`.
        Defaults to ``RuleEngineConfig()`` (standard thresholds:
        SAFE ≤ 30, SUSPICIOUS ≤ 60, HIGH_RISK > 60).
    engine_version : str, optional
        Semantic version tag recorded in every ``RiskResult.metadata``
        block for reproducibility.  Defaults to ``"1.0.0"``.
    include_score_breakdown_in_metadata : bool, optional
        When ``True`` (default), the full :class:`ScoreBreakdown` dictionary
        is serialised into ``RiskResult.metadata["score_breakdown"]``.
        Set to ``False`` in production environments where payload size
        matters; retain ``True`` for research and debugging.
    include_rule_detail_in_metadata : bool, optional
        When ``True`` (default), the :class:`RuleEngineResult` dictionary
        (minus ``decision_explanation``, which goes into its own key) is
        serialised into ``RiskResult.metadata["rule_detail"]``.
    """

    scoring_config:                      ScoringConfig    = field(
        default_factory=ScoringConfig
    )
    rule_config:                         RuleEngineConfig = field(
        default_factory=RuleEngineConfig
    )
    engine_version:                      str  = _ENGINE_VERSION
    include_score_breakdown_in_metadata: bool = True
    include_rule_detail_in_metadata:     bool = True


# ===========================================================================
# Risk Engine
# ===========================================================================

class RiskEngine:
    """Central orchestrator for the Risk Assessment pipeline.

    Instantiate once and call :meth:`assess` once per QR detection event.
    The engine is stateless between calls and thread-safe.

    Parameters
    ----------
    config : RiskEngineConfig, optional
        Engine configuration.  Defaults to ``RiskEngineConfig()`` which
        uses standard scoring and rule-engine defaults.

    Raises
    ------
    ValueError
        If ``config`` is not a :class:`RiskEngineConfig` instance.
    """

    def __init__(self, config: RiskEngineConfig | None = None) -> None:
        if config is None:
            config = RiskEngineConfig()
        if not isinstance(config, RiskEngineConfig):
            raise ValueError(
                f"config must be a RiskEngineConfig instance, "
                f"got {type(config).__name__!r}."
            )

        self._config         = config
        self._scoring_engine = ScoringEngine(config=config.scoring_config)
        self._rule_engine    = RuleEngine(config=config.rule_config)

        logger.info(
            "RiskEngine initialised — version=%s, "
            "rule_thresholds=(safe≤%.0f, suspicious≤%.0f), "
            "anomaly_override=%s",
            config.engine_version,
            config.rule_config.safe_max,
            config.rule_config.suspicious_max,
            config.rule_config.anomaly_override_enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> RiskEngineConfig:
        """Read-only access to the active engine configuration."""
        return self._config

    def assess(
        self,
        detection_result: dict[str, Any],
        # FUTURE-TAMPER: add tamper_result: TamperResult | None = None
        # when tamper_analysis is available (Week 3 completion).
        # No other signature change is needed at that point.
        image_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RiskResult:
        """Perform a complete risk assessment on a QR detection result.

        This is the primary public method.  It orchestrates the full
        pipeline: input validation → signal extraction → scoring →
        rule evaluation → :class:`RiskResult` construction.

        The method is **fail-safe**: all internal exceptions are caught,
        logged, and converted into a HIGH_RISK ``RiskResult`` so that the
        caller always receives a valid, serialisable result object.

        Parameters
        ----------
        detection_result : dict[str, Any]
            The ``DetectionResult`` dictionary returned by
            ``qr_detector.detect_qr()``.  Expected top-level keys:

            * ``"detected"``    – ``bool``
            * ``"count"``       – ``int``
            * ``"detector_used"`` – ``"opencv" | "pyzbar" | "none"``
            * ``"image_info"``  – ``{"width": int, "height": int}``
            * ``"detections"``  – ``list[dict]``; each item has:

              * ``"data"``          – decoded QR string
              * ``"confidence"``    – ``None`` (reserved; both backends
                                      return ``None`` in the current
                                      implementation)
              * ``"corner_points"`` – ``[[x,y], ...]``
              * ``"bbox_tuple"``    – ``[x, y, w, h]``
              * ``"bbox_dict"``     – ``{"x":…, "y":…, "w":…, "h":…}``

        image_id : str, optional
            An optional identifier for the source image (e.g. filename or
            database key).  Stored in ``RiskResult.metadata["image_id"]``
            for audit and research traceability.
        extra_metadata : dict[str, Any], optional
            Additional key-value pairs to merge into
            ``RiskResult.metadata``.  Caller-supplied keys must not clash
            with the engine's reserved keys:
            ``"engine_version"``, ``"image_id"``, ``"detector_used"``,
            ``"qr_count"``, ``"qr_data"``, ``"score_breakdown"``,
            ``"rule_detail"``, ``"decision_explanation"``,
            ``"scoring_explanation"``, ``"error"``.

        Returns
        -------
        RiskResult
            Immutable, frozen result object.  Always returned even when an
            internal error occurs (returned as HIGH_RISK with an
            ``"error"`` key in ``metadata``).

        Examples
        --------
        Typical usage::

            from src.qr_detector.qr_detector import detect_qr
            from src.risk_assessment.risk_engine import RiskEngine

            engine = RiskEngine()
            detection = detect_qr("path/to/image.png")
            result    = engine.assess(detection, image_id="img_001")
            print(result.summary())
            payload = result.to_dict()   # FastAPI response body

        No QR code found::

            detection = {
                "detected": False, "count": 0,
                "detector_used": "none",
                "image_info": {"width": 640, "height": 480},
                "detections": [],
            }
            result = engine.assess(detection)
            assert result.risk_level == RiskLevel.SAFE
        """
        t_start = time.perf_counter()

        logger.info(
            "RiskEngine.assess — START  image_id=%r",
            image_id or "<unset>",
        )

        try:
            return self._run_assessment(
                detection_result=detection_result,
                image_id=image_id,
                extra_metadata=extra_metadata or {},
                t_start=t_start,
            )

        except Exception as exc:  # noqa: BLE001
            # Fail-safe: any unexpected error produces a HIGH_RISK result
            # so callers always receive a valid RiskResult.
            elapsed_ms = (time.perf_counter() - t_start) * 1_000.0
            logger.exception(
                "RiskEngine.assess — UNHANDLED EXCEPTION after %.1f ms: %s",
                elapsed_ms,
                exc,
            )
            return self._make_error_result(
                reason=f"Internal risk assessment error: {exc}",
                processing_time_ms=elapsed_ms,
                image_id=image_id,
                extra_metadata=extra_metadata or {},
            )

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------

    def _run_assessment(
        self,
        detection_result: dict[str, Any],
        image_id: str | None,
        extra_metadata: dict[str, Any],
        t_start: float,
    ) -> RiskResult:
        """Internal pipeline — called by :meth:`assess`.

        Separated from the public method so that :meth:`assess` can wrap
        the entire call in a single top-level try/except without
        complicating the flow of this method.

        Parameters
        ----------
        detection_result : dict[str, Any]
            Raw QR detector output.
        image_id : str | None
            Optional source image identifier for metadata.
        extra_metadata : dict[str, Any]
            Caller-supplied annotations to merge into ``RiskResult.metadata``.
        t_start : float
            ``time.perf_counter()`` value captured at the start of
            :meth:`assess`.

        Returns
        -------
        RiskResult
            Fully populated result object.
        """
        # ── 1. Validate input ────────────────────────────────────────────────
        self._validate_detection_result(detection_result)

        # ── 2. Extract signals from QR detection result ──────────────────────
        qr_signals = self._extract_qr_signals(detection_result)

        logger.debug(
            "RiskEngine — QR signals extracted: detected=%s, count=%d, "
            "detector=%r, qr_data=%r",
            qr_signals["detected"],
            qr_signals["qr_count"],
            qr_signals["detector_used"],
            qr_signals["qr_data"],
        )

        # ── 3. Build ScoringInputs ───────────────────────────────────────────
        # FUTURE-TAMPER: pass tamper_result here when available.
        scoring_inputs = self._build_scoring_inputs(
            qr_signals=qr_signals,
            # tamper_result=tamper_result,  # FUTURE-TAMPER
        )

        # ── 4. Invoke ScoringEngine ──────────────────────────────────────────
        logger.debug("RiskEngine — invoking ScoringEngine …")
        score_breakdown: ScoreBreakdown = self._scoring_engine.compute_score(
            scoring_inputs
        )

        logger.info(
            "RiskEngine — ScoringEngine result: total_score=%.2f/100, "
            "applicable_factors=%d",
            score_breakdown.total_score,
            sum(1 for fs in score_breakdown.factor_scores if fs.applicable),
        )

        # ── 5. Build anomaly indicators for the RuleEngine ──────────────────
        # FUTURE-TAMPER: add overlay/edge/contour flags from tamper_result.
        anomaly_indicators = self._build_anomaly_indicators(
            qr_signals=qr_signals,
            score_breakdown=score_breakdown,
            # tamper_result=tamper_result,  # FUTURE-TAMPER
        )

        # ── 6. Invoke RuleEngine ─────────────────────────────────────────────
        logger.debug(
            "RiskEngine — invoking RuleEngine (score=%.2f, indicators=%s) …",
            score_breakdown.total_score,
            {k: v for k, v in anomaly_indicators.items() if v} or "none",
        )
        rule_result: RuleEngineResult = self._rule_engine.evaluate(
            score=score_breakdown.total_score,
            anomaly_indicators=anomaly_indicators,
        )

        logger.info(
            "RiskEngine — RuleEngine decision: level=%s, rules_applied=%s",
            rule_result.risk_level.value,
            rule_result.applied_rules,
        )

        # ── 7. Measure processing time ───────────────────────────────────────
        processing_time_ms = (time.perf_counter() - t_start) * 1_000.0

        # ── 8. Assemble metadata ─────────────────────────────────────────────
        metadata = self._build_metadata(
            qr_signals=qr_signals,
            score_breakdown=score_breakdown,
            rule_result=rule_result,
            image_id=image_id,
            extra_metadata=extra_metadata,
        )

        # ── 9. Map RiskLevel from rule_engine to risk_result ─────────────────
        # Both modules define a RiskLevel enum with identical values; map by
        # value string to remain decoupled from rule_engine's enum class.
        risk_level = RiskLevel(rule_result.risk_level.value)

        # ── 10. Construct RiskResult ─────────────────────────────────────────
        # score: scoring.py produces [0, 100]; RiskResult expects [0.0, 1.0].
        # confidence: derived from score_breakdown's applicable weight ratio.
        result = RiskResult(
            risk_level=risk_level,
            score=self._normalise_score(score_breakdown.total_score),
            confidence=self._derive_confidence(score_breakdown, rule_result),
            reasons=list(rule_result.reasons),
            recommendation=rule_result.recommendation,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(tz=timezone.utc),
            metadata=metadata,
        )

        logger.info(
            "RiskEngine.assess — COMPLETE  level=%s  score=%.4f  "
            "confidence=%.2f  elapsed=%.1f ms",
            result.risk_level.value,
            result.score,
            result.confidence,
            processing_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_detection_result(detection_result: Any) -> None:
        """Validate the structure of a QR detector output dictionary.

        Parameters
        ----------
        detection_result : Any
            The value passed by the caller.

        Raises
        ------
        TypeError
            If ``detection_result`` is not a ``dict``.
        ValueError
            If any required key is missing or has an unexpected type.
        """
        if not isinstance(detection_result, dict):
            raise TypeError(
                f"detection_result must be a dict, "
                f"got {type(detection_result).__name__!r}."
            )

        required_keys = {"detected", "count", "detector_used",
                         "image_info", "detections"}
        missing = required_keys - detection_result.keys()
        if missing:
            raise ValueError(
                f"detection_result is missing required keys: {sorted(missing)}."
            )

        if not isinstance(detection_result["detected"], bool):
            raise ValueError(
                f"detection_result['detected'] must be bool, "
                f"got {type(detection_result['detected']).__name__!r}."
            )
        if not isinstance(detection_result["count"], int):
            raise ValueError(
                f"detection_result['count'] must be int, "
                f"got {type(detection_result['count']).__name__!r}."
            )
        if not isinstance(detection_result["detections"], list):
            raise ValueError(
                f"detection_result['detections'] must be list, "
                f"got {type(detection_result['detections']).__name__!r}."
            )
        if not isinstance(detection_result["image_info"], dict):
            raise ValueError(
                f"detection_result['image_info'] must be dict, "
                f"got {type(detection_result['image_info']).__name__!r}."
            )

        logger.debug("RiskEngine — detection_result validation passed.")

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_qr_signals(detection_result: dict[str, Any]) -> dict[str, Any]:
        """Extract and normalise relevant signals from a QR detection result.

        The primary QR detection is the first entry in ``detections``; if
        multiple QR codes were found, each additional code is treated as a
        mild anomaly signal (unexpected QR multiplicity).

        Parameters
        ----------
        detection_result : dict[str, Any]
            Validated ``DetectionResult`` from ``qr_detector.detect_qr()``.

        Returns
        -------
        dict[str, Any]
            Flat, normalised signal dictionary consumed by
            :meth:`_build_scoring_inputs` and
            :meth:`_build_anomaly_indicators`. Keys:

            * ``"detected"``        – bool
            * ``"qr_count"``        – int
            * ``"detector_used"``   – str
            * ``"qr_data"``         – str | None (first detection's payload)
            * ``"qr_confidence"``   – None (always; both backends return None)
            * ``"multiple_qr"``     – bool (True when count > 1)
            * ``"image_width"``     – int
            * ``"image_height"``    – int
        """
        detections    = detection_result.get("detections", [])
        first         = detections[0] if detections else {}
        image_info    = detection_result.get("image_info", {})
        qr_count: int = detection_result.get("count", 0)

        return {
            "detected":      detection_result.get("detected", False),
            "qr_count":      qr_count,
            "detector_used": detection_result.get("detector_used", "none"),
            "qr_data":       first.get("data") or None,
            "qr_confidence": first.get("confidence"),   # currently always None
            "multiple_qr":   qr_count > 1,
            "image_width":   image_info.get("width", 0),
            "image_height":  image_info.get("height", 0),
        }

    # ------------------------------------------------------------------
    # ScoringInputs construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_scoring_inputs(
        qr_signals: dict[str, Any],
        # FUTURE-TAMPER: tamper_result: TamperResult | None = None,
    ) -> ScoringInputs:
        """Build a :class:`ScoringInputs` instance from available signals.

        Currently populates only those fields derivable from QR detection.
        The tamper-analysis fields (``overlay_detected``,
        ``overlay_confidence``, ``edge_inconsistency_score``,
        ``contour_mismatch_score``, ``finder_pattern_damage_score``,
        ``tamper_confidence``) are left at their safe defaults (``0.0`` /
        ``False`` / ``None``) until tamper analysis is integrated.

        .. note::
            ``detection_confidence`` is always ``None`` in the current
            ``qr_detector.py`` implementation because neither the OpenCV
            nor pyzbar backend reports a confidence value.  The field is
            passed through as-is so that the ``DetectionConfidenceFactor``
            in ``scoring.py`` correctly marks itself as inapplicable.

        Parameters
        ----------
        qr_signals : dict[str, Any]
            Output of :meth:`_extract_qr_signals`.

        Returns
        -------
        ScoringInputs
            Populated inputs ready for :class:`ScoringEngine`.

        .. rubric:: FUTURE-TAMPER extension

        When ``tamper_result`` is added as a parameter, populate::

            overlay_detected           = tamper_result.overlay_detected,
            overlay_confidence         = tamper_result.overlay_confidence,
            edge_inconsistency_score   = tamper_result.edge_inconsistency_score,
            contour_mismatch_score     = tamper_result.contour_mismatch_score,
            finder_pattern_damage_score= tamper_result.finder_pattern_damage_score,
            tamper_confidence          = tamper_result.tamper_confidence,
            anomaly_count              = tamper_result.anomaly_count,
        """
        # Anomaly count: each additional QR code beyond the first is one
        # discrete anomaly (unexpected QR multiplicity).
        anomaly_count = max(0, qr_signals["qr_count"] - 1)

        return ScoringInputs(
            # detection_confidence: None because qr_detector.py never
            # populates this field (both backends return confidence=None).
            detection_confidence=qr_signals["qr_confidence"],

            # Tamper-analysis fields — all at safe defaults until Week 3
            # Tamper Analysis module is integrated.  See FUTURE-TAMPER notes.
            overlay_detected=False,            # FUTURE-TAMPER
            overlay_confidence=0.0,            # FUTURE-TAMPER
            edge_inconsistency_score=0.0,      # FUTURE-TAMPER
            contour_mismatch_score=0.0,        # FUTURE-TAMPER
            finder_pattern_damage_score=0.0,   # FUTURE-TAMPER
            tamper_confidence=None,            # FUTURE-TAMPER
            ai_confidence=None,                # FUTURE-TAMPER (ML classifier)

            # Corroborating anomaly count — non-zero when multiple QR codes
            # are present in one image (unusual for a legitimate QR payload).
            anomaly_count=anomaly_count,

            # Pass-through metadata for audit traceability.
            metadata={
                "detector_used": qr_signals["detector_used"],
                "qr_count":      qr_signals["qr_count"],
            },
        )

    # ------------------------------------------------------------------
    # Anomaly indicators for RuleEngine
    # ------------------------------------------------------------------

    @staticmethod
    def _build_anomaly_indicators(
        qr_signals: dict[str, Any],
        score_breakdown: ScoreBreakdown,
        # FUTURE-TAMPER: tamper_result: TamperResult | None = None,
    ) -> dict[str, bool]:
        """Build the anomaly indicator dictionary consumed by :class:`RuleEngine`.

        Each key maps to a boolean that signals whether a specific type of
        tampering or phishing indicator is active.  Active indicators may
        trigger anomaly-override escalation in the rule engine regardless of
        the numeric score.

        The ``critical_anomaly_keys`` default in :class:`RuleEngineConfig`
        is ``["url_mismatch", "domain_spoofing", "qr_overlay"]``.  The keys
        produced here are designed to align with those defaults.

        Parameters
        ----------
        qr_signals : dict[str, Any]
            Output of :meth:`_extract_qr_signals`.
        score_breakdown : ScoreBreakdown
            Output of :class:`ScoringEngine`, used to check whether
            specific scoring factors were active.

        Returns
        -------
        dict[str, bool]
            Indicator mapping passed verbatim to
            :meth:`RuleEngine.evaluate(anomaly_indicators=…)`.

        .. rubric:: FUTURE-TAMPER extension

        When ``tamper_result`` is available, populate::

            "qr_overlay":         tamper_result.overlay_detected,
            "edge_inconsistency": tamper_result.edge_inconsistency_score > 0.5,
            "contour_mismatch":   tamper_result.contour_mismatch_score > 0.5,
            "finder_damage":      tamper_result.finder_pattern_damage_score > 0.5,
        """
        return {
            # True when more than one QR code appears in a single image — an
            # unusual pattern that may indicate an overlay attack.
            "multiple_qr_codes":  qr_signals["multiple_qr"],

            # Tamper-analysis indicators — False until tamper_result is wired in.
            "qr_overlay":         False,   # FUTURE-TAMPER
            "edge_inconsistency": False,   # FUTURE-TAMPER
            "contour_mismatch":   False,   # FUTURE-TAMPER
            "finder_damage":      False,   # FUTURE-TAMPER

            # URL / domain heuristics — False until a URL-analysis module
            # is integrated (future Week 4+).
            "url_mismatch":       False,   # FUTURE: url_analyzer module
            "domain_spoofing":    False,   # FUTURE: url_analyzer module
        }

    # ------------------------------------------------------------------
    # Score / confidence conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_score(total_score: float) -> float:
        """Convert a ``[0, 100]`` score to the ``[0.0, 1.0]`` range.

        ``scoring.py`` produces scores on ``[0, 100]``.
        ``RiskResult.score`` requires ``[0.0, 1.0]``.
        This helper performs the division and clamps for floating-point safety.

        Parameters
        ----------
        total_score : float
            Score from :class:`ScoreBreakdown` in the range ``[0, 100]``.

        Returns
        -------
        float
            Score clamped to ``[0.0, 1.0]``.
        """
        return max(0.0, min(1.0, total_score / _SCORE_TO_UNIT_DIVISOR))

    @staticmethod
    def _derive_confidence(
        score_breakdown: ScoreBreakdown,
        rule_result: RuleEngineResult,
    ) -> float:
        """Derive a pipeline confidence value for ``RiskResult.confidence``.

        Confidence reflects how many applicable scoring factors contributed
        to the final assessment.  A result derived from many independent
        signals is more trustworthy than one derived from a single factor.

        The formula is::

            confidence = applicable_weight_total
                         / total_possible_weight_from_config

        where ``total_possible_weight_from_config`` is the sum of all
        weights in ``ScoringConfig`` (the denominator when every factor is
        applicable).  This value is extracted from ``config_snapshot``.

        If no weight data is available, confidence defaults to ``0.5``
        (neutral / unknown).

        Parameters
        ----------
        score_breakdown : ScoreBreakdown
            Scoring output with ``applicable_weight_total`` and
            ``config_snapshot``.
        rule_result : RuleEngineResult
            Rule output (currently unused; reserved for future weighted-rule
            confidence signals).

        Returns
        -------
        float
            Confidence value clamped to ``[0.0, 1.0]``.
        """
        snapshot = score_breakdown.config_snapshot
        if not snapshot:
            logger.debug(
                "RiskEngine._derive_confidence — no config snapshot; "
                "defaulting to 0.5."
            )
            return 0.5

        # Sum all per-factor weights from the snapshot (excluding
        # structural keys that are not weight values).
        weight_keys = {
            "detection_confidence_weight",
            "overlay_weight",
            "edge_inconsistency_weight",
            "contour_mismatch_weight",
            "finder_pattern_damage_weight",
            "anomaly_count_weight",
            "tamper_confidence_weight",
            "ai_confidence_weight",
        }
        total_possible: float = sum(
            float(snapshot[k])
            for k in weight_keys
            if k in snapshot and isinstance(snapshot[k], (int, float))
        )

        if total_possible <= 0.0:
            logger.debug(
                "RiskEngine._derive_confidence — total_possible=0; "
                "defaulting to 0.5."
            )
            return 0.5

        raw_confidence = score_breakdown.applicable_weight_total / total_possible
        return max(0.0, min(1.0, raw_confidence))

    # ------------------------------------------------------------------
    # Metadata construction
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        qr_signals: dict[str, Any],
        score_breakdown: ScoreBreakdown,
        rule_result: RuleEngineResult,
        image_id: str | None,
        extra_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the ``RiskResult.metadata`` dictionary.

        Parameters
        ----------
        qr_signals : dict[str, Any]
            Extracted QR signals.
        score_breakdown : ScoreBreakdown
            Scoring engine output.
        rule_result : RuleEngineResult
            Rule engine output.
        image_id : str | None
            Optional source image identifier.
        extra_metadata : dict[str, Any]
            Caller-supplied annotations to merge in last (may not overwrite
            engine-reserved keys).

        Returns
        -------
        dict[str, Any]
            Flat, JSON-serialisable metadata dictionary.
        """
        metadata: dict[str, Any] = {
            # Engine provenance
            "engine_version": self._config.engine_version,

            # Source image identity (if provided)
            "image_id": image_id,

            # QR detection summary
            "detector_used": qr_signals["detector_used"],
            "qr_count":      qr_signals["qr_count"],
            "qr_data":       qr_signals["qr_data"],

            # Scoring narrative (always included — short string)
            "scoring_explanation": score_breakdown.explanation,

            # Rule narrative (always included — short string)
            "decision_explanation": rule_result.decision_explanation,
        }

        # Optional verbose payloads (controlled by config flags)
        if self._config.include_score_breakdown_in_metadata:
            metadata["score_breakdown"] = score_breakdown.to_dict()

        if self._config.include_rule_detail_in_metadata:
            metadata["rule_detail"] = rule_result.to_dict()

        # Merge caller-supplied extras last; reserved keys are not overwritten.
        reserved = {
            "engine_version", "image_id", "detector_used", "qr_count",
            "qr_data", "score_breakdown", "rule_detail",
            "decision_explanation", "scoring_explanation", "error",
        }
        for key, value in extra_metadata.items():
            if key not in reserved:
                metadata[key] = value
            else:
                logger.warning(
                    "RiskEngine — extra_metadata key %r is reserved "
                    "and was not merged.", key
                )

        return metadata

    # ------------------------------------------------------------------
    # Error result factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_error_result(
        reason: str,
        processing_time_ms: float,
        image_id: str | None,
        extra_metadata: dict[str, Any],
    ) -> RiskResult:
        """Construct a HIGH_RISK :class:`RiskResult` to represent a
        pipeline failure.

        Used by :meth:`assess` as the fail-safe return value when an
        unexpected exception propagates all the way up to the top-level
        try/except.  Downstream consumers (FastAPI, Flutter, reporting)
        will see HIGH_RISK and find the exception details in
        ``metadata["error"]``.

        Parameters
        ----------
        reason : str
            Human-readable description of the failure.
        processing_time_ms : float
            Elapsed wall-clock time before the failure.
        image_id : str | None
            Optional source image identifier.
        extra_metadata : dict[str, Any]
            Caller-supplied extra metadata.

        Returns
        -------
        RiskResult
            HIGH_RISK result with ``confidence=0.0`` to signal low
            certainty, and ``metadata["error"]`` populated with *reason*.
        """
        metadata: dict[str, Any] = {
            "engine_version": _ENGINE_VERSION,
            "image_id":       image_id,
            "error":          reason,
            **{k: v for k, v in extra_metadata.items()
               if k not in {"engine_version", "image_id", "error"}},
        }

        return RiskResult(
            risk_level=RiskLevel.HIGH_RISK,
            score=1.0,
            confidence=0.0,
            reasons=[
                "Risk assessment pipeline encountered an internal error.",
                "The QR code has been flagged HIGH_RISK as a precaution.",
                reason,
            ],
            recommendation=(
                "Do not interact with this QR code.  "
                "An internal assessment error occurred; please retry or "
                "report the issue to your security team."
            ),
            processing_time_ms=max(0.0, processing_time_ms),
            timestamp=datetime.now(tz=timezone.utc),
            metadata=metadata,
        )


# ===========================================================================
# Module-level convenience API
# ===========================================================================

def assess(
    detection_result: dict[str, Any],
    *,
    config: RiskEngineConfig | None = None,
    image_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> RiskResult:
    """Assess a QR detection result using a default :class:`RiskEngine`.

    Convenience wrapper for call sites that do not need to retain an
    engine instance between calls (e.g. single-shot scripts, unit tests,
    and ``main.py`` integration once the risk pipeline is wired in).

    For production services with repeated assessments, prefer instantiating
    :class:`RiskEngine` once and calling :meth:`~RiskEngine.assess` directly
    to avoid the overhead of re-creating ``ScoringEngine`` and
    ``RuleEngine`` on every call.

    Parameters
    ----------
    detection_result : dict[str, Any]
        Output from ``qr_detector.detect_qr()``.
    config : RiskEngineConfig, optional
        Engine configuration.  Defaults to ``RiskEngineConfig()``.
    image_id : str, optional
        Source image identifier for audit traceability.
    extra_metadata : dict[str, Any], optional
        Additional annotations merged into ``RiskResult.metadata``.

    Returns
    -------
    RiskResult
        Immutable result object.  Always returned; never raises.

    Example
    -------
    ::

        from src.risk_assessment.risk_engine import assess
        from src.qr_detector.qr_detector import detect_qr

        result = assess(detect_qr("sample_qr.png"), image_id="sample_qr.png")
        print(result.summary())
        print(result.to_dict())
    """
    engine = RiskEngine(config=config)
    return engine.assess(
        detection_result=detection_result,
        image_id=image_id,
        extra_metadata=extra_metadata,
    )


def create_default_engine() -> RiskEngine:
    """Instantiate a :class:`RiskEngine` with default configuration.

    Convenience factory for ``main.py`` and testing harnesses that do not
    need custom weight or threshold tuning.

    Returns
    -------
    RiskEngine
        Engine with standard scoring and rule-engine defaults.
    """
    return RiskEngine(config=RiskEngineConfig())


# ===========================================================================
# Demo / development entry-point
# ===========================================================================

def _demo() -> None:  # pragma: no cover
    """Demonstrate the Risk Engine with representative detection results.

    Run directly::

        python src/risk_assessment/risk_engine.py
    """
    import json

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    engine = create_default_engine()

    _make_detection = lambda detected, data="https://example.com", count=1: {
        "detected":      detected,
        "count":         count if detected else 0,
        "detector_used": "opencv" if detected else "none",
        "image_info":    {"width": 1920, "height": 1080},
        "detections":    [
            {
                "data":          data,
                "confidence":    None,
                "corner_points": [[100, 100], [200, 100], [200, 200], [100, 200]],
                "bbox_tuple":    [100, 100, 100, 100],
                "bbox_dict":     {"x": 100, "y": 100, "w": 100, "h": 100},
            }
        ] * (count if detected else 0),
    }

    test_cases: list[dict[str, Any]] = [
        {
            "label":  "No QR code detected",
            "input":  _make_detection(False),
        },
        {
            "label":  "Single clean QR code (typical safe scan)",
            "input":  _make_detection(True, data="https://legitimate-bank.com/pay"),
        },
        {
            "label":  "Multiple QR codes in one image (anomaly)",
            "input":  _make_detection(True, data="https://suspect-site.ru/qr", count=3),
        },
        {
            "label":  "Malformed detection_result (missing 'detected')",
            "input":  {"count": 1, "detector_used": "opencv",
                       "image_info": {}, "detections": []},
        },
    ]

    separator = "=" * 70

    for case in test_cases:
        print(f"\n{separator}")
        print(f"  CASE : {case['label']}")
        result = engine.assess(
            detection_result=case["input"],
            image_id=f"demo_{case['label'][:20].replace(' ', '_')}",
        )
        print(result.summary())
        compact = {
            "risk_level":        result.risk_level.value,
            "score":             round(result.score, 4),
            "confidence":        round(result.confidence, 4),
            "processing_ms":     round(result.processing_time_ms, 2),
            "reasons_count":     len(result.reasons),
            "engine_version":    result.metadata.get("engine_version"),
        }
        print(f"\n  Compact payload: {json.dumps(compact, indent=4)}")

    print(f"\n{separator}")
    print("  Demo complete.")
    print(separator)


if __name__ == "__main__":
    _demo()