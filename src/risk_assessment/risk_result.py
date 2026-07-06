"""
risk_result.py
==============
Canonical output model for the Risk Assessment module.
------------------------------------------------------
Week 3 – Member 2: Risk Assessment
Project: Computer Vision-Based Graphic Tamper Detection for QR Code Phishing Prevention

This module defines the immutable :class:`RiskResult` dataclass, which is the
single authoritative output contract of the Risk Assessment pipeline.  Every
consumer — Reporting module, Integration pipeline, FastAPI backend, Flutter
mobile application, and research evaluation framework — must rely solely on
this contract.

Design principles
-----------------
*  **Immutability by default.**  The dataclass is frozen (``frozen=True``);
   callers receive a value object that cannot be accidentally mutated in place.
*  **Self-contained.**  No imports from other project modules; ``RiskResult``
   can be instantiated and serialised anywhere in the stack.
*  **REST-ready.**  :meth:`RiskResult.to_dict` produces a JSON-serialisable
   dictionary whose structure maps 1-to-1 to the REST response schema, with
   ISO 8601 timestamps and plain Python primitives throughout.
*  **Research-ready.**  All floating-point fields carry explicit semantics in
   their docstrings so that evaluation scripts can consume them without
   additional documentation.
*  **Extensible.**  The ``metadata`` field is an open ``dict[str, Any]`` that
   downstream modules (scoring.py, rule_engine.py, FastAPI) can populate with
   version tags, model identifiers, per-rule breakdowns, or A/B test flags
   without breaking the contract.

Compatibility
-------------
Designed to complement ``TamperResult`` (``src/tamper_analysis/tamper_result.py``).
The ``metadata`` field intentionally mirrors the open-extension slot in that
class.  Both result types share the same ``processing_time_ms`` / ``timestamp``
convention so that the integration layer can merge them without impedance.

Typical usage
-------------
::

    from src.risk_assessment.risk_result import RiskLevel, RiskResult
    from datetime import datetime, timezone

    result = RiskResult(
        risk_level=RiskLevel.HIGH_RISK,
        score=0.87,
        confidence=0.91,
        reasons=[
            "URL domain registered within the last 7 days.",
            "QR code pixel density anomaly detected (tamper score: 0.74).",
        ],
        recommendation="Do not scan this QR code.  Report it to your IT department.",
        processing_time_ms=43.2,
        timestamp=datetime.now(tz=timezone.utc),
        metadata={"rule_engine_version": "1.0.0", "tamper_score": 0.74},
    )

    print(result.summary())
    payload = result.to_dict()   # safe to pass to json.dumps() or FastAPI response
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel used by from_dict() to distinguish a missing key from None.
# ---------------------------------------------------------------------------
_MISSING = object()


# ===========================================================================
# RiskLevel Enum
# ===========================================================================

class RiskLevel(str, Enum):
    """Ordinal classification of the threat posed by a scanned QR code.

    The enum inherits from :class:`str` so that instances serialise naturally
    to JSON strings (e.g. ``"HIGH_RISK"``) without a custom encoder.  Ordinal
    ordering is exposed via :attr:`RiskLevel.severity` to allow comparisons
    such as ``result.risk_level.severity >= RiskLevel.SUSPICIOUS.severity``.

    Members
    -------
    SAFE
        No tamper signals detected.  The QR code appears authentic and the
        encoded URL / payload has not triggered any heuristic rules.
    SUSPICIOUS
        One or more weak tamper or URL signals detected.  Manual review is
        advised before interacting with the encoded content.
    HIGH_RISK
        Strong tamper evidence or confirmed malicious-URL patterns detected.
        Interaction is strongly discouraged.
    """

    SAFE       = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK  = "HIGH_RISK"

    # ------------------------------------------------------------------
    # Severity as an integer so callers can do arithmetic comparisons.
    # ------------------------------------------------------------------

    @property
    def severity(self) -> int:
        """Return an integer severity index (SAFE=0, SUSPICIOUS=1, HIGH_RISK=2).

        Example
        -------
        ::

            if result.risk_level.severity >= RiskLevel.SUSPICIOUS.severity:
                alert_user()
        """
        _order = {
            RiskLevel.SAFE:       0,
            RiskLevel.SUSPICIOUS: 1,
            RiskLevel.HIGH_RISK:  2,
        }
        return _order[self]

    @property
    def display_label(self) -> str:
        """Return a human-readable label suitable for UI rendering.

        Example
        -------
        ::

            badge_text = result.risk_level.display_label   # "⚠ Suspicious"
        """
        _labels = {
            RiskLevel.SAFE:       "✅ Safe",
            RiskLevel.SUSPICIOUS: "⚠ Suspicious",
            RiskLevel.HIGH_RISK:  "🔴 High Risk",
        }
        return _labels[self]


# ===========================================================================
# RiskResult dataclass
# ===========================================================================

@dataclass(frozen=True, order=False)
class RiskResult:
    """Immutable output record produced by the Risk Assessment module.

    This is the canonical data contract consumed by every downstream component:

    * **Reporting module** – uses :meth:`to_dict` or :meth:`summary` to build
      human-readable and machine-readable reports.
    * **Integration pipeline** (``main.py``) – receives ``RiskResult`` as the
      final stage output; merges it with ``TamperResult`` for the composite
      report.
    * **FastAPI backend** – calls :meth:`to_dict` and returns the resulting
      dictionary as a JSON response body; no custom encoder required.
    * **Flutter mobile app** – consumes the REST JSON produced by FastAPI;
      the ``risk_level`` string maps to a Dart enum and ``timestamp`` to
      ``DateTime.parse``.
    * **Research evaluation framework** – iterates over ``RiskResult`` objects
      loaded via :meth:`from_dict`; uses ``score``, ``confidence``, and
      ``risk_level`` to compute precision / recall / F1 metrics.

    Attributes
    ----------
    risk_level : RiskLevel
        Ordinal threat classification (SAFE, SUSPICIOUS, HIGH_RISK).
    score : float
        Composite risk score in the range ``[0.0, 1.0]``.  Higher values
        indicate greater risk.  Produced by ``scoring.py``.
    confidence : float
        Model / rule confidence in the ``score`` estimate, in ``[0.0, 1.0]``.
        Reflects the agreement and strength of the signals that produced the
        score.  Not the same as ``score``; a high-confidence SAFE result
        (score=0.05, confidence=0.98) is categorically different from a
        low-confidence SUSPICIOUS result (score=0.55, confidence=0.30).
    reasons : tuple[str, ...]
        Ordered, human-readable explanations for the assigned ``risk_level``.
        Each string is one discrete signal (e.g. ``"Tamper score exceeds 0.70
        threshold."``).  Stored as a tuple to preserve immutability; serialised
        as a JSON array.
    recommendation : str
        A single, actionable instruction for the end-user or integrating system
        (e.g. ``"Do not scan this QR code."``).  Produced by ``rule_engine.py``
        based on the ``risk_level``.
    processing_time_ms : float
        Wall-clock time spent inside the Risk Assessment module, in
        milliseconds.  Excludes QR detection and tamper analysis time.
    timestamp : datetime
        UTC datetime at which this ``RiskResult`` was produced.  Always stored
        with ``tzinfo=timezone.utc``; serialised as an ISO 8601 string.
    metadata : dict[str, Any]
        Open extension slot for pipeline-specific annotations.  Examples::

            {
                "rule_engine_version": "1.0.0",
                "tamper_score":        0.74,
                "qr_data":             "https://example.com/path?ref=abc",
                "image_id":            "img_20240601_001",
                "model_id":            "risk-v2",
                "ab_variant":          "B",
            }

        Keys are strings; values must be JSON-serialisable for REST transport.
        Downstream modules should namespace their keys to avoid collisions
        (e.g. ``"tamper.score"`` rather than plain ``"score"``).

    Notes
    -----
    *  The dataclass is ``frozen=True``: attributes cannot be reassigned after
       construction.  Use :func:`dataclasses.replace` to derive a modified copy.
    *  ``reasons`` is stored as a ``tuple`` internally (immutable) but accepted
       as any ``Sequence[str]`` in the constructor and converted on init.
    *  ``metadata`` is the only mutable Python object inside the frozen
       dataclass — the dict itself is not copied on construction.  Callers
       should not mutate the dict after passing it in.
    """

    # ── Core classification fields ──────────────────────────────────────────
    risk_level:          RiskLevel
    score:               float
    confidence:          float
    reasons:             tuple[str, ...]
    recommendation:      str

    # ── Operational / audit fields ──────────────────────────────────────────
    processing_time_ms:  float
    timestamp:           datetime

    # ── Open extension slot ─────────────────────────────────────────────────
    metadata:            dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Post-init validation and normalisation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate field values and normalise mutable containers.

        Raises
        ------
        TypeError
            If any field has an incorrect type.
        ValueError
            If ``score`` or ``confidence`` is outside ``[0.0, 1.0]``, or if
            ``processing_time_ms`` is negative.
        """
        # --- Type checks ---------------------------------------------------
        if not isinstance(self.risk_level, RiskLevel):
            raise TypeError(
                f"risk_level must be a RiskLevel enum member, "
                f"got {type(self.risk_level).__name__!r}."
            )
        if not isinstance(self.score, (int, float)):
            raise TypeError(
                f"score must be a float, got {type(self.score).__name__!r}."
            )
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(
                f"confidence must be a float, got {type(self.confidence).__name__!r}."
            )
        if not isinstance(self.reasons, (list, tuple)):
            raise TypeError(
                f"reasons must be a list or tuple, got {type(self.reasons).__name__!r}."
            )
        if not isinstance(self.recommendation, str):
            raise TypeError(
                f"recommendation must be a str, got {type(self.recommendation).__name__!r}."
            )
        if not isinstance(self.processing_time_ms, (int, float)):
            raise TypeError(
                f"processing_time_ms must be a float, "
                f"got {type(self.processing_time_ms).__name__!r}."
            )
        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                f"timestamp must be a datetime, got {type(self.timestamp).__name__!r}."
            )
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"metadata must be a dict, got {type(self.metadata).__name__!r}."
            )

        # --- Range checks --------------------------------------------------
        if not (0.0 <= float(self.score) <= 1.0):
            raise ValueError(
                f"score must be in [0.0, 1.0], got {self.score!r}."
            )
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence!r}."
            )
        if float(self.processing_time_ms) < 0.0:
            raise ValueError(
                f"processing_time_ms must be non-negative, "
                f"got {self.processing_time_ms!r}."
            )

        # --- Normalise reasons to an immutable tuple -----------------------
        # frozen=True prevents direct assignment; use object.__setattr__.
        object.__setattr__(self, "reasons", tuple(self.reasons))

        # --- Normalise score and confidence to plain float -----------------
        object.__setattr__(self, "score",      float(self.score))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "processing_time_ms", float(self.processing_time_ms))

        # --- Ensure timestamp is timezone-aware (UTC) ----------------------
        if self.timestamp.tzinfo is None:
            # Treat naïve datetime as UTC and attach the timezone.
            object.__setattr__(
                self,
                "timestamp",
                self.timestamp.replace(tzinfo=timezone.utc),
            )
            logger.debug(
                "RiskResult: naïve timestamp converted to UTC: %s",
                self.timestamp.isoformat(),
            )

    # ==================================================================
    # Serialisation helpers
    # ==================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialise this result to a JSON-compatible dictionary.

        All values are plain Python primitives (``str``, ``float``, ``int``,
        ``bool``, ``list``, ``dict``, ``None``).  The returned dict can be
        passed directly to :func:`json.dumps` or returned as a FastAPI
        ``JSONResponse`` body without a custom encoder.

        Key mapping
        -----------
        ========================  ============================================
        Python attribute          JSON key / value type
        ========================  ============================================
        ``risk_level``            ``"risk_level"`` → ``str`` (enum value)
        ``score``                 ``"score"`` → ``float``
        ``confidence``            ``"confidence"`` → ``float``
        ``reasons``               ``"reasons"`` → ``list[str]``
        ``recommendation``        ``"recommendation"`` → ``str``
        ``processing_time_ms``    ``"processing_time_ms"`` → ``float``
        ``timestamp``             ``"timestamp"`` → ISO 8601 ``str``
        ``metadata``              ``"metadata"`` → ``dict[str, Any]``
        ``risk_level_severity``   ``"risk_level_severity"`` → ``int`` (0-2)
        ``risk_level_label``      ``"risk_level_label"`` → ``str``
        ========================  ============================================

        The two computed fields (``risk_level_severity``, ``risk_level_label``)
        are included so that REST clients do not need to re-implement enum
        ordinal logic.

        Returns
        -------
        dict[str, Any]
            A flat, JSON-serialisable dictionary.

        Example
        -------
        ::

            payload = result.to_dict()
            response_body = json.dumps(payload, indent=2)
        """
        return {
            "risk_level":           self.risk_level.value,
            "risk_level_severity":  self.risk_level.severity,
            "risk_level_label":     self.risk_level.display_label,
            "score":                round(self.score, 6),
            "confidence":           round(self.confidence, 6),
            "reasons":              list(self.reasons),
            "recommendation":       self.recommendation,
            "processing_time_ms":   round(self.processing_time_ms, 3),
            "timestamp":            self.timestamp.isoformat(),
            "metadata":             dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialise this result directly to a JSON string.

        Thin convenience wrapper around :meth:`to_dict` +
        :func:`json.dumps` for call sites (CLI tools, log sinks, message
        queues) that want a JSON string rather than a dict — avoiding the
        easy-to-forget two-step ``json.dumps(result.to_dict())`` at every
        call site.

        Parameters
        ----------
        indent : int, optional
            Passed through to :func:`json.dumps`. ``None`` (default)
            produces compact single-line JSON, suitable for log lines and
            message queues; use ``2`` for pretty-printed output.

        Returns
        -------
        str
            JSON-encoded string equivalent to ``json.dumps(self.to_dict())``.

        Example
        -------
        ::

            log.info("risk_result=%s", result.to_json())
            with open("result.json", "w") as fh:
                fh.write(result.to_json(indent=2))
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskResult":
        """Deserialise a ``RiskResult`` from a plain dictionary.

        Accepts the exact format produced by :meth:`to_dict`, making round-trip
        serialisation lossless (modulo floating-point rounding in
        ``processing_time_ms``).  Also accepts raw pipeline dictionaries that
        omit the computed ``risk_level_severity`` / ``risk_level_label`` keys.

        The ``timestamp`` value may be:

        * an ISO 8601 string (e.g. ``"2024-06-01T12:34:56.789+00:00"``),
        * a :class:`datetime` object (passed through as-is), or
        * ``None`` / missing key (defaults to the current UTC time with a
          warning).

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary conforming to the :meth:`to_dict` schema.

        Returns
        -------
        RiskResult
            A fully validated, immutable ``RiskResult`` instance.

        Raises
        ------
        KeyError
            If a required key is absent from *data*.
        ValueError
            If ``risk_level`` is not a valid :class:`RiskLevel` member, or if
            ``score`` / ``confidence`` are out of range.
        TypeError
            If any field value has an incompatible type.

        Example
        -------
        ::

            with open("risk_output.json") as fh:
                raw = json.load(fh)
            result = RiskResult.from_dict(raw)
        """
        # --- risk_level -------------------------------------------------------
        risk_level_raw = data["risk_level"]
        try:
            risk_level = RiskLevel(risk_level_raw)
        except ValueError:
            valid = [e.value for e in RiskLevel]
            raise ValueError(
                f"Invalid risk_level {risk_level_raw!r}.  "
                f"Must be one of: {valid}."
            )

        # --- timestamp --------------------------------------------------------
        ts_raw = data.get("timestamp", _MISSING)
        if ts_raw is _MISSING or ts_raw is None:
            timestamp = datetime.now(tz=timezone.utc)
            logger.warning(
                "RiskResult.from_dict: 'timestamp' missing; defaulting to now()."
            )
        elif isinstance(ts_raw, datetime):
            timestamp = ts_raw
        else:
            timestamp = datetime.fromisoformat(str(ts_raw))

        return cls(
            risk_level=risk_level,
            score=float(data["score"]),
            confidence=float(data["confidence"]),
            reasons=tuple(data.get("reasons", [])),
            recommendation=str(data.get("recommendation", "")),
            processing_time_ms=float(data.get("processing_time_ms", 0.0)),
            timestamp=timestamp,
            metadata=dict(data.get("metadata", {})),
        )

    # ==================================================================
    # Immutable metadata helpers
    # ==================================================================

    def with_metadata(self, **updates: Any) -> "RiskResult":
        """Return a new :class:`RiskResult` with ``metadata`` updated.

        Since ``RiskResult`` is frozen, its ``metadata`` dict cannot be
        mutated in place (and should not be, per the class docstring's
        warning that the dict itself is not copied on construction). This
        helper is the supported way to attach or overwrite metadata keys
        after construction — e.g. when the Integration pipeline (
        ``main.py``) wants to stamp a request ID onto a ``RiskResult``
        that Risk Assessment already produced, without reaching into the
        internals of a frozen dataclass.

        Parameters
        ----------
        **updates : Any
            Key-value pairs to merge into a *copy* of the existing
            ``metadata`` dict. Keys not already present are added; keys
            already present are overwritten. The original instance and
            its ``metadata`` dict are left untouched.

        Returns
        -------
        RiskResult
            A new, independently-validated instance with every field
            identical to ``self`` except the merged ``metadata``.

        Example
        -------
        ::

            tagged = result.with_metadata(request_id="req_123", retried=False)
            assert result.metadata.get("request_id") is None   # original untouched
            assert tagged.metadata["request_id"] == "req_123"
        """
        merged_metadata = {**self.metadata, **updates}
        return replace(self, metadata=merged_metadata)

    # ==================================================================
    # Human-readable representations
    # ==================================================================

    def summary(self) -> str:
        """Return a compact, human-readable one-block summary of this result.

        Suitable for CLI output, logging, and report section headers.

        Returns
        -------
        str
            A multi-line string formatted for terminal display.

        Example output
        --------------
        ::

            ┌─────────────────────────────────────────────────┐
            │  Risk Assessment Result                         │
            ├─────────────────────────────────────────────────┤
            │  Risk Level       : 🔴 High Risk [HIGH_RISK]    │
            │  Score            : 0.8700   (confidence: 0.91) │
            │  Processing time  : 43.200 ms                   │
            │  Timestamp        : 2024-06-01T12:34:56+00:00   │
            ├─────────────────────────────────────────────────┤
            │  Reasons:                                       │
            │    [1] URL domain registered within 7 days.     │
            │    [2] QR pixel density anomaly (score: 0.74).  │
            ├─────────────────────────────────────────────────┤
            │  Recommendation:                                │
            │    Do not scan this QR code.                    │
            └─────────────────────────────────────────────────┘
        """
        width = 55
        sep   = "─" * width

        lines: list[str] = [
            f"┌{sep}┐",
            f"│{'  Risk Assessment Result':<{width}}│",
            f"├{sep}┤",
            f"│  {'Risk Level':<17}: {self.risk_level.display_label} [{self.risk_level.value}]",
            f"│  {'Score':<17}: {self.score:.4f}   (confidence: {self.confidence:.2f})",
            f"│  {'Processing time':<17}: {self.processing_time_ms:.3f} ms",
            f"│  {'Timestamp':<17}: {self.timestamp.isoformat()}",
            f"├{sep}┤",
            f"│  Reasons:",
        ]

        if self.reasons:
            for idx, reason in enumerate(self.reasons, start=1):
                lines.append(f"│    [{idx}] {reason}")
        else:
            lines.append("│    (none)")

        lines += [
            f"├{sep}┤",
            f"│  Recommendation:",
            f"│    {self.recommendation}",
            f"└{sep}┘",
        ]

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a concise single-line description of the result.

        Suitable for log messages and list representations.

        Example
        -------
        ::

            "RiskResult(HIGH_RISK | score=0.8700 | confidence=0.91)"
        """
        return (
            f"RiskResult("
            f"{self.risk_level.value} | "
            f"score={self.score:.4f} | "
            f"confidence={self.confidence:.2f})"
        )

    def __repr__(self) -> str:
        """Return a complete, unambiguous developer representation.

        The output is valid Python that can reconstruct the object (assuming
        ``RiskLevel`` and ``datetime`` are in scope).

        Example
        -------
        ::

            RiskResult(
                risk_level=<RiskLevel.HIGH_RISK: 'HIGH_RISK'>,
                score=0.87,
                confidence=0.91,
                reasons=('URL domain registered within 7 days.',),
                recommendation='Do not scan this QR code.',
                processing_time_ms=43.2,
                timestamp=datetime.datetime(2024, 6, 1, 12, 34, 56, tzinfo=...),
                metadata={'rule_engine_version': '1.0.0'}
            )
        """
        return (
            f"RiskResult(\n"
            f"    risk_level={self.risk_level!r},\n"
            f"    score={self.score!r},\n"
            f"    confidence={self.confidence!r},\n"
            f"    reasons={self.reasons!r},\n"
            f"    recommendation={self.recommendation!r},\n"
            f"    processing_time_ms={self.processing_time_ms!r},\n"
            f"    timestamp={self.timestamp!r},\n"
            f"    metadata={self.metadata!r}\n"
            f")"
        )


# ===========================================================================
# Module-level convenience factory
# ===========================================================================

def make_safe_result(
    *,
    score: float = 0.0,
    confidence: float = 1.0,
    processing_time_ms: float = 0.0,
    reasons: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RiskResult:
    """Convenience factory for constructing a SAFE :class:`RiskResult`.

    Useful in unit tests and pipeline fallback paths where a non-threatening
    result must be returned without constructing all fields manually.

    Parameters
    ----------
    score : float, optional
        Risk score.  Defaults to ``0.0``.
    confidence : float, optional
        Confidence in the assessment.  Defaults to ``1.0``.
    processing_time_ms : float, optional
        Elapsed time.  Defaults to ``0.0``.
    reasons : list[str], optional
        Reason strings.  Defaults to ``["No risk signals detected."]``.
    metadata : dict[str, Any], optional
        Additional annotations.  Defaults to ``{}``.

    Returns
    -------
    RiskResult
        A frozen, validated SAFE result.
    """
    return RiskResult(
        risk_level=RiskLevel.SAFE,
        score=score,
        confidence=confidence,
        reasons=reasons or ["No risk signals detected."],
        recommendation="This QR code appears safe to interact with.",
        processing_time_ms=processing_time_ms,
        timestamp=datetime.now(tz=timezone.utc),
        metadata=metadata or {},
    )


def make_high_risk_result(
    reasons: list[str],
    *,
    score: float = 1.0,
    confidence: float = 1.0,
    processing_time_ms: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> RiskResult:
    """Convenience factory for constructing a HIGH_RISK :class:`RiskResult`.

    Parameters
    ----------
    reasons : list[str]
        Non-empty list of human-readable risk signals.
    score : float, optional
        Risk score.  Defaults to ``1.0``.
    confidence : float, optional
        Confidence in the assessment.  Defaults to ``1.0``.
    processing_time_ms : float, optional
        Elapsed time.  Defaults to ``0.0``.
    metadata : dict[str, Any], optional
        Additional annotations.  Defaults to ``{}``.

    Returns
    -------
    RiskResult
        A frozen, validated HIGH_RISK result.
    """
    return RiskResult(
        risk_level=RiskLevel.HIGH_RISK,
        score=score,
        confidence=confidence,
        reasons=reasons,
        recommendation=(
            "Do not scan or interact with this QR code.  "
            "Report it to your security team."
        ),
        processing_time_ms=processing_time_ms,
        timestamp=datetime.now(tz=timezone.utc),
        metadata=metadata or {},
    )