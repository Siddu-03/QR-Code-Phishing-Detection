"""
report_generator.py
====================
QR Shield — Report Generator
-----------------------------
Project: Computer Vision-Based Graphic Tamper Detection for QR Code Phishing
Prevention

This module is the single source of truth for turning the outputs of every
pipeline stage — QR detection, tamper analysis, and risk assessment — into
one unified, serialisable ``Report`` object.

Consumers
---------
* **CLI** (``main.py``) — may optionally build a ``Report`` after the
  existing detection/visualisation steps to print or persist a structured
  summary. The Report Generator does not import or depend on ``main.py`` and
  has no knowledge of the CLI; it only consumes the plain ``dict`` shape
  that ``qr_detector.detect_qr`` (and therefore ``main.py``'s
  ``detection_result``) already produces.
* **Evaluation Framework** — consumes ``Report.to_dict()`` /
  ``Report.risk_information`` / ``Report.tamper_information`` to compute
  aggregate metrics across a batch of images.
* **FastAPI backend** — returns ``Report.to_dict()`` directly as a JSON
  response body; no custom encoder is required because every field is a
  JSON-safe primitive, list, or dict.
* **Progressive Web Application** — consumes the same JSON shape returned
  by the FastAPI backend.

Design principles
-----------------
* **Independent of the CLI.** This module never imports ``main.py`` and
  never calls ``print()``. It only accepts already-computed result objects
  and plain dictionaries.
* **Tolerant of partial input.** ``TamperResult`` and ``RiskResult`` are
  required (they are the two canonical, validated pipeline contracts).
  QR detection, image metadata, timing, evaluation metadata, and URL
  analysis are all optional so this module keeps working as new stages are
  added or run in isolation (e.g. during unit testing of only the risk
  engine).
* **Extensible without breaking compatibility.** The unified report always
  contains the exact ``required_output_fields`` contract. Anything beyond
  that — including the not-yet-built URL Analyzer — is attached under the
  open ``extra_sections`` slot (see :meth:`Report.with_section`), so new
  sections can be appended without changing the shape of existing fields
  or breaking any consumer that only reads the required fields.
* **No unnecessary copies.** Sub-sections are built once as plain ``dict``
  objects directly from the source result objects; the final ``Report`` is
  a frozen dataclass so it can be shared safely across a batch-evaluation
  loop without defensive copying.

Assumptions
-----------
See the accompanying compatibility audit for the complete list. The single
most important one: :class:`~src.risk_assessment.risk_result.RiskLevel`
only defines three ordinal members (``SAFE``, ``SUSPICIOUS``,
``HIGH_RISK``), while this reporting contract's recommendation ladder has
five tiers (``SAFE`` / ``LOW_RISK`` / ``MEDIUM_RISK`` / ``HIGH_RISK`` /
``CRITICAL``). :func:`ReportGenerator._derive_overall_status` bridges this
gap by refining ``SUSPICIOUS`` and ``HIGH_RISK`` using the continuous
``score`` (and, for the ``CRITICAL`` tier, ``confidence``) already present
on ``RiskResult``, via configurable thresholds on
:class:`ReportGeneratorConfig`. This is reporting-layer refinement only —
it never mutates or replaces the underlying ``RiskResult``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from src.risk_assessment.risk_result import RiskLevel, RiskResult
from src.tamper_analysis.tamper_result import TamperResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the *pipeline* (bump when detection/risk logic
#: changes materially). Independent of the report schema version below.
PIPELINE_VERSION = "1.0.0"

#: Semantic version of the *report shape itself*. Bump only when a
#: required field is added, removed, or renamed — never for additive,
#: optional sections such as ``extra_sections``.
REPORT_SCHEMA_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()



# ---------------------------------------------------------------------------
# Presentation formatting helpers
# ---------------------------------------------------------------------------
# Pure formatting only — no scoring/thresholding logic. TamperResult.confidence
# and RiskResult.score/confidence are normalised floats in [0.0, 1.0]
# internally (unchanged); these helpers are what every user-visible surface
# (Markdown report, HTML fragment, summary sentence) goes through so
# confidence is always a whole-number percentage and risk score is always
# shown on a 0-100 scale, matching the Live Scanner and Evaluation Framework
# report renderers.

def _format_confidence_pct(confidence: float) -> str:
    """Format a normalised [0.0, 1.0] confidence value as a whole-number percentage."""
    return f"{confidence:.0%}"


def _format_score_0_100(score: float) -> str:
    """Format a normalised [0.0, 1.0] score on a 0-100 scale, one decimal."""
    return f"{score * 100:.1f}/100"


def _format_score_from_0_100(score: float) -> str:
    """Format a score already expressed on a 0-100 scale, one decimal.

    Used for URL Analyzer values (``risk_score``), which — unlike
    ``RiskResult.score`` — are already produced on a 0-100 scale rather
    than normalised to [0.0, 1.0].
    """
    return f"{float(score):.1f}/100"


def _format_pct_from_0_100(value: float) -> str:
    """Format a confidence value already expressed on a 0-100 scale as a percentage.

    Used for URL Analyzer confidence, which is documented as a 0-100
    percentage rather than a normalised [0.0, 1.0] float.
    """
    return f"{float(value):.0f}%"


def _new_report_id() -> str:
    """Generate a short, collision-resistant report identifier.

    Format: ``qrs-<YYYYMMDDTHHMMSS>-<8 hex chars>``, e.g.
    ``qrs-20260707T101530-9f3a2c1d``. The timestamp prefix keeps IDs
    roughly sortable in logs and object storage; the suffix guarantees
    uniqueness within the same second across concurrent batch workers.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"qrs-{ts}-{uuid.uuid4().hex[:8]}"


# ===========================================================================
# Overall status ladder
# ===========================================================================

class OverallStatus(str, Enum):
    """Five-tier report-level status, per the QR Shield reporting contract.

    This is distinct from — and derived from — the three-tier
    :class:`~src.risk_assessment.risk_result.RiskLevel` produced by the Risk
    Assessment module. See :func:`ReportGenerator._derive_overall_status`.
    """

    SAFE = "SAFE"
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


#: Recommendation copy for each overall status tier, per
#: ``recommendation_logic`` in the work order.
_RECOMMENDATIONS: Dict[OverallStatus, str] = {
    OverallStatus.SAFE: (
        "This QR code shows no signs of tampering or risk. "
        "It appears safe to open."
    ),
    OverallStatus.LOW_RISK: (
        "Minor risk signals were detected. You may open this QR code, "
        "but proceed with caution."
    ),
    OverallStatus.MEDIUM_RISK: (
        "Several risk signals were detected. Manual verification is "
        "recommended before opening this QR code."
    ),
    OverallStatus.HIGH_RISK: (
        "Strong evidence of tampering or a malicious target was found. "
        "Do not open this QR code."
    ),
    OverallStatus.CRITICAL: (
        "Critical risk detected with high confidence. Block this QR code "
        "immediately and do not interact with it."
    ),
}


class ReportGenerationError(Exception):
    """Raised when a ``Report`` cannot be assembled from the supplied inputs."""


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass(frozen=True)
class ReportGeneratorConfig:
    """Tunable thresholds used to refine :class:`RiskLevel` into
    :class:`OverallStatus`.

    Attributes
    ----------
    suspicious_low_risk_ceiling : float
        Within ``RiskLevel.SUSPICIOUS``, scores at or below this value are
        reported as ``LOW_RISK``; scores above it are reported as
        ``MEDIUM_RISK``.
    critical_score_floor : float
        Within ``RiskLevel.HIGH_RISK``, scores at or above this value
        *and* meeting ``critical_confidence_floor`` are escalated to
        ``CRITICAL``.
    critical_confidence_floor : float
        Confidence required (alongside ``critical_score_floor``) to
        escalate a ``HIGH_RISK`` result to ``CRITICAL``.
    escalate_on_tamper_mismatch : bool
        If ``True`` (default), a confirmed ``TamperResult.tampered`` finding
        is never allowed to be reported as ``OverallStatus.SAFE`` even if
        the Risk Assessment module (which may weigh non-tamper signals too)
        independently classified the image as ``RiskLevel.SAFE``. The
        status is floored to ``MEDIUM_RISK`` in that case. This is a
        defensive safety rule, not a Risk Assessment override — the
        underlying ``RiskResult`` is left untouched.
    """

    suspicious_low_risk_ceiling: float = 0.35
    critical_score_floor: float = 0.90
    critical_confidence_floor: float = 0.85
    escalate_on_tamper_mismatch: bool = True


# ===========================================================================
# Report
# ===========================================================================

@dataclass(frozen=True)
class Report:
    """The unified QR Shield report — single source of truth for all
    downstream consumers (CLI, Evaluation Framework, FastAPI, PWA).

    All fields listed under ``required_output_fields`` in the work order
    are present as top-level attributes. Everything else — including the
    not-yet-built URL Analyzer output — lives under ``extra_sections`` so
    that new sections can be appended without ever changing the shape of
    an existing field.
    """

    # ---- required_output_fields ----------------------------------------
    report_id: str
    timestamp: str
    pipeline_version: str
    overall_status: str
    summary: str
    recommendation: str
    qr_information: Dict[str, Any]
    tamper_information: Dict[str, Any]
    risk_information: Dict[str, Any]
    technical_metadata: Dict[str, Any]
    processing_statistics: Dict[str, Any]

    # ---- open extension slot --------------------------------------------
    # Additive-only. Current known members: "image_information",
    # "url_analysis". Future sections (e.g. evaluation metadata annotations)
    # are appended here without touching the fields above.
    extra_sections: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a flat, JSON-serialisable dictionary.

        Safe to pass directly to ``json.dumps`` or return as a FastAPI
        response body. No nested dataclasses or enums remain — every
        sub-section was already built as a plain dict by
        :class:`ReportGenerator`.
        """
        payload: Dict[str, Any] = {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "pipeline_version": self.pipeline_version,
            "overall_status": self.overall_status,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "qr_information": self.qr_information,
            "tamper_information": self.tamper_information,
            "risk_information": self.risk_information,
            "technical_metadata": self.technical_metadata,
            "processing_statistics": self.processing_statistics,
        }
        payload.update(self.extra_sections)
        return payload

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialise this report directly to a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Render this report as a Markdown document.

        Suitable for CLI ``--export md``, static evaluation artefacts, or
        attaching to an issue tracker. Uses only the data already present
        on this report; no additional computation is performed.
        """
        qr = self.qr_information
        tamper = self.tamper_information
        risk = self.risk_information
        stats = self.processing_statistics
        image = self.extra_sections.get("image_information", {})
        url = self.extra_sections.get("url_analysis", {})

        lines: List[str] = [
            f"# QR Shield Report `{self.report_id}`",
            "",
            f"**Generated:** {self.timestamp}  ",
            f"**Pipeline version:** {self.pipeline_version}  ",
            f"**Overall status:** `{self.overall_status}`",
            "",
            "## Summary",
            "",
            self.summary,
            "",
            "## Recommendation",
            "",
            f"> {self.recommendation}",
            "",
        ]

        if image:
            lines += [
                "## Image Information",
                "",
                f"- Path: `{image.get('path', 'n/a')}`",
                f"- Resolution: {image.get('width', '?')} × "
                f"{image.get('height', '?')} px",
                f"- Format: {image.get('format', 'n/a')}",
                "",
            ]

        lines += [
            "## QR Detection Summary",
            "",
            f"- QR codes detected: {qr.get('qr_count', 0)}",
            f"- Detector used: {qr.get('detector_used', 'n/a')}",
            "",
            "## Tamper Analysis Summary",
            "",
            f"- Tampered: {tamper.get('tampered')}",
            f"- Confidence: {_format_confidence_pct(tamper.get('confidence', 0))}",
            f"- Anomalies detected: {tamper.get('anomaly_count', 0)}",
        ]
        for reason in tamper.get("reasons", []):
            lines.append(f"  - {reason}")

        lines += [
            "",
            "## URL Analysis Summary",
            "",
        ]
        if url.get("available"):
            lines += [
                f"- Classification: {url.get('recommendation', 'n/a')}",
                f"- URL score: {_format_score_from_0_100(url.get('risk_score', 0))}",
                f"- Confidence: {_format_pct_from_0_100(url.get('confidence', 0))}",
            ]
            for reason in url.get("reasons", []):
                lines.append(f"  - {reason}")
        else:
            lines.append("Not available for this scan.")

        lines += [
            "",
            "## Risk Assessment Summary",
            "",
            f"- Risk level: {risk.get('risk_level')}",
            f"- Score: {_format_score_0_100(risk.get('score', 0))}",
            f"- Confidence: {_format_confidence_pct(risk.get('confidence', 0))}",
        ]
        for reason in risk.get("reasons", []):
            lines.append(f"  - {reason}")

        lines += [
            "",
            "## Pipeline Timing",
            "",
            f"- Total pipeline time: {stats.get('total_pipeline_time_ms', 0):.1f} ms",
            f"- Tamper analysis time: {stats.get('tamper_analysis_time_ms', 0):.1f} ms",
            f"- Risk assessment time: {stats.get('risk_assessment_time_ms', 0):.1f} ms",
        ]

        return "\n".join(lines)

    def to_html(self) -> str:
        """Render this report as a minimal, self-contained HTML fragment.

        Intended for embedding into an email, evaluation dashboard, or a
        FastAPI ``HTMLResponse``. Deliberately dependency-free (no Jinja)
        so it works the same way in the CLI, the API, and the PWA build.
        """
        import html as _html

        def esc(value: Any) -> str:
            return _html.escape(str(value))

        qr = self.qr_information
        tamper = self.tamper_information
        risk = self.risk_information
        url = self.extra_sections.get("url_analysis", {})

        reason_items = "".join(f"<li>{esc(r)}</li>" for r in tamper.get("reasons", []))
        risk_reason_items = "".join(f"<li>{esc(r)}</li>" for r in risk.get("reasons", []))

        if url.get("available"):
            url_reason_items = "".join(
                f"<li>{esc(r)}</li>" for r in url.get("reasons", [])
            )
            url_section = (
                f"<h2>URL Analysis</h2>"
                f"<p>Classification: {esc(url.get('recommendation', 'n/a'))}, "
                f"URL score: {_format_score_from_0_100(url.get('risk_score', 0))}, "
                f"Confidence: {_format_pct_from_0_100(url.get('confidence', 0))}</p>"
                f"<ul>{url_reason_items}</ul>"
            )
        else:
            url_section = "<h2>URL Analysis</h2><p>Not available for this scan.</p>"

        return (
            f"<section class='qr-shield-report' data-report-id='{esc(self.report_id)}'>"
            f"<h1>QR Shield Report</h1>"
            f"<p><strong>Status:</strong> {esc(self.overall_status)}</p>"
            f"<p>{esc(self.summary)}</p>"
            f"<blockquote>{esc(self.recommendation)}</blockquote>"
            f"<h2>QR Detection</h2>"
            f"<p>{esc(qr.get('qr_count', 0))} QR code(s) via "
            f"{esc(qr.get('detector_used', 'n/a'))}</p>"
            f"<h2>Tamper Analysis</h2>"
            f"<p>Tampered: {esc(tamper.get('tampered'))}, "
            f"Confidence: {_format_confidence_pct(tamper.get('confidence', 0))}</p>"
            f"<ul>{reason_items}</ul>"
            f"{url_section}"
            f"<h2>Risk Assessment</h2>"
            f"<p>Level: {esc(risk.get('risk_level'))}, "
            f"Score: {_format_score_0_100(risk.get('score', 0))}</p>"
            f"<ul>{risk_reason_items}</ul>"
            f"</section>"
        )

    # ------------------------------------------------------------------
    # Extension helper
    # ------------------------------------------------------------------
    def with_section(self, name: str, data: Dict[str, Any]) -> "Report":
        """Return a new ``Report`` with an additional/overwritten section
        under ``extra_sections``.

        This is the supported way to attach future data — most notably the
        URL Analyzer output — without altering any required field on the
        existing report.

        Parameters
        ----------
        name : str
            Section key, e.g. ``"url_analysis"``.
        data : dict
            JSON-serialisable payload for that section.

        Returns
        -------
        Report
            A new instance; ``self`` is left unmodified.
        """
        merged = {**self.extra_sections, name: data}
        return Report(
            report_id=self.report_id,
            timestamp=self.timestamp,
            pipeline_version=self.pipeline_version,
            overall_status=self.overall_status,
            summary=self.summary,
            recommendation=self.recommendation,
            qr_information=self.qr_information,
            tamper_information=self.tamper_information,
            risk_information=self.risk_information,
            technical_metadata=self.technical_metadata,
            processing_statistics=self.processing_statistics,
            extra_sections=merged,
        )


# ===========================================================================
# Report Generator
# ===========================================================================

class ReportGenerator:
    """Builds a :class:`Report` from validated pipeline outputs.

    Stateless aside from configuration — safe to instantiate once and reuse
    across a batch-evaluation loop, or to construct per-request in a FastAPI
    handler.

    Public methods
    --------------
    generate
        Build a complete :class:`Report` from a ``TamperResult``, a
        ``RiskResult``, and any combination of optional pipeline context.
    """

    def __init__(self, config: Optional[ReportGeneratorConfig] = None) -> None:
        self._config = config or ReportGeneratorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        tamper_result: TamperResult,
        risk_result: RiskResult,
        qr_detection_result: Optional[Mapping[str, Any]] = None,
        image_info: Optional[Mapping[str, Any]] = None,
        processing_stats: Optional[Mapping[str, Any]] = None,
        url_analysis_result: Optional[Mapping[str, Any]] = None,
        evaluation_metadata: Optional[Mapping[str, Any]] = None,
        pipeline_version: Optional[str] = None,
        report_id: Optional[str] = None,
    ) -> Report:
        """Assemble a unified :class:`Report`.

        Parameters
        ----------
        tamper_result : TamperResult
            Required. Output of the Tamper Detection Engine.
        risk_result : RiskResult
            Required. Output of the Risk Assessment module.
        qr_detection_result : Mapping[str, Any], optional
            The ``DetectionResult`` dict produced by
            ``qr_detector.detect_qr`` (the same shape ``main.py`` already
            works with: ``count``, ``detections``, ``detector_used``, ...).
            If omitted, the QR Detection Summary section is populated with
            neutral defaults and a warning is logged.
        image_info : Mapping[str, Any], optional
            The ``load_image`` result dict (``path``, ``width``, ``height``,
            ``channels``, ``format``) as already produced in ``main.py``.
        processing_stats : Mapping[str, Any], optional
            Free-form timing dict. Recognised keys: ``total_pipeline_time_ms``,
            ``qr_detection_time_ms``, ``preprocessing_time_ms``. Tamper and
            risk timing are always taken from the result objects themselves
            and do not need to be repeated here.
        url_analysis_result : Mapping[str, Any], optional
            Reserved for the future URL Analyzer. When omitted, the report
            still contains a clearly-marked "not yet available" placeholder
            section so downstream schemas do not need to special-case its
            absence.
        evaluation_metadata : Mapping[str, Any], optional
            Free-form annotations from the Evaluation Framework (e.g.
            dataset split, ground-truth label, run ID). Stored verbatim
            under ``technical_metadata.evaluation``.
        pipeline_version : str, optional
            Overrides the module-level :data:`PIPELINE_VERSION` default.
        report_id : str, optional
            Overrides the auto-generated report identifier. Useful when the
            caller already has a correlation/request ID to reuse.

        Returns
        -------
        Report

        Raises
        ------
        ReportGenerationError
            If ``tamper_result`` or ``risk_result`` fail validation.
        """
        logger.debug("ReportGenerator.generate: starting report assembly.")

        self._validate_tamper_result(tamper_result)
        self._validate_risk_result(risk_result)

        rid = report_id or _new_report_id()
        timestamp = _utc_now_iso()
        version = pipeline_version or PIPELINE_VERSION

        qr_information = self._build_qr_information(qr_detection_result)
        tamper_information = self._build_tamper_information(tamper_result)
        risk_information = self._build_risk_information(risk_result)
        processing_statistics = self._build_processing_statistics(
            tamper_result, risk_result, processing_stats
        )
        technical_metadata = self._build_technical_metadata(
            version, evaluation_metadata
        )

        overall_status = self._derive_overall_status(risk_result, tamper_result)
        recommendation = _RECOMMENDATIONS[overall_status]
        summary = self._build_summary(
            overall_status, qr_information, tamper_information, risk_information
        )

        extra_sections: Dict[str, Any] = {
            "url_analysis": self._build_url_analysis_placeholder(url_analysis_result),
        }
        image_information = self._build_image_information(image_info)
        if image_information is not None:
            extra_sections["image_information"] = image_information

        report = Report(
            report_id=rid,
            timestamp=timestamp,
            pipeline_version=version,
            overall_status=overall_status.value,
            summary=summary,
            recommendation=recommendation,
            qr_information=qr_information,
            tamper_information=tamper_information,
            risk_information=risk_information,
            technical_metadata=technical_metadata,
            processing_statistics=processing_statistics,
            extra_sections=extra_sections,
        )

        logger.info(
            "Report generated: id=%s status=%s tampered=%s risk_level=%s",
            rid,
            overall_status.value,
            tamper_result.tampered,
            risk_result.risk_level.value,
        )
        return report

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_tamper_result(tamper_result: TamperResult) -> None:
        if not isinstance(tamper_result, TamperResult):
            logger.error(
                "ReportGenerator: invalid tamper_result type: %s",
                type(tamper_result).__name__,
            )
            raise ReportGenerationError(
                "tamper_result must be a TamperResult instance, got "
                f"{type(tamper_result).__name__!r}."
            )

    @staticmethod
    def _validate_risk_result(risk_result: RiskResult) -> None:
        if not isinstance(risk_result, RiskResult):
            logger.error(
                "ReportGenerator: invalid risk_result type: %s",
                type(risk_result).__name__,
            )
            raise ReportGenerationError(
                "risk_result must be a RiskResult instance, got "
                f"{type(risk_result).__name__!r}."
            )

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------
    @staticmethod
    def _build_qr_information(
        qr_detection_result: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if qr_detection_result is None:
            logger.warning(
                "ReportGenerator: no qr_detection_result supplied; "
                "QR Detection Summary will use neutral defaults."
            )
            return {
                "qr_count": 0,
                "has_qr_codes": False,
                "detector_used": None,
                "detections": [],
            }

        count = int(qr_detection_result.get("count", 0))
        detections = list(qr_detection_result.get("detections", []))
        return {
            "qr_count": count,
            "has_qr_codes": count > 0,
            "detector_used": qr_detection_result.get("detector_used"),
            "detections": [
                {
                    "data": det.get("data"),
                    "bbox_tuple": det.get("bbox_tuple"),
                    "corner_points": det.get("corner_points"),
                }
                for det in detections
            ],
        }

    @staticmethod
    def _build_image_information(
        image_info: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if image_info is None:
            return None
        return {
            "path": image_info.get("path"),
            "width": image_info.get("width"),
            "height": image_info.get("height"),
            "channels": image_info.get("channels"),
            "format": image_info.get("format"),
        }

    @staticmethod
    def _build_tamper_information(tamper_result: TamperResult) -> Dict[str, Any]:
        highest = tamper_result.highest_severity()
        return {
            "tampered": tamper_result.tampered,
            "confidence": round(tamper_result.confidence, 4),
            "reasons": list(tamper_result.reasons),
            "analysis_time_ms": round(float(tamper_result.analysis_time_ms), 2),
            "anomaly_count": len(tamper_result.anomalies),
            "highest_severity": highest.value if highest else None,
            "visualization_path": tamper_result.visualization_path,
        }

    @staticmethod
    def _build_risk_information(risk_result: RiskResult) -> Dict[str, Any]:
        return {
            "risk_level": risk_result.risk_level.value,
            "risk_level_severity": risk_result.risk_level.severity,
            "score": round(risk_result.score, 6),
            "confidence": round(risk_result.confidence, 6),
            "reasons": list(risk_result.reasons),
            "recommendation": risk_result.recommendation,
            "processing_time_ms": round(risk_result.processing_time_ms, 3),
        }

    @staticmethod
    def _build_url_analysis_placeholder(
        url_analysis_result: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if url_analysis_result is not None:
            return {"available": True, **dict(url_analysis_result)}
        return {
            "available": False,
            "note": "URL analysis is not yet integrated into this pipeline.",
        }

    @staticmethod
    def _build_processing_statistics(
        tamper_result: TamperResult,
        risk_result: RiskResult,
        processing_stats: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        extra = dict(processing_stats or {})
        tamper_ms = float(tamper_result.analysis_time_ms)
        risk_ms = float(risk_result.processing_time_ms)
        qr_ms = extra.pop("qr_detection_time_ms", None)
        prep_ms = extra.pop("preprocessing_time_ms", None)
        total_ms = extra.pop("total_pipeline_time_ms", None)

        if total_ms is None:
            # Best-effort total from the components we know about; callers
            # that track wall-clock time end-to-end should pass
            # total_pipeline_time_ms explicitly for an authoritative figure.
            total_ms = tamper_ms + risk_ms + (qr_ms or 0.0) + (prep_ms or 0.0)

        stages_completed = ["tamper_analysis", "risk_assessment"]
        if qr_ms is not None:
            stages_completed.insert(0, "qr_detection")
        if prep_ms is not None:
            stages_completed.insert(0, "preprocessing")

        stats = {
            "total_pipeline_time_ms": round(float(total_ms), 2),
            "tamper_analysis_time_ms": round(tamper_ms, 2),
            "risk_assessment_time_ms": round(risk_ms, 2),
            "qr_detection_time_ms": round(float(qr_ms), 2) if qr_ms is not None else None,
            "preprocessing_time_ms": round(float(prep_ms), 2) if prep_ms is not None else None,
            "stages_completed": stages_completed,
        }
        stats.update(extra)
        return stats

    @staticmethod
    def _build_technical_metadata(
        pipeline_version: str,
        evaluation_metadata: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "pipeline_version": pipeline_version,
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "generator": "ReportGenerator",
        }
        if evaluation_metadata:
            metadata["evaluation"] = dict(evaluation_metadata)
        return metadata

    # ------------------------------------------------------------------
    # Status derivation
    # ------------------------------------------------------------------
    def _derive_overall_status(
        self, risk_result: RiskResult, tamper_result: TamperResult
    ) -> OverallStatus:
        """Refine the three-tier ``RiskLevel`` into the five-tier
        ``OverallStatus`` required by the reporting contract.

        See the module docstring's "Assumptions" section and
        :class:`ReportGeneratorConfig` for the thresholds used here.
        """
        cfg = self._config
        level = risk_result.risk_level

        if level is RiskLevel.SAFE:
            status = OverallStatus.SAFE
        elif level is RiskLevel.SUSPICIOUS:
            status = (
                OverallStatus.LOW_RISK
                if risk_result.score <= cfg.suspicious_low_risk_ceiling
                else OverallStatus.MEDIUM_RISK
            )
        else:  # RiskLevel.HIGH_RISK
            status = (
                OverallStatus.CRITICAL
                if (
                    risk_result.score >= cfg.critical_score_floor
                    and risk_result.confidence >= cfg.critical_confidence_floor
                )
                else OverallStatus.HIGH_RISK
            )

        if (
            cfg.escalate_on_tamper_mismatch
            and tamper_result.tampered
            and status is OverallStatus.SAFE
        ):
            logger.warning(
                "ReportGenerator: TamperResult reports tampering but "
                "RiskResult classified the image as SAFE; escalating "
                "overall_status to MEDIUM_RISK for the report."
            )
            status = OverallStatus.MEDIUM_RISK

        return status

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary(
        overall_status: OverallStatus,
        qr_information: Dict[str, Any],
        tamper_information: Dict[str, Any],
        risk_information: Dict[str, Any],
    ) -> str:
        qr_count = qr_information["qr_count"]
        qr_clause = (
            f"{qr_count} QR code(s) detected"
            if qr_count
            else "no QR codes detected"
        )
        tamper_clause = (
            "tampering evidence found"
            if tamper_information["tampered"]
            else "no tampering evidence found"
        )
        return (
            f"Overall status: {overall_status.value}. {qr_clause}; "
            f"{tamper_clause} (confidence "
            f"{_format_confidence_pct(tamper_information['confidence'])}); risk level "
            f"{risk_information['risk_level']} (score "
            f"{_format_score_0_100(risk_information['score'])}, confidence "
            f"{_format_confidence_pct(risk_information['confidence'])})."
        )


# ===========================================================================
# Module-level convenience factory
# ===========================================================================

def generate_report(
    *,
    tamper_result: TamperResult,
    risk_result: RiskResult,
    qr_detection_result: Optional[Mapping[str, Any]] = None,
    image_info: Optional[Mapping[str, Any]] = None,
    processing_stats: Optional[Mapping[str, Any]] = None,
    url_analysis_result: Optional[Mapping[str, Any]] = None,
    evaluation_metadata: Optional[Mapping[str, Any]] = None,
    config: Optional[ReportGeneratorConfig] = None,
) -> Report:
    """Convenience wrapper around ``ReportGenerator().generate(...)``.

    Useful for one-off call sites (CLI, quick scripts, tests) that don't
    need to hold onto a ``ReportGenerator`` instance. Batch-processing call
    sites (Evaluation Framework, FastAPI app startup) should instantiate
    :class:`ReportGenerator` once and reuse it instead.
    """
    return ReportGenerator(config=config).generate(
        tamper_result=tamper_result,
        risk_result=risk_result,
        qr_detection_result=qr_detection_result,
        image_info=image_info,
        processing_stats=processing_stats,
        url_analysis_result=url_analysis_result,
        evaluation_metadata=evaluation_metadata,
    )