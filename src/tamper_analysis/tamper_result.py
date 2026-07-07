"""
tamper_result.py
-----------------
Unified result container for the QR Shield Tamper Detection Engine.

This module reconciles three previously-divergent `TamperResult` contracts
found across the QR Shield codebase:

1. The original rich result object   -> is_tampered / anomalies / detector_scores
2. The Week 3/4 test suite's contract -> tampered / confidence / reasons
3. The work-order specification       -> tampered / reasons / analysis_time_ms / metadata

`TamperResult` below is a single dataclass whose canonical fields are
`tampered`, `confidence`, and `reasons` (matching the test suite and the
work order), extended with `analysis_time_ms` and `metadata` (also from
the work order), plus the original rich fields (`anomalies`,
`detector_scores`, `visualization_path`, `timestamp`, `image_source`) for
callers that need full diagnostic detail.

`is_tampered` and `processing_time_ms` are kept as read-only properties
aliasing `tampered` / `analysis_time_ms`, for source compatibility with
any code written against the original field names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(Enum):
    """Qualitative severity of a single detected anomaly."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TamperType(Enum):
    """Category of a single detected anomaly."""

    EDGE_DISCONTINUITY = "edge_discontinuity"
    CONTOUR_IRREGULARITY = "contour_irregularity"
    OVERLAY_STICKER = "overlay_sticker"
    MODULE_GRID_MISMATCH = "module_grid_mismatch"
    FINDER_PATTERN_DAMAGE = "finder_pattern_damage"
    UNKNOWN = "unknown"


@dataclass
class Anomaly:
    """A single detected anomaly from any analysis stage."""

    type: TamperType
    severity: Severity
    confidence: float
    description: str
    bbox: Optional[List[int]] = None  # [x, y, w, h] if localized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "confidence": round(float(self.confidence), 4),
            "description": self.description,
            "bbox": self.bbox,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Anomaly":
        return cls(
            type=TamperType(d.get("type", TamperType.UNKNOWN.value)),
            severity=Severity(d.get("severity", Severity.LOW.value)),
            confidence=float(d.get("confidence", 0.0)),
            description=d.get("description", ""),
            bbox=d.get("bbox"),
        )


@dataclass
class DetectorScore:
    """Per-detector raw output, retained for weighted-aggregation audit trails."""

    name: str
    score: float
    weight: float
    processing_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(float(self.score), 4),
            "weight": round(float(self.weight), 4),
            "processing_time_ms": round(float(self.processing_time_ms), 2),
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DetectorScore":
        return cls(
            name=d.get("name", "unknown"),
            score=float(d.get("score", 0.0)),
            weight=float(d.get("weight", 0.0)),
            processing_time_ms=float(d.get("processing_time_ms", 0.0)),
            details=d.get("details", {}),
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TamperResult:
    """Unified output of the Tamper Detection Engine.

    Canonical fields
    ----------------
    tampered : bool
        Final binary tamper decision.
    confidence : float
        Overall confidence in [0.0, 1.0].
    reasons : List[str]
        Deduplicated, order-preserving list of human-readable reasons.

    Extended fields (all optional; default to empty/neutral values)
    -----------------------------------------------------------------
    analysis_time_ms, metadata, anomalies, detector_scores,
    visualization_path, timestamp, image_source.
    """

    tampered: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    analysis_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[Anomaly] = field(default_factory=list)
    detector_scores: List[DetectorScore] = field(default_factory=list)
    visualization_path: Optional[str] = None
    timestamp: str = field(default_factory=_utc_now_iso)
    image_source: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tampered, bool):
            raise TypeError(
                f"TamperResult.tampered must be a bool, got {type(self.tampered).__name__}"
            )
        if not isinstance(self.confidence, (int, float)) or not (
            0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError(
                f"TamperResult.confidence must be within [0.0, 1.0], got {self.confidence!r}"
            )
        self.confidence = float(self.confidence)

    # ------------------------------------------------------------------
    # Backward-compatibility aliases (original rich-contract field names)
    # ------------------------------------------------------------------
    @property
    def is_tampered(self) -> bool:
        """Alias for `tampered`."""
        return self.tampered

    @property
    def processing_time_ms(self) -> float:
        """Alias for `analysis_time_ms`."""
        return self.analysis_time_ms

    # ------------------------------------------------------------------
    # Canonical (lightweight) serialization — matches the test-suite and
    # work-order contract exactly: {tampered, confidence, reasons}.
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tampered": self.tampered,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
        }

    # ------------------------------------------------------------------
    # Extended (rich) serialization — full diagnostic shape.
    # ------------------------------------------------------------------
    def to_full_dict(self) -> Dict[str, Any]:
        return {
            "tampered": self.tampered,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
            "analysis_time_ms": round(float(self.analysis_time_ms), 2),
            "metadata": self.metadata,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "detector_scores": [d.to_dict() for d in self.detector_scores],
            "visualization_path": self.visualization_path,
            "timestamp": self.timestamp,
            "image_source": self.image_source,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_full_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TamperResult":
        """Reconstruct a TamperResult from either `to_dict()` or
        `to_full_dict()` output."""
        return cls(
            tampered=bool(d["tampered"]),
            confidence=float(d["confidence"]),
            reasons=list(d.get("reasons", [])),
            analysis_time_ms=float(d.get("analysis_time_ms", 0.0)),
            metadata=d.get("metadata", {}),
            anomalies=[Anomaly.from_dict(a) for a in d.get("anomalies", [])],
            detector_scores=[
                DetectorScore.from_dict(s) for s in d.get("detector_scores", [])
            ],
            visualization_path=d.get("visualization_path"),
            timestamp=d.get("timestamp", _utc_now_iso()),
            image_source=d.get("image_source"),
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def merge_reasons(self, *reason_lists: List[str]) -> None:
        """Merge one or more reason lists into `self.reasons`, de-duplicating
        while preserving first-seen order (existing reasons included)."""
        seen = set(self.reasons)
        for reasons in reason_lists:
            for reason in reasons:
                if reason not in seen:
                    self.reasons.append(reason)
                    seen.add(reason)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        status = "TAMPERED" if self.tampered else "CLEAN"
        lines = [
            f"Status: {status} (confidence={self.confidence:.2%})",
            f"Analysis time: {self.analysis_time_ms:.1f} ms",
            f"Reasons: {len(self.reasons)}",
        ]
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        if self.anomalies:
            lines.append(f"Anomalies detected: {len(self.anomalies)}")
            for a in self.anomalies:
                lines.append(
                    f"  - [{a.severity.value.upper()}] {a.type.value}: {a.description}"
                )
        return "\n".join(lines)

    def highest_severity(self) -> Optional[Severity]:
        order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
        if not self.anomalies:
            return None
        return max((a.severity for a in self.anomalies), key=lambda s: order[s])