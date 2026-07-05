"""
tamper_result.py
----------------
Standardized result object for the Tamper Detection Engine.

This module defines TamperResult, a dataclass that every detector
(edge, contour, overlay, pattern) and the aggregate TamperDetector
use to report findings in a consistent shape.

Member 1 — Tamper Detection Engine
Branch: feature/tamper-engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import json


@dataclass
class TamperResult:
    """
    Standardized output of a tamper analysis.

    Attributes:
        tampered (bool): Final aggregate decision — True if the QR code
            is judged to be tampered with.
        confidence (float): Confidence score in [0.0, 1.0] for the decision.
        reasons (List[str]): Human-readable reasons supporting the decision
            (e.g. "overlay detected", "contour mismatch").
        details (Dict[str, Any]): Optional raw sub-scores / metadata from
            each individual detector, useful for debugging and for
            downstream consumers (e.g. the reporting/UI branch).
    """

    tampered: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Defensive validation so bad inputs fail fast instead of
        # silently propagating downstream.
        if not isinstance(self.tampered, bool):
            raise TypeError("tampered must be a bool")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        self.confidence = round(float(self.confidence), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Return the exact dict shape required by the deliverable spec."""
        return {
            "tampered": self.tampered,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Return dict including internal details, for debugging/logging."""
        return {
            "tampered": self.tampered,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }

    def to_json(self, full: bool = False) -> str:
        """Serialize to a JSON string."""
        payload = self.to_full_dict() if full else self.to_dict()
        return json.dumps(payload, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TamperResult":
        """Reconstruct a TamperResult from a dict (e.g. loaded from JSON)."""
        return cls(
            tampered=bool(data["tampered"]),
            confidence=float(data["confidence"]),
            reasons=list(data.get("reasons", [])),
            details=dict(data.get("details", {})),
        )

    def merge_reasons(self, *reason_lists: List[str]) -> None:
        """Utility to combine reason lists from multiple sub-detectors,
        de-duplicating while preserving order."""
        seen = set(self.reasons)
        for rlist in reason_lists:
            for r in rlist:
                if r and r not in seen:
                    self.reasons.append(r)
                    seen.add(r)

