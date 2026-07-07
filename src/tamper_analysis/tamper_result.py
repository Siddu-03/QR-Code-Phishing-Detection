"""
tamper_result.py
Unified result container for the QR Shield Tamper Detection Engine.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TamperType(Enum):
    EDGE_DISCONTINUITY = "edge_discontinuity"
    CONTOUR_IRREGULARITY = "contour_irregularity"
    OVERLAY_STICKER = "overlay_sticker"
    MODULE_GRID_MISMATCH = "module_grid_mismatch"
    FINDER_PATTERN_DAMAGE = "finder_pattern_damage"
    UNKNOWN = "unknown"


@dataclass
class Anomaly:
    """A single detected anomaly from any detector."""
    type: TamperType
    severity: Severity
    confidence: float          # 0.0 - 1.0
    description: str
    bbox: Optional[List[int]] = None   # [x, y, w, h] if localized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "confidence": round(float(self.confidence), 4),
            "description": self.description,
            "bbox": self.bbox,
        }


@dataclass
class DetectorScore:
    """Per-detector raw output, used for weighted aggregation."""
    name: str
    score: float                # 0.0 (clean) - 1.0 (tampered)
    weight: float
    processing_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(float(self.score), 4),
            "weight": round(float(self.weight), 4),
            "processing_time_ms": round(float(self.processing_time_ms), 2),
            "details": self.details,
        }


@dataclass
class TamperResult:
    """Unified output of the Tamper Detection Engine."""
    is_tampered: bool
    confidence: float                       # overall confidence 0-1
    anomalies: List[Anomaly] = field(default_factory=list)
    detector_scores: List[DetectorScore] = field(default_factory=list)
    processing_time_ms: float = 0.0
    visualization_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    image_source: Optional[str] = None

    def summary(self) -> str:
        status = "TAMPERED" if self.is_tampered else "CLEAN"
        lines = [
            f"Status: {status} (confidence={self.confidence:.2%})",
            f"Processing time: {self.processing_time_ms:.1f} ms",
            f"Anomalies detected: {len(self.anomalies)}",
        ]
        for a in self.anomalies:
            lines.append(f"  - [{a.severity.value.upper()}] {a.type.value}: {a.description}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_tampered": self.is_tampered,
            "confidence": round(float(self.confidence), 4),
            "anomalies": [a.to_dict() for a in self.anomalies],
            "detector_scores": [d.to_dict() for d in self.detector_scores],
            "processing_time_ms": round(float(self.processing_time_ms), 2),
            "visualization_path": self.visualization_path,
            "timestamp": self.timestamp,
            "image_source": self.image_source,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def highest_severity(self) -> Optional[Severity]:
        order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
        if not self.anomalies:
            return None
        return max((a.severity for a in self.anomalies), key=lambda s: order[s])
