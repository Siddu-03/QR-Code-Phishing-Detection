from dataclasses import dataclass


@dataclass
class Report:
    detected: bool
    tampered: bool
    confidence: float
    risk_level: str
    score: int
    summary: str
    timestamp: str