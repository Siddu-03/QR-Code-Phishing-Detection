"""
Pydantic models for outgoing JSON responses. Keeping these separate from
request schemas makes the API contract explicit for Member 2 (frontend)
and Member 3 (integration/docs).
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str
    environment: str
    timestamp: datetime


class DetectorScore(BaseModel):
    name: str
    raw_score: float
    weight: float
    weighted_score: float


class TamperResult(BaseModel):
    is_tampered: bool
    confidence: float = Field(..., description="Overall weighted tamper confidence, 0-1")
    threshold: float
    detectors: List[DetectorScore]
    reasons: List[str] = Field(default_factory=list)
    engine: str = Field(
        ..., description="Fully-qualified name of the QR Shield engine component that produced this result"
    )


class URLAnalysisResult(BaseModel):
    """
    Populated once the backend is wired to src/url_analysis. Optional/None
    until that integration lands - see Change 2/3 of the backend audit.
    """
    analyzed: bool = False
    url: Optional[str] = None
    is_suspicious: Optional[bool] = None
    reasons: List[str] = Field(default_factory=list)


class RiskAssessmentResult(BaseModel):
    """
    Populated once the backend is wired to src/risk_assessment. Optional/
    None until that integration lands - see Change 2/3 of the backend audit.
    """
    assessed: bool = False
    risk_level: Optional[str] = Field(default=None, description="e.g. low, medium, high, critical")
    score: Optional[float] = None
    recommendation: Optional[str] = None


class ProcessingTimes(BaseModel):
    qr_detection_ms: Optional[float] = None
    tamper_analysis_ms: Optional[float] = None
    url_analysis_ms: Optional[float] = None
    risk_assessment_ms: Optional[float] = None
    report_generation_ms: Optional[float] = None
    total_ms: Optional[float] = None


class QRDecodeResult(BaseModel):
    decoded: bool
    data: Optional[str] = None
    qr_type: Optional[str] = None
    bounding_box: Optional[List[List[float]]] = None


class ScanResponse(BaseModel):
    scan_id: str
    filename: str
    scanned_at: datetime
    qr: QRDecodeResult
    tamper: TamperResult
    url_analysis: URLAnalysisResult = Field(default_factory=URLAnalysisResult)
    risk_assessment: RiskAssessmentResult = Field(default_factory=RiskAssessmentResult)
    verdict: str = Field(..., description="One of: safe, suspicious, tampered, no_qr_found")
    recommendation: Optional[str] = Field(
        default=None, description="Human-readable recommendation for the end user"
    )
    confidence: float = Field(
        default=0.0, description="Top-level overall confidence surfaced for the frontend, 0-1"
    )
    processing_times: ProcessingTimes = Field(default_factory=ProcessingTimes)


class ScanHistoryItem(BaseModel):
    scan_id: str
    filename: str
    scanned_at: datetime
    verdict: str
    confidence: float


class HistoryResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[ScanHistoryItem]


class ReportResponse(BaseModel):
    scan_id: str
    report_url: str
    generated_at: datetime


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
