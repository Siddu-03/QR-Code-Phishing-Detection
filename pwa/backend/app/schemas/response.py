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
    verdict: str = Field(..., description="One of: safe, suspicious, tampered, no_qr_found")


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
