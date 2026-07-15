"""
Pydantic models for incoming request payloads.
Note: the primary scan endpoint accepts multipart/form-data (UploadFile),
so its "body" is handled via FastAPI's File(...)/Form(...) params rather
than a JSON schema. These models cover the JSON-based endpoints.
"""
from typing import Optional

from pydantic import BaseModel, Field


class ScanOptions(BaseModel):
    """
    Optional tuning parameters sent alongside an image upload (as form
    fields) to control tamper-detection sensitivity.
    """
    edge_weight: float = Field(default=0.35, ge=0, le=1)
    contour_weight: float = Field(default=0.35, ge=0, le=1)
    overlay_weight: float = Field(default=0.30, ge=0, le=1)
    tamper_threshold: float = Field(
        default=0.5, ge=0, le=1,
        description="Confidence score above which an image is flagged as tampered"
    )


class ReportRequest(BaseModel):
    scan_id: str = Field(..., description="ID of a previously completed scan")
    format: str = Field(default="pdf", pattern="^(pdf|json)$")
    include_image: bool = Field(default=True)


class HistoryQuery(BaseModel):
    limit: int = Field(default=20, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    tampered_only: Optional[bool] = Field(default=None)
