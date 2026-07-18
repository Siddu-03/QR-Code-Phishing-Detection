"""
Pydantic models for incoming request payloads.
Note: the primary scan endpoint accepts multipart/form-data (UploadFile),
so its "body" is handled via FastAPI's File(...)/Form(...) params rather
than a JSON schema. These models cover the JSON-based endpoints.
"""
from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    scan_id: str = Field(..., description="ID of a previously completed scan")
    format: str = Field(default="pdf", pattern="^(pdf|json)$")
    include_image: bool = Field(default=True)
