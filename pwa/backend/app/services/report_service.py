"""
Report Service: generates downloadable reports (PDF or JSON) summarizing
a completed scan (QR decode + tamper analysis result).
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ReportService:
    def __init__(self):
        self.report_dir = Path(settings.REPORT_DIR)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, record: Dict[str, Any], fmt: str = "pdf") -> str:
        if fmt == "json":
            return self._generate_json(record)
        return self._generate_pdf(record)

    def _generate_json(self, record: Dict[str, Any]) -> str:
        path = self.report_dir / f"{record['scan_id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        return str(path)

    def _generate_pdf(self, record: Dict[str, Any]) -> str:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
            from reportlab.pdfgen import canvas
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF generation dependency 'reportlab' is not installed",
            ) from exc

        path = self.report_dir / f"{record['scan_id']}.pdf"
        c = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        y = height - inch

        def line(text: str, size: int = 11, gap: float = 0.28):
            nonlocal y
            c.setFont("Helvetica", size)
            c.drawString(inch, y, text)
            y -= gap * inch

        c.setFont("Helvetica-Bold", 18)
        c.drawString(inch, y, "QR Shield - Scan Report")
        y -= 0.4 * inch

        line(f"Scan ID: {record.get('scan_id', 'N/A')}")
        line(f"Filename: {record.get('filename', 'N/A')}")
        line(f"Scanned At: {record.get('scanned_at', datetime.utcnow().isoformat())}")
        line(f"Verdict: {record.get('verdict', 'unknown').upper()}")
        y -= 0.1 * inch

        qr = record.get("qr", {})
        line("QR Decode:", size=13)
        line(f"  Decoded: {qr.get('decoded')}")
        if qr.get("decoded"):
            line(f"  Data: {qr.get('data')}")
        y -= 0.1 * inch

        tamper = record.get("tamper", {})
        line("Tamper Analysis:", size=13)
        line(f"  Is Tampered: {tamper.get('is_tampered')}")
        line(f"  Confidence: {tamper.get('confidence')} (threshold {tamper.get('threshold')})")
        for det in tamper.get("detectors", []):
            line(f"    - {det.get('name')}: raw={det.get('raw_score')} weighted={det.get('weighted_score')}")
        for reason in tamper.get("reasons", []):
            line(f"    * {reason}")

        c.showPage()
        c.save()
        return str(path)


report_service = ReportService()
