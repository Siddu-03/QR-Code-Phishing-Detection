"""
Lightweight persistence layer for scan history.

No SQL database is specified in the project scope, so history is kept
as a JSON file on disk (storage/history.json). This keeps Member 1's
backend dependency-free while still surviving server restarts. If the
team later adds Postgres/SQLite, this module is the only place that
needs to change (services/endpoints call HistoryStore, not the file
directly).
"""
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

_lock = threading.Lock()


class HistoryStore:
    """Thread-safe JSON-file-backed store for scan records."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or settings.HISTORY_FILE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, records: List[Dict[str, Any]]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)

    def add(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with _lock:
            records = self._read()
            record.setdefault("scanned_at", datetime.utcnow().isoformat())
            records.insert(0, record)  # newest first
            self._write(records)
            return record

    def get(self, scan_id: str) -> Optional[Dict[str, Any]]:
        for record in self._read():
            if record.get("scan_id") == scan_id:
                return record
        return None

    def list(
        self, limit: int = 20, offset: int = 0, tampered_only: Optional[bool] = None
    ) -> Dict[str, Any]:
        records = self._read()
        if tampered_only is not None:
            records = [
                r for r in records
                if r.get("tamper", {}).get("is_tampered") == tampered_only
            ]
        total = len(records)
        page = records[offset: offset + limit]
        return {"total": total, "items": page}


history_store = HistoryStore()
