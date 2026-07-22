"""
Shared pytest fixtures for the QR Shield backend test suite (Change 8).

Environment variables must be set BEFORE `app.core.config.get_settings()`
is first called anywhere (it's `@lru_cache`d and evaluated at import time
in several modules), so this module sets them at import time, before any
`app.*` / `main` import happens in the fixtures below.
"""
import os
import shutil
import tempfile
from pathlib import Path

TEST_API_KEY = "test-suite-api-key"

# Isolated, throwaway storage dir for the whole test session so test runs
# never touch a developer's real ./storage/history.json.
_STORAGE_DIR = Path(tempfile.mkdtemp(prefix="qr_shield_backend_tests_"))

def _guess_core_path() -> str:
    """Best-effort default for QR_SHIELD_CORE_PATH when it isn't already
    set in the environment: looks for a sibling checkout of the main
    QR Shield repo (a directory containing `src/qr_detector`) near this
    backend project. Falls back to the same relative default used by
    app.core.config.Settings if nothing is found - tests that need the
    real engine simply skip (see `engine_available` fixture) rather than
    guessing wrong and faking a result.
    """
    backend_root = Path(__file__).resolve().parents[1]
    candidates = [
        backend_root.parent / "QR-Code-Phishing-Detection",
        backend_root.parent,
        backend_root / "QR-Code-Phishing-Detection",
    ]
    for candidate in candidates:
        if (candidate / "src" / "qr_detector").is_dir():
            return str(candidate)
    return "../qr_shield_core"


os.environ.setdefault("API_KEY", TEST_API_KEY)
os.environ.setdefault("AUTH_DISABLED", "false")
os.environ.setdefault("QR_SHIELD_CORE_PATH", _guess_core_path())
os.environ.setdefault("UPLOAD_DIR", str(_STORAGE_DIR / "uploads"))
os.environ.setdefault("REPORT_DIR", str(_STORAGE_DIR / "reports"))
os.environ.setdefault("HISTORY_FILE", str(_STORAGE_DIR / "history.json"))
os.environ.setdefault("LOG_FILE", str(_STORAGE_DIR / "logs" / "backend.log"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from app.services.qr_service import qr_service  # noqa: E402
from app.services.risk_service import risk_service  # noqa: E402
from app.services.url_service import url_service  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_STORAGE_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture(scope="session")
def engine_available() -> bool:
    """True only when the real QR Shield engine (src.*) could be imported -
    i.e. QR_SHIELD_CORE_PATH points at a real checkout with its
    dependencies (pyzbar, etc.) installed. Tests that need a real scan
    result are skipped when this is False rather than faking one, since
    this backend must never fall back to a second implementation."""
    return qr_service.engine_available and url_service.engine_available and risk_service.engine_available


@pytest.fixture(scope="session")
def sample_qr_image_bytes():
    """A freshly generated QR-code PNG encoding a URL, used to exercise
    the full scan pipeline. Skips dependent tests if `qrcode` isn't
    installed, rather than silently testing nothing."""
    qrcode = pytest.importorskip("qrcode")
    import io

    img = qrcode.make("https://example.com/test-page")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
