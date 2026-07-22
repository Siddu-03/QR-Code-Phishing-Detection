"""
Shared bootstrap for locating the existing QR Shield engine (the `src/`
package: qr_detector, tamper_analysis, url_analyzer, risk_assessment,
reporting, ...) on `sys.path`.

Every service module that wraps a piece of the real engine (qr_service,
url_service, risk_service) imports `QRShieldEngineUnavailable` and relies
on this module having already inserted QR_SHIELD_CORE_PATH onto
`sys.path` - instead of each service repeating its own copy of the same
sys.path bootstrap (previously duplicated per-module; consolidated here
as part of Change 6 cleanup).
"""
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class QRShieldEngineUnavailable(RuntimeError):
    """
    Raised when a required QR Shield engine component (anything under
    `src.*`) cannot be imported, or raises while running.

    This backend is only a wrapper around the QR Shield engine - callers
    (the API layer) must translate this into a 503 Service Unavailable
    response and must never fall back to a second, divergent
    implementation.
    """


CORE_PATH = Path(settings.QR_SHIELD_CORE_PATH).resolve()
if CORE_PATH.exists() and str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))
    logger.info("QR Shield core engine path added to sys.path: %s", CORE_PATH)
else:
    logger.warning(
        "QR Shield core path '%s' does not exist. Engine-backed endpoints "
        "will return 503 until QR_SHIELD_CORE_PATH points at a valid "
        "QR Shield engine checkout.",
        CORE_PATH,
    )
