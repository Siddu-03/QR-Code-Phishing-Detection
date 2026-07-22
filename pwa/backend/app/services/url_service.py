"""
URL Service: orchestrates the existing QR Shield engine's URL Analyzer
module against a QR code's decoded payload. This module is ONLY a
wrapper - it must never re-implement URL heuristics itself.

Populated as part of Change 5 (Complete Remaining Processing Pipeline):
previously the backend stopped after Tamper Analysis; this wires in
`src.url_analyzer.URLAnalyzer` for the URL Analysis stage.
"""
from app.core.logger import get_logger
from app.core.qr_shield_engine import QRShieldEngineUnavailable

logger = get_logger(__name__)

_CORE_AVAILABLE = False
_URLAnalyzer = None
try:
    from src.url_analyzer.url_analyzer import URLAnalyzer as _URLAnalyzer  # type: ignore

    _CORE_AVAILABLE = True
    logger.info("Loaded real QR Shield url_analyzer engine module")
except Exception as exc:  # noqa: BLE001 - broad on purpose, this is an optional integration
    logger.warning(
        "qr_shield_core url_analyzer module not found (%s). URL analysis "
        "will be skipped until QR_SHIELD_CORE_PATH points at a valid QR "
        "Shield engine checkout.",
        exc,
    )

URL_ANALYSIS_ENGINE_NAME = "qr_shield_core.url_analyzer.URLAnalyzer"


class URLService:
    """Analyzes a decoded QR payload URL for phishing/tamper indicators."""

    def __init__(self):
        self._use_real_core = _CORE_AVAILABLE
        if self._use_real_core:
            self._analyzer = _URLAnalyzer()

    @property
    def engine_available(self) -> bool:
        return self._use_real_core

    def analyze(self, url: str):
        """Runs the real URLAnalyzer, returning its raw URLResult object
        (also what `RiskEngine.assess()` expects as `url_result`)."""
        if not self._use_real_core:
            raise QRShieldEngineUnavailable(
                "QR Shield url_analyzer engine is not available. "
                "Set QR_SHIELD_CORE_PATH to a valid QR Shield engine checkout."
            )
        try:
            return self._analyzer.analyze(url)
        except Exception as exc:  # noqa: BLE001
            logger.error("QR Shield URLAnalyzer raised an error: %s", exc)
            raise QRShieldEngineUnavailable(f"URL analysis engine failed: {exc}") from exc


url_service = URLService()
