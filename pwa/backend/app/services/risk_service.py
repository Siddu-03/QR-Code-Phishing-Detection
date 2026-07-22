"""
Risk Service: orchestrates the existing QR Shield engine's Risk
Assessment module, combining QR-detection, tamper-analysis, and
URL-analysis signals into a single weighted risk verdict. This module
is ONLY a wrapper - it must never re-implement risk scoring itself.

Populated as part of Change 5 (Complete Remaining Processing Pipeline).
"""
from typing import Any, Dict, Optional

from app.core.logger import get_logger
from app.core.qr_shield_engine import QRShieldEngineUnavailable

logger = get_logger(__name__)

_CORE_AVAILABLE = False
_RiskEngine = None
try:
    from src.risk_assessment.risk_engine import RiskEngine as _RiskEngine  # type: ignore

    _CORE_AVAILABLE = True
    logger.info("Loaded real QR Shield risk_assessment engine module")
except Exception as exc:  # noqa: BLE001 - broad on purpose, this is an optional integration
    logger.warning(
        "qr_shield_core risk_assessment module not found (%s). Risk "
        "assessment will be skipped until QR_SHIELD_CORE_PATH points at a "
        "valid QR Shield engine checkout.",
        exc,
    )

RISK_ENGINE_NAME = "qr_shield_core.risk_assessment.RiskEngine"


class RiskService:
    """Combines QR/tamper/URL signals into a final weighted risk verdict."""

    def __init__(self):
        self._use_real_core = _CORE_AVAILABLE
        if self._use_real_core:
            self._engine = _RiskEngine()

    @property
    def engine_available(self) -> bool:
        return self._use_real_core

    def assess(
        self,
        detection_result: Dict[str, Any],
        image_id: Optional[str] = None,
        tamper_result: Any = None,
        url_result: Any = None,
    ):
        """Runs the real RiskEngine, returning its raw RiskResult object."""
        if not self._use_real_core:
            raise QRShieldEngineUnavailable(
                "QR Shield risk_assessment engine is not available. "
                "Set QR_SHIELD_CORE_PATH to a valid QR Shield engine checkout."
            )
        try:
            return self._engine.assess(
                detection_result,
                image_id=image_id,
                tamper_result=tamper_result,
                url_result=url_result,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("QR Shield RiskEngine raised an error: %s", exc)
            raise QRShieldEngineUnavailable(f"Risk assessment engine failed: {exc}") from exc


risk_service = RiskService()
