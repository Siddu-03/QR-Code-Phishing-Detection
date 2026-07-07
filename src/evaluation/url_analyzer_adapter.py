"""
url_analyzer_adapter.py
========================
Evaluation Framework Update — optional integration point for the future
**URL Analyzer** module (not yet implemented anywhere in the codebase).

Why this file exists
---------------------
The evaluation framework must benchmark the URL Analyzer "once integrated"
without anyone editing ``src/evaluation/`` again on that day, and without
this framework ever importing, modifying, or depending on
``src.risk_assessment``, ``src.tamper_analysis``, or any other pipeline
package for this purpose (per the Evaluation Framework Update's strict
file-boundary rules). This module is the single, isolated seam where that
optional dependency lives.

Expected future interface (assumption — see project audit)
------------------------------------------------------------
Every other Week-1..4 module in this codebase follows the convention
``src/<module_name>/<module_name>.py`` with a module-level convenience
function (``detect_qr()``, ``assess()``, ``load_image()``). This adapter
therefore looks, in order, for:

1. ``src.url_analyzer.url_analyzer.analyze_url(url: str) -> Any``
2. ``src.url_analyzer.url_analyzer.assess(url: str) -> Any``
3. A class ``URLAnalyzer`` in that same module, instantiated with no
   arguments, exposing ``.analyze(url: str)`` or ``.assess(url: str)``.

Whatever is returned (a dataclass, a plain object, or a ``dict``) is
normalised via :func:`_extract` using a small set of alias field names per
signal (e.g. ``https`` is read from ``is_https``, ``https``, or
``uses_https`` — whichever the real object exposes), so small naming
differences in the eventual real module should not require editing this
adapter. If the real interface differs more substantially, only this one
file needs updating — no other evaluation module needs to change.

If the module cannot be imported at all (the normal case today), or the
call raises, :func:`run_url_analysis` returns ``{"available": False, ...}``
and the rest of the evaluation framework skips URL-specific metrics,
columns, and plots entirely — see ``evaluate_dataset.py`` and
``utils.URL_SIGNAL_KEYS``.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from dataclasses import is_dataclass, asdict
from typing import Any, Callable, Optional

logger = logging.getLogger("evaluation.url_analyzer_adapter")

_CANDIDATE_MODULE = "src.url_analyzer.url_analyzer"
_CANDIDATE_FUNCTIONS = ("analyze_url", "assess")
_CANDIDATE_CLASS = "URLAnalyzer"
_CANDIDATE_METHODS = ("analyze", "assess")

#: Alias field names tried, in order, for each normalised URL signal.
#: See module docstring — this is the adapter's one tunable surface.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "decoded_url": ("decoded_url", "url", "raw_url"),
    "url_valid": ("url_valid", "is_valid", "valid"),
    "https": ("is_https", "https", "uses_https"),
    "scheme": ("scheme", "url_scheme"),
    "domain": ("domain", "registered_domain"),
    "subdomain": ("subdomain",),
    "tld": ("tld", "top_level_domain"),
    "port": ("port",),
    "path": ("path", "url_path"),
    "query_parameters": ("query_parameters", "query_params", "query"),
    "fragment": ("fragment",),
    "contains_ip": ("contains_ip", "is_ip_address", "ip_address_detected"),
    "is_shortener": ("is_shortener", "shortener_detected", "url_shortener"),
    "is_homograph": ("is_homograph", "homograph_detected"),
    "entropy_score": ("entropy_score", "high_entropy_score", "entropy"),
    "suspicious_keywords": ("suspicious_keywords", "keyword_matches"),
    "suspicious_tld": ("suspicious_tld", "is_suspicious_tld"),
    "overall_url_risk": ("overall_url_risk", "url_risk_score", "risk_score"),
    "url_risk_confidence": ("url_risk_confidence", "risk_confidence", "confidence"),
    "url_risk_level": ("url_risk_level", "risk_level"),
}

#: Threshold applied to a normalised 0-1 ``overall_url_risk`` score when the
#: real module doesn't also supply a categorical ``url_risk_level``.
_DEFAULT_RISK_SCORE_THRESHOLD = 0.5

#: Categorical risk-level strings treated as "malicious" when present.
_MALICIOUS_LEVEL_STRINGS = {"HIGH_RISK", "HIGH", "SUSPICIOUS", "MALICIOUS"}

# Per-process cache: avoid retrying a failed import on every single image.
_entry_point_cache: dict[str, Any] = {"tried": False, "callable": None}


def _resolve_entry_point() -> Optional[Callable[[str], Any]]:
    """Locate a callable ``(url: str) -> Any`` for the future URL Analyzer.

    Cached per worker process after the first attempt (successful or not),
    so repeated failures across a large dataset cost one import attempt,
    not one per image.
    """
    if _entry_point_cache["tried"]:
        return _entry_point_cache["callable"]

    _entry_point_cache["tried"] = True
    try:
        module = importlib.import_module(_CANDIDATE_MODULE)
    except ImportError:
        logger.debug(
            "URL Analyzer module (%s) not found — evaluation will run without "
            "URL-analysis metrics.",
            _CANDIDATE_MODULE,
        )
        return None

    for fn_name in _CANDIDATE_FUNCTIONS:
        candidate = getattr(module, fn_name, None)
        if callable(candidate):
            logger.info("URL Analyzer detected: %s.%s()", _CANDIDATE_MODULE, fn_name)
            _entry_point_cache["callable"] = candidate
            return candidate

    cls = getattr(module, _CANDIDATE_CLASS, None)
    if cls is not None:
        try:
            instance = cls()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Found %s but could not instantiate it: %s", _CANDIDATE_CLASS, exc)
            return None
        for method_name in _CANDIDATE_METHODS:
            method = getattr(instance, method_name, None)
            if callable(method):
                logger.info(
                    "URL Analyzer detected: %s.%s().%s()",
                    _CANDIDATE_MODULE, _CANDIDATE_CLASS, method_name,
                )
                _entry_point_cache["callable"] = method
                return method

    logger.warning(
        "URL Analyzer module %s was importable but exposed none of the expected "
        "entry points (%s, %s, or %s.%s). Update _CANDIDATE_* in "
        "url_analyzer_adapter.py to match its real interface.",
        _CANDIDATE_MODULE, _CANDIDATE_FUNCTIONS, _CANDIDATE_CLASS, _CANDIDATE_METHODS,
    )
    return None


def is_url_analyzer_available() -> bool:
    """Return True if a callable URL Analyzer entry point was found in this process."""
    return _resolve_entry_point() is not None


def _get_any(source: Any, aliases: tuple[str, ...]) -> Any:
    """Read the first present attribute/key in *aliases* from *source*."""
    if source is None:
        return None
    if isinstance(source, dict):
        for key in aliases:
            if key in source:
                return source[key]
        return None
    for key in aliases:
        if hasattr(source, key):
            return getattr(source, key)
    return None


def _extract(raw: Any) -> dict[str, Any]:
    """Normalise whatever the URL Analyzer returned into a flat, aliased dict."""
    if is_dataclass(raw) and not isinstance(raw, type):
        raw = asdict(raw)
    elif hasattr(raw, "to_dict") and callable(raw.to_dict):
        raw = raw.to_dict()

    return {field: _get_any(raw, aliases) for field, aliases in _FIELD_ALIASES.items()}


def _predicted_malicious(signals: dict[str, Any]) -> Optional[bool]:
    """Derive a boolean malicious/benign prediction from whatever signals are present."""
    level = signals.get("url_risk_level")
    if isinstance(level, str):
        return level.upper() in _MALICIOUS_LEVEL_STRINGS

    score = signals.get("overall_url_risk")
    if isinstance(score, (int, float)) and not math.isnan(score):
        # Normalise a possible 0-100 scale down to 0-1 before thresholding.
        normalised = score / 100.0 if score > 1.0 else float(score)
        return normalised >= _DEFAULT_RISK_SCORE_THRESHOLD

    # Fall back to any strongly-indicative boolean flag.
    for flag in ("is_homograph", "is_shortener", "contains_ip", "suspicious_tld"):
        if signals.get(flag) is True:
            return True
    if any(k in signals and signals[k] is not None for k in ("is_homograph", "is_shortener")):
        return False
    return None


def run_url_analysis(url: str | None) -> dict[str, Any]:
    """Run the future URL Analyzer on *url*, if it is available.

    Never raises. Always returns a dict with at least the key
    ``"available"``. When unavailable (module absent, no callable entry
    point, or no decoded URL to analyse), all other keys are omitted so
    that callers can trivially detect and skip URL-specific processing
    (``if not result["available"]: continue``) — this is also what keeps
    JSON output backward compatible when the module doesn't exist.

    Parameters
    ----------
    url : str | None
        The decoded QR payload, if any. ``None`` when no QR code was
        detected/decoded for the current image.

    Returns
    -------
    dict[str, Any]
        ``{"available": False}`` if analysis did not run, or
        ``{"available": True, "elapsed_ms": float, "predicted_malicious":
        bool | None, **normalised_signals}`` on success (``error`` key
        present instead if the call raised).
    """
    if not url:
        return {"available": False, "reason": "no_decoded_url"}

    entry_point = _resolve_entry_point()
    if entry_point is None:
        return {"available": False, "reason": "module_not_found"}

    t0 = time.perf_counter()
    try:
        raw_result = entry_point(url)
    except Exception as exc:  # noqa: BLE001 — analyzer must never crash a worker
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.warning("URL Analyzer raised for %r: %s", url, exc)
        return {
            "available": True,
            "elapsed_ms": elapsed_ms,
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    signals = _extract(raw_result)
    signals["available"] = True
    signals["elapsed_ms"] = elapsed_ms
    signals["error"] = None
    signals["predicted_malicious"] = _predicted_malicious(signals)
    return signals