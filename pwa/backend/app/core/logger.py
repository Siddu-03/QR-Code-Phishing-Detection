"""
Centralized logging configuration for the QR Shield backend.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import get_settings

settings = get_settings()

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance. Safe to call multiple times;
    handlers are only attached once per logger name.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(settings.LOG_LEVEL.upper())
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = RotatingFileHandler(
            settings.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Falls back to console-only logging if file path is unwritable
        logger.warning("Could not attach file handler for logger '%s'", name)

    logger.propagate = False
    return logger
