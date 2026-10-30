"""Logging setup shared by every module and script."""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from churn.config.settings import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def _configure_logging() -> None:
    """Attach a console handler and a rotating file handler, once."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    level_name = os.environ.get("CHURN_LOG_LEVEL", settings.logging.level).upper()
    level = getattr(logging, level_name, logging.INFO)

    base_logger = logging.getLogger("churn")
    base_logger.setLevel(level)
    base_logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    base_logger.addHandler(console)

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    base_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the 'churn' namespace."""
    _configure_logging()
    if not name.startswith("churn"):
        name = f"churn.{name}"
    return logging.getLogger(name)