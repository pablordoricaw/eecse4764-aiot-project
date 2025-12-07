"""
utils.py

Shared logging configuration for stdout/stderr.

- Provides a simple formatter that includes timestamp, level, logger name.
- Configures root logger with separate handlers for stdout (INFO and below)
  and stderr (WARNING and above).
- Returns a module-specific logger via get_logger(name).
"""

import logging
import logging.config
from typing import Optional

# Shared logging configuration for console output
BASE_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {
            "format": "[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": "WARNING",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        # Root logger: everything goes through stdout/stderr handlers
        "": {
            "level": "INFO",
            "handlers": ["stdout", "stderr"],
        },
    },
}


def setup_base_logging(config: Optional[dict] = None) -> None:
    """
    Apply the base console logging configuration.

    Can be called once near process startup before retrieving loggers.
    """
    cfg = config or BASE_LOGGING_CONFIG
    logging.config.dictConfig(cfg)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given name.

    Usage:
        from logs_utils import setup_base_logging, get_logger

        setup_base_logging()
        logger = get_logger(__name__)
        logger.info("hello")
    """
    return logging.getLogger(name)
