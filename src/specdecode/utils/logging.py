"""Logging and measurement utilities for SpecDecode Edge."""

import logging
import sys
from rich.logging import RichHandler


def get_logger(name: str = "specdecode") -> logging.Logger:
    """Return a configured logger with rich formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
