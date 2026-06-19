"""Central logging for violation engines."""

from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    level_name = os.environ.get("VL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
