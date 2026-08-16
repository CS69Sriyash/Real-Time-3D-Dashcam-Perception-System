"""
logging_config.py — Central loguru sink configuration.

Call configure_logging() ONCE, early, from each entry point (main.py,
smoke_test.py). loguru's logger is a global singleton, so every other
module's `from loguru import logger` picks up this configuration
automatically — they never configure a sink themselves, which is what
keeps the format/level consistent across the whole codebase instead of
each module doing its own thing.
"""

import sys

from loguru import logger

from config import LOG_LEVEL

_LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)


def configure_logging(level: str | None = None) -> None:
    """Reset loguru to a single stderr sink with our format/level.

    `level` overrides config.LOG_LEVEL for this run (used by --verbose).
    """
    logger.remove()  # drop loguru's default sink so level/format aren't set twice
    logger.add(sys.stderr, level=level or LOG_LEVEL, format=_LOG_FORMAT, colorize=True)
