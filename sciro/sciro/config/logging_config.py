"""
SCIRO Logging Configuration.

Provides consistent logging setup across all modules.
"""

import logging
import sys
from typing import Optional


_CONFIGURED = False


def setup_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """Configure logging for the SCIRO application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from settings.
        log_format: Custom log format string. If None, uses default.
    """
    global _CONFIGURED

    if _CONFIGURED:
        return

    if level is None:
        # Import here to avoid circular imports
        from sciro.config.settings import get_settings
        level = get_settings().LOG_LEVEL

    if log_format is None:
        log_format = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level, logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level, logging.INFO))
    console_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))

    # Avoid duplicate handlers on repeated calls
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("psycopg2").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with SCIRO configuration applied.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    setup_logging()
    return logging.getLogger(name)
