"""Logging helpers for the pipeline package."""

import logging
import os
from datetime import datetime
from pathlib import Path

_LOG_FILE: Path | None = None


def _default_log_file() -> Path:
    """Return the default log file path under the logs directory."""
    global _LOG_FILE
    if _LOG_FILE is not None:
        return _LOG_FILE
    log_dir = Path(os.environ.get("PLEB_LOG_DIR", "logs")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = log_dir / f"pleb_{ts}.log"
    return _LOG_FILE

def get_logger(name: str = "pleb", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger for pipeline modules.

    The logger uses a simple stream handler with a timestamped format and avoids
    adding duplicate handlers on repeated calls.

    Args:
        name: Logger name to retrieve or create.
        level: Logging level to set on the logger.

    Returns:
        A configured :class:`logging.Logger` instance.

    Notes:
        This helper mutates global logger state. Call it early in module import
        to ensure consistent formatting across modules.
    """
    logger = logging.getLogger(name)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(_default_log_file())
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger
