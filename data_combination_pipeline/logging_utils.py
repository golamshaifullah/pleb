"""Logging helpers for the pipeline package."""

import logging

def get_logger(name: str = "data_combination_pipeline", level: int = logging.INFO) -> logging.Logger:
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
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
