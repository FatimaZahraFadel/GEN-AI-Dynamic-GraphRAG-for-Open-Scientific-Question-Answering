"""
Centralized logging configuration for the Dynamic GraphRAG pipeline.
"""

import logging
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class PipelineLogger:
    """
    Wrapper around the standard Python logging module providing
    consistent formatting and log-level control across all pipeline stages.

    Attributes:
        name: Name of the logger (typically the calling module's __name__).
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        logger: Underlying Python Logger instance.
    """

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        """
        Initialize the logger with a name and verbosity level.

        Args:
            name: Logger name, usually the module or class name.
            level: Minimum severity level to emit. Defaults to INFO.
        """
        self.name = name
        self.level = level
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            self.logger.addHandler(handler)
        self.logger.setLevel(level)

    def get_logger(self) -> logging.Logger:
        """
        Return the configured Logger instance.

        Returns:
            logging.Logger: Ready-to-use logger.
        """
        return self.logger

    def set_level(self, level: int) -> None:
        """
        Dynamically change the logging verbosity.

        Args:
            level: New logging level (e.g., logging.DEBUG).
        """
        self.level = level
        self.logger.setLevel(level)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Convenience factory: create and return a configured Logger.

    Args:
        name: Logger name.
        level: Logging level. Defaults to INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return PipelineLogger(name, level).get_logger()
