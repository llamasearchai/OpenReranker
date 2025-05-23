import json
import logging
import sys
from typing import Any, Dict, Optional

from open_reranker.core.config import settings


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        """
        log_record: Dict[str, Any] = {}

        # Add basic log record attributes
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["name"] = record.name
        log_record["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "id",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                log_record[key] = value

        return json.dumps(log_record, default=str)


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: The logging level to use. If None, uses DEBUG in debug mode, otherwise INFO.

    Returns:
        A configured logger instance.
    """
    if level is None:
        level = "DEBUG" if settings.DEBUG else "INFO"

    # Create logger
    logger = logging.getLogger("open_reranker")
    logger.setLevel(getattr(logging, level))
    logger.propagate = False

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    return logger
