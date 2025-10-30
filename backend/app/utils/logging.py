from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(
    *,
    level: str,
    log_dir: Path,
    file_name: str,
    max_bytes: int,
    backup_count: int,
) -> None:
    """Configure application wide logging.

    A rotating file handler is installed alongside the default stream handler so
    that operational issues can be diagnosed after the fact. Existing handlers
    are removed before configuration to avoid duplicated log entries when the
    application restarts in the same interpreter (e.g. during tests).
    """

    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        log_dir / file_name,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

