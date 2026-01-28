"""Logging helpers with contextual fields."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s [rank=%(rank)s client=%(client_id)s]: %(message)s",
    )
    logging.getLogger().addFilter(_ContextFilter())


def get_logger(name: str, rank: Optional[int] = None, client_id: Optional[str] = None) -> logging.LoggerAdapter:
    base = logging.getLogger(name)
    return logging.LoggerAdapter(base, {"rank": rank if rank is not None else 0, "client_id": client_id or "-"})


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "rank"):
            record.rank = 0
        if not hasattr(record, "client_id"):
            record.client_id = "-"
        return True
