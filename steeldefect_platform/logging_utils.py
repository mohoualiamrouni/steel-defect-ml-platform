"""Minimal structured logging for the SteelDefect platform.

Replaces ad-hoc ``print`` calls with a single configured ``logging.Logger`` so
that Databricks job logs carry timestamps, levels and the task name — the bare
minimum for production observability. Level is taken from ``STEELDEFECT_LOG_LEVEL``
(default ``INFO``). Idempotent: re-importing never duplicates handlers.
"""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_configured = False


def get_logger(name: str = "steeldefect") -> logging.Logger:
    """Return a process-wide configured logger (single handler, env-driven level)."""
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        level = os.environ.get("STEELDEFECT_LOG_LEVEL", "INFO").upper()
        root = logging.getLogger("steeldefect")
        if not root.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(h)
        root.setLevel(getattr(logging, level, logging.INFO))
        _configured = True
    return logger
