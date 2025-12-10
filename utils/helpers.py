#!/usr/bin/env python3
"""
Helper Utilities
Common helper functions for logging, formatting, and parsing
"""

import logging
import shlex
import re
import uuid
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def make_trace_logger() -> tuple[str, Callable[[str, str, int], None]]:
    """Create a per-request trace logger with a short id."""
    trace_id = uuid.uuid4().hex[:8]

    def log(stage: str, message: str, level: int = logging.INFO):
        logger.log(level, f"[{trace_id}] {stage} | {message}")

    return trace_id, log

def emit_log(log_fn: Optional[Callable[[str, str, int], None]], stage: str, message: str, level: int = logging.INFO):
    """Emit a log line using the trace logger if provided."""
    if log_fn:
        log_fn(stage, message, level)
    else:
        logger.log(level, f"{stage} | {message}")

def safe_cmd(cmd: list[str]) -> str:
    """Return a shell-safe string for logging."""
    return " ".join(shlex.quote(str(part)) for part in cmd)

def truncate(text: str, limit: int = 200) -> str:
    """Truncate long text for logs."""
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"

def parse_uuid(candidate: str) -> Optional[str]:
    """Return the UUID string if candidate matches a UUID format."""
    if re.match(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", candidate):
        return candidate
    return None
