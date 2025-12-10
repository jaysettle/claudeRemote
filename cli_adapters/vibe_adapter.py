#!/usr/bin/env python3
"""
Mistral Vibe CLI Adapter
Handles Vibe CLI execution and session management for Devstral models
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Callable, Optional

from config import GEMINI_TIMEOUT  # Reuse timeout config
from utils.helpers import emit_log as _emit_log, safe_cmd as _safe_cmd, parse_uuid as _parse_uuid

logger = logging.getLogger(__name__)

# Vibe CLI path (uses ~/.local/bin/vibe installed via install.sh)
VIBE_CLI_PATH = str(Path.home() / ".local" / "bin" / "vibe")
VIBE_TIMEOUT = GEMINI_TIMEOUT  # Reuse timeout


class VibeSessionNotFoundError(Exception):
    """Raised when Vibe CLI session is not found."""
    pass


async def find_latest_vibe_session_id() -> Optional[str]:
    """Find the most recent Vibe session ID from ~/.vibe/sessions directory."""
    try:
        base = Path.home() / ".vibe" / "sessions"
        if not base.exists():
            return None
        files = sorted(
            base.glob("session-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for filepath in files:
            # Extract UUID from session-<uuid>.json
            match = re.search(r"session-([a-f0-9-]{36})\.json", filepath.name)
            if match:
                uuid_str = _parse_uuid(match.group(1))
                if uuid_str:
                    return uuid_str
        return None
    except Exception as e:
        logger.error(f"Error finding Vibe session ID: {e}")
        return None


async def run_vibe_prompt(
    prompt: str,
    session_id: Optional[str] = None,
    timeout: int = VIBE_TIMEOUT,
    trace: Optional[Callable[[str, str, int], None]] = None,
) -> tuple[str, Optional[str]]:
    """
    Run Vibe CLI prompt and return (response, new_session_id).
    Supports --resume <id> and -p mode for programmatic execution.
    """
    start = time.time()
    try:
        cmd = [VIBE_CLI_PATH, "-p", prompt, "--auto-approve", "--output", "text"]
        if session_id:
            cmd.extend(["--resume", session_id])

        _emit_log(trace, "vibe.exec.start", f"cmd={_safe_cmd(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise Exception(f"Vibe CLI timed out after {timeout} seconds")

        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode() if stderr else ""
        _emit_log(
            trace,
            "vibe.exec.done",
            f"rc={process.returncode} elapsed={time.time() - start:.2f}s stdout_len={len(stdout_text)}",
        )

        if process.returncode != 0:
            if session_id and ("resume" in stderr_text.lower() or "session" in stderr_text.lower() or "not found" in stderr_text.lower()):
                _emit_log(trace, "vibe.exec.session_missing", f"session={session_id}")
                raise VibeSessionNotFoundError(session_id)
            raise Exception(f"Vibe CLI failed: {stderr_text or stdout_text or 'Unknown error'}")

        response_text = stdout_text
        new_session_id = await find_latest_vibe_session_id() if not session_id else session_id
        return response_text, new_session_id

    except VibeSessionNotFoundError:
        raise
    except Exception as e:
        _emit_log(trace, "vibe.exec.error", str(e), level=logging.ERROR)
        raise
