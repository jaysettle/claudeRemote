#!/usr/bin/env python3
"""
Gemini CLI Adapter
Handles Gemini CLI execution and session management
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Callable, Optional

from config import GEMINI_CLI_PATH, GEMINI_TIMEOUT
from utils.helpers import emit_log as _emit_log, safe_cmd as _safe_cmd, parse_uuid as _parse_uuid

logger = logging.getLogger(__name__)


class GeminiSessionNotFoundError(Exception):
    """Raised when Gemini CLI session is not found."""
    pass

async def find_latest_gemini_session_id() -> Optional[str]:
    """Find the most recent Gemini session ID from ~/.gemini/tmp/*/chats directory."""
    try:
        base = Path.home() / ".gemini" / "tmp"
        if not base.exists():
            return None
        files = sorted(
            base.glob("*/*/session-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for filepath in files:
            match = re.search(r"session-[0-9T\-]+-([a-f0-9-]{36})\.json", filepath.name)
            if match:
                uuid_str = _parse_uuid(match.group(1))
                if uuid_str:
                    return uuid_str
        return None
    except Exception as e:
        logger.error(f"Error finding Gemini session ID: {e}")
        return None

async def run_gemini_prompt(
    prompt: str,
    session_id: Optional[str] = None,
    timeout: int = GEMINI_TIMEOUT,
    trace: Optional[Callable[[str, str, int], None]] = None,
) -> tuple[str, Optional[str]]:
    """
    Run Gemini CLI prompt and return (response, new_session_id).
    Supports --resume <id> and YOLO mode for auto-approvals.
    """
    start = time.time()
    try:
        cmd = [GEMINI_CLI_PATH, "--yolo", "--output-format", "text"]
        if session_id:
            cmd.extend(["--resume", session_id, prompt])
        else:
            cmd.append(prompt)

        _emit_log(trace, "gemini.exec.start", f"cmd={_safe_cmd(cmd)}")
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
            raise Exception(f"Gemini CLI timed out after {timeout} seconds")

        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode() if stderr else ""
        _emit_log(
            trace,
            "gemini.exec.done",
            f"rc={process.returncode} elapsed={time.time() - start:.2f}s stdout_len={len(stdout_text)}",
        )

        if process.returncode != 0:
            if session_id and ("resume" in stderr_text.lower() or "session" in stderr_text.lower() or "not found" in stderr_text.lower()):
                _emit_log(trace, "gemini.exec.session_missing", f"session={session_id}")
                raise GeminiSessionNotFoundError(session_id)
            raise Exception(f"Gemini CLI failed: {stderr_text or stdout_text or 'Unknown error'}")

        response_text = stdout_text
        new_session_id = await find_latest_gemini_session_id() if not session_id else session_id
        return response_text, new_session_id

    except GeminiSessionNotFoundError:
        raise
    except Exception as e:
        _emit_log(trace, "gemini.exec.error", str(e), level=logging.ERROR)
        raise

