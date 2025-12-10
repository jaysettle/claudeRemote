#!/usr/bin/env python3
"""
Codex CLI Adapter
Handles Codex CLI execution, session management, and streaming
"""

import asyncio
import json
import logging
import tempfile
import time
import uuid
import re
from pathlib import Path
from typing import AsyncGenerator, Callable, List, Optional

from config import CODEX_CLI_PATH, CODEX_TIMEOUT, MCP_SYSTEM_PROMPT
from utils.helpers import emit_log as _emit_log, safe_cmd as _safe_cmd, parse_uuid as _parse_uuid
from session_manager import set_session_id, clear_session_id

logger = logging.getLogger(__name__)


class CodexSessionNotFoundError(Exception):
    """Raised when Codex CLI session is not found."""
    pass

async def find_latest_codex_session_id() -> Optional[str]:
    """Find the most recent Codex session ID from ~/.codex/sessions directory."""
    try:
        sessions_dir = Path.home() / ".codex" / "sessions"
        if not sessions_dir.exists():
            return None

        files = sorted(
            sessions_dir.rglob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for filepath in files:
            # Filenames contain a trailing session UUID after the timestamp
            match = re.search(r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", filepath.name)
            if match:
                uuid_str = _parse_uuid(match.group(1))
                if uuid_str:
                    return uuid_str
        return None
    except Exception as e:
        logger.error(f"Error finding Codex session ID: {e}")
        return None

def normalize_codex_response(text: str) -> str:
    """
    Codex can emit JSON with follow_ups. If JSON, extract the primary message string.
    Fallback: return original text.
    """
    try:
        data = json.loads(text)
        # If it's already a string, return it
        if isinstance(data, str):
            return data
        # If it's a dict with a string in known keys, return that
        if isinstance(data, dict):
            for key in ("message", "content", "text", "output", "response"):
                if isinstance(data.get(key), str):
                    return data[key]
            # If dict has a single string value, return it
            string_values = [v for v in data.values() if isinstance(v, str)]
            if len(string_values) == 1:
                return string_values[0]
            # If dict has a list of strings under follow_ups, format nicely
            if isinstance(data.get("follow_ups"), list):
                return "\n".join(f"- {s}" for s in data["follow_ups"] if isinstance(s, str))
        # If it's a list of strings, join them
        if isinstance(data, list):
            strings = [str(x) for x in data]
            return "\n".join(strings)
    except Exception:
        pass
    return text

def is_followup_prompt(text: str) -> bool:
    """Detect Open WebUI follow-up generator prompt (works with custom prompts)."""
    # Check for JSON follow_ups format requirement - works with any prompt that asks for JSON
    return '"follow_ups"' in text

def extract_followups(text: str) -> List[str]:
    """Extract follow-up questions from a response that may be JSON or bullets."""
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("follow_ups"), list):
            return [str(x) for x in data["follow_ups"] if isinstance(x, str)]
        if isinstance(data, list):
            return [str(x) for x in data if isinstance(x, str)]
        if isinstance(data, dict):
            # Flatten any string values
            vals = [v for v in data.values() if isinstance(v, str)]
            if vals:
                return vals
    except Exception:
        pass

    # Fallback: parse bullet lines
    lines = text.splitlines()
    followups = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ")):
            line = line[2:].strip()
        followups.append(line)
    return followups

async def run_codex_prompt(
    prompt: str,
    session_id: Optional[str] = None,
    timeout: int = CODEX_TIMEOUT,
    trace: Optional[Callable[[str, str, int], None]] = None,
    followup_prompt: bool = False,
) -> tuple[str, Optional[str]]:
    """
    Run Codex CLI prompt (exec mode) and return (response, new_session_id).
    If session_id is provided, resumes that session.
    Raises CodexSessionNotFoundError if session doesn't exist (caller should retry with history).
    """
    start = time.time()
    output_file = Path(tempfile.gettempdir()) / f"codex_out_{uuid.uuid4().hex}.txt"
    try:
        # Build command
        cmd = [CODEX_CLI_PATH, "exec", "--color", "never", "--output-last-message", str(output_file), "--skip-git-repo-check"]
        # Align with Claude's permissive mode to avoid interactive prompts in the bridge context.
        cmd.append("--dangerously-bypass-approvals-and-sandbox")

        if session_id:
            cmd.extend(["resume", session_id, prompt])
        else:
            cmd.append(prompt)

        _emit_log(trace, "codex.exec.start", f"cmd={_safe_cmd(cmd)}")
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
            raise Exception(f"Codex CLI timed out after {timeout} seconds")

        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode() if stderr else ""
        _emit_log(
            trace,
            "codex.exec.done",
            f"rc={process.returncode} elapsed={time.time() - start:.2f}s stdout_len={len(stdout_text)}",
        )

        if process.returncode != 0:
            # Session not found detection
            if session_id and ("session" in stderr_text.lower() or "resume" in stderr_text.lower() or "not found" in stderr_text.lower()):
                _emit_log(trace, "codex.exec.session_missing", f"session={session_id}")
                raise CodexSessionNotFoundError(session_id)
            raise Exception(f"Codex CLI failed: {stderr_text or stdout_text or 'Unknown error'}")

        # Prefer the output-last-message file for the assistant's reply
        response_text = ""
        if output_file.exists():
            response_text = output_file.read_text(encoding="utf-8", errors="ignore").strip()
            _emit_log(trace, "codex.exec.output_file", f"chars={len(response_text)}")
        if not response_text:
            response_text = stdout_text

        # If this was a follow-up generator, normalize follow-ups into bullets
        if followup_prompt:
            followups = extract_followups(response_text)
            if followups:
                response_text = json.dumps({"follow_ups": followups})
            else:
                response_text = normalize_codex_response(response_text)
        else:
            # Strip any structured JSON (e.g., follow_ups) and return only assistant text
            response_text = normalize_codex_response(response_text)

        new_session_id = await find_latest_codex_session_id() if not session_id else session_id
        return response_text, new_session_id

    except CodexSessionNotFoundError:
        raise
    except Exception as e:
        _emit_log(trace, "codex.exec.error", str(e), level=logging.ERROR)
        raise
    finally:
        if output_file.exists():
            try:
                output_file.unlink()
            except Exception:
                pass

async def stream_codex_incremental(
    chat_id: str,
    session_id: Optional[str],
    prompt: str,
    history_prompt: Optional[str],
    trace: Optional[Callable[[str, str, int], None]],
) -> AsyncGenerator[str, None]:
    """Stream codex output incrementally as it arrives from CLI."""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    def make_chunk(content: str, finish: bool = False):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "codex-cli",
            "choices": [{
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None
            }]
        }

    # Initial thinking indicator
    yield f"data: {json.dumps(make_chunk('...'))}\n\n"

    output_file = Path(tempfile.gettempdir()) / f"codex_out_{uuid.uuid4().hex}.txt"
    start = time.time()

    try:
        # Build command
        cmd = [CODEX_CLI_PATH, "exec", "--color", "never", "--output-last-message", str(output_file), "--skip-git-repo-check"]
        cmd.append("--dangerously-bypass-approvals-and-sandbox")

        if session_id:
            cmd.extend(["resume", session_id, prompt])
        else:
            cmd.append(prompt)

        _emit_log(trace, "codex.stream.start", f"cmd={_safe_cmd(cmd)}")

        # Start the process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Stream stdout incrementally
        full_output = []
        stderr_chunks = []

        # Read stderr in background
        async def read_stderr():
            if process.stderr:
                async for line in process.stderr:
                    stderr_chunks.append(line)

        stderr_task = asyncio.create_task(read_stderr())

        # Stream stdout as it arrives
        if process.stdout:
            async for line in process.stdout:
                try:
                    text = line.decode('utf-8', errors='ignore')
                    full_output.append(text)
                    # Send each line as a chunk
                    yield f"data: {json.dumps(make_chunk(text))}\n\n"
                except Exception as e:
                    _emit_log(trace, "codex.stream.decode_error", str(e), level=logging.WARNING)

        # Wait for process to complete
        await process.wait()
        await stderr_task

        elapsed = time.time() - start
        stderr_text = b''.join(stderr_chunks).decode('utf-8', errors='ignore') if stderr_chunks else ""
        stdout_text = ''.join(full_output)

        _emit_log(
            trace,
            "codex.stream.done",
            f"rc={process.returncode} elapsed={elapsed:.2f}s stdout_len={len(stdout_text)}",
        )

        # Handle errors
        if process.returncode != 0:
            if session_id and ("session" in stderr_text.lower() or "resume" in stderr_text.lower() or "not found" in stderr_text.lower()):
                _emit_log(trace, "codex.stream.session_missing", f"session={session_id}")
                # Retry with history
                clear_session_id("codex-cli", chat_id)
                fallback_prompt = history_prompt if history_prompt else prompt
                fallback_prompt = f"[SYSTEM]\n{MCP_SYSTEM_PROMPT}\n[END SYSTEM]\n\n{fallback_prompt}"
                _emit_log(trace, "codex.stream.retry.history", "retrying without resume after session miss")

                # Re-stream with fallback
                async for chunk in stream_codex_incremental(chat_id, None, fallback_prompt, None, trace):
                    yield chunk
                return
            else:
                error_msg = f"Codex CLI failed: {stderr_text or stdout_text or 'Unknown error'}"
                yield f"data: {json.dumps(make_chunk(error_msg))}\n\n"
                yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
                yield "data: [DONE]\n\n"
                return

        # Update session mapping
        new_session_id = await find_latest_codex_session_id() if not session_id else session_id
        if new_session_id and new_session_id != session_id:
            set_session_id("codex-cli", chat_id, new_session_id)
            _emit_log(trace, "session.map.updated", f"model=codex-cli chat_id={chat_id} session={new_session_id}")

    except Exception as e:
        error_msg = f"Error: {e}"
        _emit_log(trace, "codex.stream.error", error_msg, level=logging.ERROR)
        yield f"data: {json.dumps(make_chunk(error_msg))}\n\n"
    finally:
        if output_file.exists():
            try:
                output_file.unlink()
            except Exception:
                pass

    yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
    yield "data: [DONE]\n\n"

