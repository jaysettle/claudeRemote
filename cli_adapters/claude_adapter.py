#!/usr/bin/env python3
"""
Claude CLI Adapter
Handles Claude CLI execution, session management, and streaming
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

from config import (
    CLAUDE_CLI_PATH, CLAUDE_TIMEOUT, CLAUDE_DISABLE_MCP,
    MCP_SYSTEM_PROMPT
)
from utils.helpers import emit_log as _emit_log, safe_cmd as _safe_cmd, truncate as _truncate, parse_uuid as _parse_uuid
from utils.mcp_loader import get_mcp_config
from session_manager import set_session_id, clear_session_id

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    """Raised when Claude CLI session is not found."""
    pass

async def find_latest_claude_session_id() -> Optional[str]:
    """Find the most recent Claude session ID from ~/.claude/projects directory."""
    try:
        projects_dir = Path.home() / ".claude" / "projects"
        if not projects_dir.exists():
            return None

        # Look for recently modified .jsonl files (UUID format only - agent-* doesn't work with --resume)
        cmd = ["find", str(projects_dir), "-name", "*.jsonl", "-mmin", "-2", "-type", "f"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()

        if stdout:
            files = [f for f in stdout.decode().strip().split('\n') if f]
            # Find first UUID format file (agent-* files don't work with --resume)
            for filepath in files:
                filename = Path(filepath).stem
                uuid_str = _parse_uuid(filename)
                if uuid_str:
                    return uuid_str
        return None
    except Exception as e:
        logger.error(f"Error finding session ID: {e}")
        return None

def is_error_response(text: str) -> bool:
    """Check if response is an error (Claude CLI puts errors in stdout with exit code 0)"""
    # Only check for CLI-specific error patterns, not generic "Error:" which appears in normal content
    error_patterns = [
        "No conversation found with session ID:",
        "not a valid UUID",
        "Error: Invalid",  # Claude CLI specific errors start with "Error: Invalid..."
        "Error: Could not",  # Another Claude CLI pattern
    ]
    return any(pattern in text for pattern in error_patterns)

async def run_claude_prompt(
    prompt: str,
    session_id: Optional[str] = None,
    timeout: int = CLAUDE_TIMEOUT,
    trace: Optional[Callable[[str, str, int], None]] = None,
) -> tuple[str, Optional[str]]:
    """
    Run Claude CLI prompt and return (response, new_session_id).
    If session_id is provided, resumes that session.
    Raises SessionNotFoundError if session doesn't exist (caller should retry with history).
    History is limited to last 10 messages to avoid "Argument list too long" errors.
    """
    start = time.time()
    try:
        cmd = [CLAUDE_CLI_PATH]
        if session_id:
            cmd.extend(["--resume", session_id])
        # Add MCP config if available (dynamically loaded from ~/.claude.json) unless disabled
        if CLAUDE_DISABLE_MCP:
            cmd.extend(["--mcp-config", '{"mcpServers":{}}', "--strict-mcp-config"])
        else:
            mcp_config = get_mcp_config()
            if mcp_config:
                cmd.extend(["--mcp-config", json.dumps(mcp_config)])
        cmd.extend(["-p", prompt, "--dangerously-skip-permissions"])

        # Log the exact prompt sent to Claude (trace-aware so it lands in per-request logs)
        _emit_log(trace, "claude.prompt.full", f"prompt={_truncate(prompt, limit=4000)}", level=logging.INFO)

        _emit_log(trace, "claude.exec.start", f"cmd={_safe_cmd(cmd)}")
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
            raise Exception(f"Claude CLI timed out after {timeout} seconds")

        response = stdout.decode().strip()
        stderr_text = stderr.decode() if stderr else ""
        _emit_log(
            trace,
            "claude.exec.done",
            f"rc={process.returncode} elapsed={time.time() - start:.2f}s resp_len={len(response)}",
        )

        # Check for errors (Claude CLI returns exit 0 even on errors, puts error in stdout)
        if process.returncode != 0 or is_error_response(response):
            error_msg = stderr_text or response or "Unknown error"
            # Raise specific exception for session not found - caller will retry with history
            if session_id and ("session" in error_msg.lower() or "not found" in error_msg.lower() or "No conversation found" in response):
                _emit_log(trace, "claude.exec.session_missing", f"session={session_id}")
                raise SessionNotFoundError(session_id)
            raise Exception(f"Claude CLI failed: {error_msg}")

        new_session_id = await find_latest_claude_session_id() if not session_id else session_id
        return response, new_session_id

    except SessionNotFoundError:
        raise  # Re-raise for caller to handle
    except Exception as e:
        _emit_log(trace, "claude.exec.error", str(e), level=logging.ERROR)
        raise

async def stream_claude_incremental(
    chat_id: str,
    session_id: Optional[str],
    prompt: str,
    history_prompt: Optional[str],
    trace: Optional[Callable[[str, str, int], None]],
) -> AsyncGenerator[str, None]:
    """Stream Claude CLI output incrementally using --output-format stream-json."""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    def make_chunk(content: str, finish: bool = False):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "claude-cli",
            "choices": [{
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None
            }]
        }

    # Initial thinking indicator
    yield f"data: {json.dumps(make_chunk('...'))}\n\n"

    start = time.time()
    accumulated_text = []
    emitted_text = False
    first_event_at = None

    try:
        # Build command with stream-json output
        cmd = [CLAUDE_CLI_PATH]
        if session_id:
            cmd.extend(["--resume", session_id])

        # Add MCP config if available (disabled by default to avoid slow MCP startup)
        if CLAUDE_DISABLE_MCP:
            cmd.extend(["--mcp-config", '{"mcpServers":{}}', "--strict-mcp-config"])
        else:
            mcp_config = get_mcp_config()
            if mcp_config:
                cmd.extend(["--mcp-config", json.dumps(mcp_config)])

        # Claude CLI requires --verbose when using --output-format=stream-json with --print/-p
        cmd.extend([
            "--verbose",
            "-p", prompt,
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
        ])

        _emit_log(trace, "claude.prompt.full", f"prompt={_truncate(prompt, limit=4000)}", level=logging.INFO)
        _emit_log(trace, "claude.stream.start", f"cmd={_safe_cmd(cmd)}")

        # Start the process (with 4MB buffer limit for image data)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=4*1024*1024  # 4MB instead of default 64KB
        )

        # Stream stdout incrementally (NDJSON format)
        stderr_chunks = []
        event_count = 0

        # Read stderr in background
        async def read_stderr():
            if process.stderr:
                async for line in process.stderr:
                    stderr_chunks.append(line)

        stderr_task = asyncio.create_task(read_stderr())

        # Stream stdout as NDJSON events arrive
        if process.stdout:
            async for line in process.stdout:
                try:
                    line_text = line.decode('utf-8', errors='ignore').strip()
                    if not line_text:
                        continue

                    _emit_log(trace, "claude.stream.raw", _truncate(line_text), level=logging.DEBUG)
                    event = json.loads(line_text)
                    event_type = event.get('type', 'unknown')
                    event_count += 1

                    if first_event_at is None:
                        first_event_at = time.time()
                        _emit_log(
                            trace,
                            "claude.stream.first_event",
                            f"type={event_type} ttfb={first_event_at - start:.2f}s",
                            level=logging.INFO,
                        )

                    _emit_log(
                        trace,
                        "claude.stream.event",
                        f"#{event_count} type={event_type} keys={list(event.keys())}",
                        level=logging.DEBUG,
                    )

                    # Handle different event types
                    if event_type == 'message':
                        # Extract text content from message events
                        content = event.get('content', '')
                        for text in _iter_text_from_content(content):
                            accumulated_text.append(text)
                            yield f"data: {json.dumps(make_chunk(text))}\n\n"
                            emitted_text = True
                            _emit_log(
                                trace,
                                "claude.stream.text",
                                f"chars={len(text)} preview={_truncate(text)}",
                                level=logging.DEBUG,
                            )

                    elif event_type == 'assistant':
                        # Claude now emits assistant events; handle same as message
                        message_obj = event.get('message', {})
                        content = message_obj.get('content', message_obj.get('text', ''))
                        for text in _iter_text_from_content(content):
                            accumulated_text.append(text)
                            yield f"data: {json.dumps(make_chunk(text))}\n\n"
                            emitted_text = True
                            _emit_log(
                                trace,
                                "claude.stream.assistant_text",
                                f"chars={len(text)} preview={_truncate(text)}",
                                level=logging.DEBUG,
                            )

                    elif event_type == 'tool_use':
                        tool_name = event.get('name') or event.get('tool') or 'unknown'
                        _emit_log(trace, "claude.stream.tool_use", f"tool={tool_name} event=#{event_count}")

                    elif event_type == 'tool_result':
                        summary = event.get('result') or event.get('content') or ''
                        if isinstance(summary, dict):
                            summary = f"keys={list(summary.keys())}"
                        _emit_log(trace, "claude.stream.tool_result", f"event=#{event_count} summary={_truncate(str(summary))}")

                    elif event_type == 'result':
                        _emit_log(trace, "claude.stream.result", "final result received")
                        # Fallback: some responses only include final result text
                        result_text = event.get('result')
                        if isinstance(result_text, str) and result_text and not emitted_text:
                            accumulated_text.append(result_text)
                            yield f"data: {json.dumps(make_chunk(result_text))}\n\n"
                            _emit_log(
                                trace,
                                "claude.stream.result_text",
                                f"chars={len(result_text)} preview={_truncate(result_text)}",
                                level=logging.DEBUG,
                            )

                except json.JSONDecodeError as e:
                    _emit_log(trace, "claude.stream.json_error", f"line={line_text[:100]} error={e}", level=logging.WARNING)
                except Exception as e:
                    _emit_log(trace, "claude.stream.decode_error", str(e), level=logging.WARNING)

        # Wait for process to complete
        await process.wait()
        await stderr_task

        elapsed = time.time() - start
        stderr_text = b''.join(stderr_chunks).decode('utf-8', errors='ignore') if stderr_chunks else ""
        full_text = ''.join(accumulated_text)

        _emit_log(
            trace,
            "claude.stream.done",
            f"rc={process.returncode} elapsed={elapsed:.2f}s events={event_count} text_len={len(full_text)}",
        )

        # Handle errors
        if process.returncode != 0:
            if session_id and ("session" in stderr_text.lower() or "resume" in stderr_text.lower() or "not found" in stderr_text.lower()):
                _emit_log(trace, "claude.stream.session_missing", f"session={session_id}")
                # Retry with history
                clear_session_id("claude-cli", chat_id)
                fallback_prompt = history_prompt if history_prompt else prompt
                fallback_prompt = f"[SYSTEM]\n{MCP_SYSTEM_PROMPT}\n[END SYSTEM]\n\n{fallback_prompt}"
                _emit_log(trace, "claude.stream.retry.history", "retrying without resume after session miss")

                # Re-stream with fallback
                async for chunk in stream_claude_incremental(chat_id, None, fallback_prompt, None, trace):
                    yield chunk
                return
            else:
                error_msg = f"Claude CLI failed: {stderr_text or 'Unknown error'}"
                yield f"data: {json.dumps(make_chunk(error_msg))}\n\n"
                yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
                yield "data: [DONE]\n\n"
                return

        # Update session mapping
        new_session_id = await find_latest_claude_session_id() if not session_id else session_id
        if new_session_id and new_session_id != session_id:
            set_session_id("claude-cli", chat_id, new_session_id)
            _emit_log(trace, "session.map.updated", f"model=claude-cli chat_id={chat_id} session={new_session_id}")

    except Exception as e:
        error_msg = f"Error: {e}"
        _emit_log(trace, "claude.stream.error", error_msg, level=logging.ERROR)
        yield f"data: {json.dumps(make_chunk(error_msg))}\n\n"

    yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
    yield "data: [DONE]\n\n"

