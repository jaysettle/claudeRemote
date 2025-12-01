#!/usr/bin/env python3
"""
Claude/Codex CLI Bridge for Open WebUI
Provides an OpenAI-compatible API that proxies requests to CLI agents
Supports persistent sessions per chat thread

v1.10.0-dev - Adds Codex + Gemini CLI support and richer sequential logging
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shlex
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# File upload directory (env override for dev/prod separation)
UPLOAD_DIR = Path(os.getenv("CLAUDE_UPLOAD_DIR", Path(tempfile.gettempdir()) / "claude_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Session storage directory (env override for dev/prod separation)
SESSION_DIR = Path(os.getenv("CLAUDE_SESSION_DIR", Path(tempfile.gettempdir()) / "claude_sessions"))
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SESSION_MAP_FILE = SESSION_DIR / "session_map.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Claude CLI Bridge", version="1.9.0-dev")

# Configuration
CLAUDE_CLI_PATH = os.getenv("CLAUDE_CLI_PATH", "claude")
CLAUDE_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "600"))  # 10 minutes default
CLAUDE_PROJECT_PATH = os.getenv("CLAUDE_PROJECT_PATH", str(Path.home()))  # Project path for MCP config
CODEX_CLI_PATH = os.getenv("CODEX_CLI_PATH", str(Path.home() / ".npm-global" / "bin" / "codex"))
CODEX_TIMEOUT = int(os.getenv("CODEX_TIMEOUT", str(CLAUDE_TIMEOUT)))
GEMINI_CLI_PATH = os.getenv("GEMINI_CLI_PATH", "/usr/local/bin/gemini")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", str(CLAUDE_TIMEOUT)))
BRIDGE_PORT = int(os.getenv("CLAUDE_BRIDGE_PORT", "8000"))

SUPPORTED_MODELS = {
    "claude-cli": {"owned_by": "anthropic"},
    "codex-cli": {"owned_by": "openai-codex"},
    "gemini-cli": {"owned_by": "google-gemini"},
}

# Cache for MCP config (reloaded periodically)
_mcp_config_cache = None
_mcp_config_cache_time = 0
MCP_CONFIG_CACHE_TTL = 60  # Reload config every 60 seconds


def get_mcp_config() -> Optional[Dict[str, Any]]:
    """
    Dynamically load MCP config from Claude CLI's config file (~/.claude.json).
    Merges both user-scoped (global) and project-scoped (local) MCP servers.
    This allows adding MCP servers with 'claude mcp add' without editing the bridge.
    """
    global _mcp_config_cache, _mcp_config_cache_time

    # Return cached config if still valid
    if _mcp_config_cache is not None and (time.time() - _mcp_config_cache_time) < MCP_CONFIG_CACHE_TTL:
        return _mcp_config_cache

    try:
        claude_config_path = Path.home() / ".claude.json"
        if not claude_config_path.exists():
            logger.warning(f"Claude config not found at {claude_config_path}")
            return None

        with open(claude_config_path) as f:
            config = json.load(f)

        # Merge user-scoped (global) and project-scoped (local) MCP servers
        # User-scoped: root level "mcpServers"
        # Project-scoped: projects[path]["mcpServers"]
        mcp_servers = {}

        # Load user-scoped servers first (can be overridden by project-scoped)
        user_servers = config.get("mcpServers", {})
        mcp_servers.update(user_servers)

        # Load project-scoped servers (override user-scoped if same name)
        projects = config.get("projects", {})
        project_config = projects.get(CLAUDE_PROJECT_PATH, {})
        project_servers = project_config.get("mcpServers", {})
        mcp_servers.update(project_servers)

        if mcp_servers:
            _mcp_config_cache = {"mcpServers": mcp_servers}
            _mcp_config_cache_time = time.time()
            logger.info(f"Loaded {len(mcp_servers)} MCP server(s): {', '.join(mcp_servers.keys())}")
            return _mcp_config_cache
        else:
            logger.info(f"No MCP servers configured")
            return None

    except Exception as e:
        logger.error(f"Error loading MCP config: {e}")
        return None


# System Prompt with MCP and formatting instructions
MCP_SYSTEM_PROMPT = """You have access to MCP (Model Context Protocol) tools.
Use the available MCP tools when they are relevant to the user's request.

FORMATTING: Always use proper markdown formatting in your responses:
- Use bullet points (- or *) for lists
- Use numbered lists (1. 2. 3.) for sequences
- Use headers (## or ###) for sections
- Use code blocks (```) for code
- Use **bold** for emphasis
- Add blank lines between sections for readability
"""

# In-memory session map: model_id -> {chat_id -> session_id}
session_map: Dict[str, Dict[str, str]] = {model: {} for model in SUPPORTED_MODELS.keys()}


def _make_trace_logger() -> tuple[str, Callable[[str, str, int], None]]:
    """Create a per-request trace logger with a short id."""
    trace_id = uuid.uuid4().hex[:8]

    def log(stage: str, message: str, level: int = logging.INFO):
        logger.log(level, f"[{trace_id}] {stage} | {message}")

    return trace_id, log


def _emit_log(log_fn: Optional[Callable[[str, str, int], None]], stage: str, message: str, level: int = logging.INFO):
    """Emit a log line using the trace logger if provided."""
    if log_fn:
        log_fn(stage, message, level)
    else:
        logger.log(level, f"{stage} | {message}")


def _safe_cmd(cmd: list[str]) -> str:
    """Return a shell-safe string for logging."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _parse_uuid(candidate: str) -> Optional[str]:
    """Return the UUID string if candidate matches a UUID format."""
    if re.match(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", candidate):
        return candidate
    return None


def load_session_map():
    """Load session map from disk, upgrading legacy single-map layout if needed."""
    global session_map
    try:
        if SESSION_MAP_FILE.exists():
            with open(SESSION_MAP_FILE, 'r') as f:
                data = json.load(f)

            # Legacy format: {"chat_id": "session_id"}
            if data and all(isinstance(v, str) for v in data.values()):
                session_map["claude-cli"] = data
                logger.info(f"Loaded {len(data)} claude sessions from legacy map")
            elif isinstance(data, dict):
                for model_id, mapping in data.items():
                    if isinstance(mapping, dict) and model_id in SUPPORTED_MODELS:
                        session_map[model_id] = mapping
                logger.info(
                    "Loaded session maps: "
                    + ", ".join(f"{m}:{len(v)}" for m, v in session_map.items())
                )
            else:
                logger.warning("Session map format unrecognized, starting fresh")
    except Exception as e:
        logger.error(f"Error loading session map: {e}")
        session_map = {model: {} for model in SUPPORTED_MODELS.keys()}


def save_session_map():
    """Save session map to disk."""
    try:
        with open(SESSION_MAP_FILE, 'w') as f:
            json.dump(session_map, f)
    except Exception as e:
        logger.error(f"Error saving session map: {e}")


def get_session_id(model_id: str, chat_id: str) -> Optional[str]:
    """Lookup session id for model/chat."""
    return session_map.get(model_id, {}).get(chat_id)


def set_session_id(model_id: str, chat_id: str, session_id: str):
    """Persist session id for model/chat."""
    session_map.setdefault(model_id, {})[chat_id] = session_id
    save_session_map()


def clear_session_id(model_id: str, chat_id: str):
    """Remove a mapping for model/chat."""
    if chat_id in session_map.get(model_id, {}):
        del session_map[model_id][chat_id]
        save_session_map()


# Load sessions on startup
load_session_map()


# Pydantic models for OpenAI-compatible API
class ImageUrl(BaseModel):
    url: str

class ContentPart(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None


def get_chat_id(request: Request, messages: List[Message]) -> str:
    """
    Generate a unique chat ID from the request.
    Uses x-openwebui-chat-id header if available, otherwise hashes the first user message.
    """
    # Try to get ID from headers (Open WebUI sends this with ENABLE_FORWARD_USER_INFO_HEADERS)
    chat_id = (
        request.headers.get("x-openwebui-chat-id") or
        request.headers.get("x-chat-id") or
        request.headers.get("x-conversation-id")
    )
    if chat_id:
        return chat_id

    # Generate ID from first user message hash
    for msg in messages:
        if msg.role == "user":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            chat_id = hashlib.md5(content[:100].encode()).hexdigest()[:16]
            return chat_id

    # Fallback: generate random ID
    return str(uuid.uuid4())[:16]


def decode_base64_file(data_url: str) -> tuple[Optional[str], Optional[str], Optional[bytes]]:
    """Decode a base64 data URL and return (mime_type, extension, content_bytes)."""
    try:
        if not data_url.startswith('data:'):
            return None, None, None

        header, b64_data = data_url.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]

        ext_map = {
            'image/png': '.png', 'image/jpeg': '.jpg', 'image/jpg': '.jpg',
            'image/gif': '.gif', 'image/webp': '.webp', 'application/pdf': '.pdf',
            'text/plain': '.txt', 'text/markdown': '.md',
            'application/json': '.json', 'text/csv': '.csv',
        }
        ext = ext_map.get(mime_type, '.bin')
        content = base64.b64decode(b64_data)
        return mime_type, ext, content
    except Exception as e:
        logger.error(f"Error decoding base64 file: {e}")
        return None, None, None


def process_uploaded_file(data_url: str) -> tuple[Optional[str], Optional[str]]:
    """Process uploaded file. Returns (text_content, binary_path)."""
    mime_type, ext, content = decode_base64_file(data_url)
    if content is None:
        return None, None

    # Text-based files - return content directly
    text_mimes = ['text/plain', 'text/markdown', 'text/csv', 'application/json']
    if mime_type in text_mimes:
        try:
            text_content = content.decode('utf-8')
            logger.info(f"Read text file ({mime_type}): {len(text_content)} chars")
            return text_content, None
        except UnicodeDecodeError:
            logger.warning(f"Could not decode {mime_type} as UTF-8")

    # Binary files - save to disk
    file_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"upload_{file_id}{ext}"
    with open(file_path, 'wb') as f:
        f.write(content)
    logger.info(f"Saved binary file to {file_path}")
    return None, str(file_path)


def process_message_content(content: Union[str, List[ContentPart]]) -> tuple[str, List[str], List[str]]:
    """Process message content. Returns (text, file_contents, binary_paths)."""
    if isinstance(content, str):
        return content, [], []

    text_parts = []
    file_contents = []
    binary_paths = []

    for part in content:
        url = None

        if isinstance(part, dict):
            part_type = part.get('type', '')
            if part_type == 'text':
                text_parts.append(part.get('text', ''))
            elif part_type == 'image_url':
                image_url = part.get('image_url', {})
                url = image_url.get('url', '') if isinstance(image_url, dict) else ''
        elif hasattr(part, 'type'):
            if part.type == 'text' and part.text:
                text_parts.append(part.text)
            elif part.type == 'image_url' and part.image_url:
                url = part.image_url.url

        if url and url.startswith('data:'):
            text_content, file_path = process_uploaded_file(url)
            if text_content:
                file_contents.append(text_content)
            elif file_path:
                binary_paths.append(file_path)

    return ' '.join(text_parts), file_contents, binary_paths


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "bridge"


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]


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


class SessionNotFoundError(Exception):
    """Raised when Claude CLI session is not found."""
    pass


class CodexSessionNotFoundError(Exception):
    """Raised when Codex CLI session is not found."""
    pass


class GeminiSessionNotFoundError(Exception):
    """Raised when Gemini CLI session is not found."""
    pass


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
    """Detect Open WebUI follow-up generator prompt."""
    t = text.lower()
    return (
        "suggest 3-5 relevant follow-up questions" in t
        and '"follow_ups"' in text
        and "json format" in t
    )


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


def is_error_response(text: str) -> bool:
    """Check if response is an error (Claude CLI puts errors in stdout with exit code 0)"""
    error_patterns = [
        "No conversation found with session ID:",
        "Error:",
        "not a valid UUID",
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
    """
    start = time.time()
    try:
        cmd = [CLAUDE_CLI_PATH]
        if session_id:
            cmd.extend(["--resume", session_id])
        # Add MCP config if available (dynamically loaded from ~/.claude.json)
        mcp_config = get_mcp_config()
        if mcp_config:
            cmd.extend(["--mcp-config", json.dumps(mcp_config)])
        cmd.extend(["-p", prompt, "--dangerously-skip-permissions"])

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
                response_text = "\n".join(f"- {f}" for f in followups)
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


async def call_model_prompt(
    model_id: str,
    prompt: str,
    session_id: Optional[str],
    history_prompt: Optional[str],
    chat_id: str,
    trace: Optional[Callable[[str, str, int], None]],
    followup_prompt: bool = False,
) -> tuple[str, Optional[str]]:
    """
    Route prompt execution to the correct model with session-not-found fallback.
    """
    if model_id == "claude-cli":
        try:
            return await run_claude_prompt(prompt, session_id, trace=trace)
        except SessionNotFoundError:
            clear_session_id(model_id, chat_id)
            fallback_prompt = history_prompt if history_prompt else prompt
            fallback_prompt = f"[SYSTEM]\n{MCP_SYSTEM_PROMPT}\n[END SYSTEM]\n\n{fallback_prompt}"
            _emit_log(trace, "claude.retry.history", "retrying without resume after session miss")
            return await run_claude_prompt(fallback_prompt, session_id=None, trace=trace)
    elif model_id == "codex-cli":
        try:
            return await run_codex_prompt(prompt, session_id, trace=trace, followup_prompt=followup_prompt)
        except CodexSessionNotFoundError:
            clear_session_id(model_id, chat_id)
            fallback_prompt = history_prompt if history_prompt else prompt
            fallback_prompt = f"[SYSTEM]\n{MCP_SYSTEM_PROMPT}\n[END SYSTEM]\n\n{fallback_prompt}"
            _emit_log(trace, "codex.retry.history", "retrying without resume after session miss")
            return await run_codex_prompt(fallback_prompt, session_id=None, trace=trace, followup_prompt=followup_prompt)
    elif model_id == "gemini-cli":
        try:
            return await run_gemini_prompt(prompt, session_id, trace=trace)
        except GeminiSessionNotFoundError:
            clear_session_id(model_id, chat_id)
            fallback_prompt = history_prompt if history_prompt else prompt
            _emit_log(trace, "gemini.retry.history", "retrying without resume after session miss")
            return await run_gemini_prompt(fallback_prompt, session_id=None, trace=trace)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_id}")


# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Claude CLI Bridge API",
        "version": app.version,
        "status": "running",
        "active_sessions": sum(len(m) for m in session_map.values())
    }


@app.get("/v1/models")
async def list_models() -> ModelList:
    """List available models (OpenAI-compatible)"""
    return ModelList(
        data=[
            Model(id=model_id, object="model", owned_by=meta["owned_by"])
            for model_id, meta in SUPPORTED_MODELS.items()
        ]
    )


@app.get("/sessions")
async def list_sessions():
    """List active sessions (debug endpoint)"""
    return {"sessions": session_map}


@app.get("/mcp")
async def get_mcp_status():
    """Show current MCP configuration (debug endpoint)"""
    mcp_config = get_mcp_config()
    if mcp_config:
        servers = list(mcp_config.get("mcpServers", {}).keys())
        return {
            "status": "configured",
            "project_path": CLAUDE_PROJECT_PATH,
            "servers": servers,
            "server_count": len(servers)
        }
    return {
        "status": "no_servers",
        "project_path": CLAUDE_PROJECT_PATH,
        "servers": [],
        "server_count": 0
    }


@app.delete("/sessions/{chat_id}")
async def delete_session(chat_id: str, model: Optional[str] = Query(None)):
    """Delete a session mapping; optionally scope to a model."""
    if model and model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    removed_models = []
    targets = [model] if model else SUPPORTED_MODELS.keys()
    for mid in targets:
        if chat_id in session_map.get(mid, {}):
            clear_session_id(mid, chat_id)
            removed_models.append(mid)

    if removed_models:
        return {"status": "deleted", "chat_id": chat_id, "models": removed_models}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Handle chat completion requests (OpenAI-compatible)"""
    trace_id, trace = _make_trace_logger()
    try:
        model_id = body.model or "claude-cli"
        if model_id not in SUPPORTED_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_id}")

        # Get chat ID and check session mapping
        chat_id = get_chat_id(request, body.messages)
        session_id = get_session_id(model_id, chat_id)
        # Detect prior sessions for other models (model switch scenario)
        other_models = [
            mid for mid in SUPPORTED_MODELS.keys()
            if mid != model_id and chat_id in session_map.get(mid, {})
        ]
        model_switch = not session_id and bool(other_models)
        trace("request.start", f"model={model_id} chat_id={chat_id} stream={body.stream} session={session_id}")
        if model_switch:
            trace("model.switch", f"chat_id={chat_id} prev_models={other_models}")

        # Build conversation history from all messages
        conversation_history = []
        current_user_message = None
        file_contents = []
        binary_paths = []

        # Find the last user message index
        last_user_idx = -1
        for i, msg in enumerate(body.messages):
            if msg.role == "user":
                last_user_idx = i

        for i, msg in enumerate(body.messages):
            text, contents, paths = process_message_content(msg.content)

            # Only collect file attachments from the LAST user message (current)
            if msg.role == "user":
                current_user_message = text
                if i == last_user_idx:
                    file_contents.extend(contents)
                    binary_paths.extend(paths)

            # Build conversation history (skip system messages)
            if msg.role in ["user", "assistant"] and text:
                role_label = "User" if msg.role == "user" else "Assistant"
                conversation_history.append(f"{role_label}: {text}")

        if not current_user_message and not file_contents and not binary_paths:
            raise HTTPException(status_code=400, detail="No user message found")

        # Build prompts - current only (optimized) and with history (fallback)
        current_only = current_user_message or ""

        history_prompt = None
        if len(conversation_history) > 1:
            history_parts = ["[Conversation history]"]
            for entry in conversation_history[:-1]:
                history_parts.append(entry)
            history_parts.append("")
            history_parts.append("[Current message]")
            history_parts.append(conversation_history[-1])
            history_prompt = "\n".join(history_parts)

        # Use current-only when session exists, history when no session.
        # If switching models, prefer history to give the new model full context.
        # For Gemini, always prefer history (no native session reuse needed to keep context).
        prompt_parts = []
        if model_id == "gemini-cli":
            prompt_parts.append(history_prompt or current_only)
        elif session_id:
            prompt_parts.append(current_only)
        elif history_prompt:
            prompt_parts.append(history_prompt)
        else:
            prompt_parts.append(current_only)

        # Add file contents
        if file_contents:
            prompt_parts.append("\n--- UPLOADED FILE CONTENT ---")
            for i, content in enumerate(file_contents, 1):
                if len(content) > 50000:
                    content = content[:50000] + "\n\n[... truncated ...]"
                prompt_parts.append(f"\n[File {i}]:\n```\n{content}\n```")
            prompt_parts.append("--- END FILE CONTENT ---")

        if binary_paths:
            prompt_parts.append("\n--- UPLOADED IMAGES/FILES ---")
            for i, path in enumerate(binary_paths, 1):
                prompt_parts.append(f"Please view the image/file at: {path}")
            prompt_parts.append("--- END UPLOADED FILES ---")

        final_message = "\n".join(prompt_parts)

        # Prepend system prompt for new sessions (resuming sessions already have context)
        if not session_id:
            final_message = f"[SYSTEM]\n{MCP_SYSTEM_PROMPT}\n[END SYSTEM]\n\n{final_message}"

        msg_preview = (current_user_message[:60] + "...") if current_user_message and len(current_user_message) > 60 else (current_user_message or "file")
        trace(
            "prompt.ready",
            f"preview={msg_preview!r} model={model_id} session={'resume' if session_id else 'new'} files_text={len(file_contents)} files_bin={len(binary_paths)} len={len(final_message)}",
        )

        # Return in OpenAI format
        if body.stream:
            return StreamingResponse(
                stream_chat_response(
                    model_id=model_id,
                    chat_id=chat_id,
                    session_id=session_id,
                    prompt=final_message,
                    history_prompt=history_prompt,
                    trace=trace,
                ),
                media_type="text/event-stream",
            )
        else:
            response_text, new_session_id = await call_model_prompt(
                model_id=model_id,
                prompt=final_message,
                session_id=session_id,
                history_prompt=history_prompt,
                chat_id=chat_id,
                trace=trace,
                followup_prompt=is_followup_prompt(final_message),
            )

            if not response_text:
                response_text = "No response generated"
            # Save session mapping if this was a new session or changed
            if new_session_id and new_session_id != session_id:
                set_session_id(model_id, chat_id, new_session_id)
                trace("session.map.updated", f"model={model_id} chat_id={chat_id} session={new_session_id}")

            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(final_message.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(final_message.split()) + len(response_text.split())
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        trace("request.error", str(e), level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_response(
    model_id: str,
    chat_id: str,
    session_id: Optional[str],
    prompt: str,
    history_prompt: Optional[str],
    trace: Optional[Callable[[str, str, int], None]],
) -> AsyncGenerator[str, None]:
    """Run the model then stream the response preserving newlines."""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    def make_chunk(content: str, finish: bool = False):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None
            }]
        }

    # Initial thinking indicator
    yield f"data: {json.dumps(make_chunk('...'))}\n\n"

    try:
        response_text, new_session_id = await call_model_prompt(
            model_id=model_id,
            prompt=prompt,
            session_id=session_id,
            history_prompt=history_prompt,
            chat_id=chat_id,
            trace=trace,
        )

        if new_session_id and new_session_id != session_id:
            set_session_id(model_id, chat_id, new_session_id)
            _emit_log(trace, "session.map.updated", f"model={model_id} chat_id={chat_id} session={new_session_id}")

        # Stream the response line by line to preserve markdown formatting
        lines = (response_text or "").split('\n')
        for i, line in enumerate(lines):
            content = line + '\n' if i < len(lines) - 1 else line
            yield f"data: {json.dumps(make_chunk(content))}\n\n"
            await asyncio.sleep(0.01)

    except Exception as e:
        error_msg = f"Error: {e}"
        _emit_log(trace, "stream.error", error_msg, level=logging.ERROR)
        yield f"data: {json.dumps(make_chunk(error_msg))}\n\n"

    yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BRIDGE_PORT, log_level="info")
