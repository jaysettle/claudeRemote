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
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union
import urllib.request
from enum import Enum
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Optional: Anthropic API for true streaming (claude-api model)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


# File upload directory (env override for dev/prod separation)
UPLOAD_DIR = Path(os.getenv("CLAUDE_UPLOAD_DIR", Path(tempfile.gettempdir()) / "claude_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Session storage directory (env override for dev/prod separation)
SESSION_DIR = Path(os.getenv("CLAUDE_SESSION_DIR", Path(tempfile.gettempdir()) / "claude_sessions"))
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SESSION_MAP_FILE = SESSION_DIR / "session_map.json"

# Configure logging
# Default to DEBUG so stream tracing is visible when tailing logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Claude CLI Bridge", version="1.12.2-dev")

# Configuration
CLAUDE_CLI_PATH = os.getenv("CLAUDE_CLI_PATH", "claude")
CLAUDE_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "600"))  # 10 minutes default
CLAUDE_PROJECT_PATH = os.getenv("CLAUDE_PROJECT_PATH", str(Path.home()))  # Project path for MCP config
CODEX_CLI_PATH = os.getenv("CODEX_CLI_PATH", str(Path.home() / ".npm-global" / "bin" / "codex"))
CODEX_TIMEOUT = int(os.getenv("CODEX_TIMEOUT", str(CLAUDE_TIMEOUT)))
GEMINI_CLI_PATH = os.getenv("GEMINI_CLI_PATH", "/usr/local/bin/gemini")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", str(CLAUDE_TIMEOUT)))
BRIDGE_PORT = int(os.getenv("CLAUDE_BRIDGE_PORT", "8000"))
CLAUDE_DISABLE_MCP = os.getenv("CLAUDE_DISABLE_MCP", "1") == "1"  # Default: disable MCP to avoid slow startup

# Interactive implementation config
def _parse_allowed_roots(raw: str) -> List[Path]:
    roots: List[Path] = []
    for part in raw.split(os.pathsep):
        if not part:
            continue
        try:
            roots.append(Path(part).expanduser().resolve())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to parse allowed root '{part}': {exc}")
    return roots

ALLOWED_WRITE_ROOTS = _parse_allowed_roots(os.getenv("IMPLEMENTATION_ALLOWED_ROOTS", str(Path.cwd())))
ALLOWED_WRITE_ROOTS.append(Path(tempfile.gettempdir()).resolve())
# Deduplicate while preserving order
_seen_roots = []
for r in ALLOWED_WRITE_ROOTS:
    if r not in _seen_roots:
        _seen_roots.append(r)
ALLOWED_WRITE_ROOTS = _seen_roots

BLOCKED_PATH_PREFIXES = [
    Path("/etc"),
    Path("/bin"),
    Path("/sbin"),
    Path("/usr"),
    Path("/lib"),
    Path("/lib64"),
    Path("/var"),
    Path("/boot"),
    Path("/opt"),
    Path("/root"),
    Path("/sys"),
    Path("/proc"),
    Path("/dev"),
]

HEALTHCHECK_CMD = os.getenv("IMPLEMENTATION_HEALTHCHECK_CMD", "")
HEALTHCHECK_URL = os.getenv("IMPLEMENTATION_HEALTHCHECK_URL", "")
HEALTHCHECK_TIMEOUT = int(os.getenv("IMPLEMENTATION_HEALTHCHECK_TIMEOUT", "15"))

SUPPORTED_MODELS = {
    "claude-cli": {"owned_by": "anthropic"},
    # "claude-api": {"owned_by": "anthropic-api"},  # DISABLED
    "codex-cli": {"owned_by": "openai-codex"},
    "gemini-cli": {"owned_by": "google-gemini"},
    "interactive-discussion": {"owned_by": "collaborative"},
}

# Anthropic API config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# ============================================================================
# CLAUDE API TOOLS - Agentic capabilities for claude-api model
# ============================================================================

import subprocess
import glob as glob_module
import shutil

# Tool definitions for Anthropic API
CLAUDE_API_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Use this to view file contents, code, configs, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed). Optional.",
                    "default": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Optional, defaults to 500.",
                    "default": 500
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "bash",
        "description": "Execute a bash command on the server. Use for running scripts, git commands, system operations, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 60.",
                    "default": 60
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern. Use to discover files in the filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern like '**/*.py' or 'src/**/*.ts'"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in. Defaults to home directory.",
                    "default": "~"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "grep",
        "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to current directory.",
                    "default": "."
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files, e.g., '*.py'",
                    "default": "*"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "list_directory",
        "description": "List files and directories in a path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to home directory.",
                    "default": "~"
                }
            },
            "required": []
        }
    }
]


def execute_tool(tool_name: str, tool_input: Dict[str, Any], trace: Optional[Callable] = None) -> str:
    """Execute a tool and return the result as a string."""
    _emit_log(trace, f"tool.{tool_name}", f"input={tool_input}")

    try:
        if tool_name == "read_file":
            return _tool_read_file(tool_input)
        elif tool_name == "write_file":
            return _tool_write_file(tool_input)
        elif tool_name == "bash":
            return _tool_bash(tool_input)
        elif tool_name == "glob":
            return _tool_glob(tool_input)
        elif tool_name == "grep":
            return _tool_grep(tool_input)
        elif tool_name == "list_directory":
            return _tool_list_directory(tool_input)
        else:
            return f"Error: Unknown tool '{tool_name}'"
    except Exception as e:
        _emit_log(trace, f"tool.{tool_name}.error", str(e), level=logging.ERROR)
        return f"Error executing {tool_name}: {str(e)}"


def _tool_read_file(input: Dict[str, Any]) -> str:
    """Read file contents."""
    path = os.path.expanduser(input.get("path", ""))
    offset = input.get("offset", 1)
    limit = input.get("limit", 500)

    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    if os.path.isdir(path):
        return f"Error: Path is a directory, not a file: {path}"

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Apply offset and limit
        start = max(0, offset - 1)
        end = start + limit
        selected_lines = lines[start:end]

        # Format with line numbers
        result_lines = []
        for i, line in enumerate(selected_lines, start=start+1):
            result_lines.append(f"{i:6d}\t{line.rstrip()}")

        result = "\n".join(result_lines)
        if len(lines) > end:
            result += f"\n... ({len(lines) - end} more lines)"

        return result
    except Exception as e:
        return f"Error reading file: {str(e)}"


def _tool_write_file(input: Dict[str, Any]) -> str:
    """Write content to file."""
    path = os.path.expanduser(input.get("path", ""))
    content = input.get("content", "")

    # Safety check - don't allow writing to certain paths
    dangerous_paths = ["/etc/", "/usr/", "/bin/", "/sbin/", "/boot/", "/root/"]
    for dp in dangerous_paths:
        if path.startswith(dp):
            return f"Error: Cannot write to protected path: {path}"

    try:
        # Create directory if needed
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def _tool_bash(input: Dict[str, Any]) -> str:
    """Execute bash command."""
    command = input.get("command", "")
    timeout = input.get("timeout", 60)

    # Safety check - block certain dangerous commands
    dangerous_patterns = ["rm -rf /", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
    for dp in dangerous_patterns:
        if dp in command:
            return f"Error: Potentially dangerous command blocked: {command}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.expanduser("~")
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n--- stderr ---\n"
            output += result.stderr

        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"

        # Truncate if too long
        if len(output) > 50000:
            output = output[:50000] + "\n... (output truncated)"

        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


def _tool_glob(input: Dict[str, Any]) -> str:
    """Find files matching glob pattern."""
    pattern = input.get("pattern", "")
    base_path = os.path.expanduser(input.get("path", "~"))

    try:
        full_pattern = os.path.join(base_path, pattern)
        matches = glob_module.glob(full_pattern, recursive=True)

        # Sort by modification time (newest first)
        matches.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        # Limit results
        if len(matches) > 100:
            matches = matches[:100]
            return "\n".join(matches) + f"\n... ({len(matches)} results shown, more available)"

        return "\n".join(matches) if matches else "No files found matching pattern"
    except Exception as e:
        return f"Error in glob: {str(e)}"


def _tool_grep(input: Dict[str, Any]) -> str:
    """Search for pattern in files."""
    pattern = input.get("pattern", "")
    path = os.path.expanduser(input.get("path", "."))
    file_pattern = input.get("file_pattern", "*")

    try:
        # Use grep command for efficiency
        cmd = f"grep -rn --include='{file_pattern}' '{pattern}' '{path}' 2>/dev/null | head -100"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern '{pattern}'"

        return output
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error in grep: {str(e)}"


def _tool_list_directory(input: Dict[str, Any]) -> str:
    """List directory contents."""
    path = os.path.expanduser(input.get("path", "~"))

    if not os.path.exists(path):
        return f"Error: Path not found: {path}"

    if not os.path.isdir(path):
        return f"Error: Not a directory: {path}"

    try:
        entries = os.listdir(path)
        result_lines = []

        for entry in sorted(entries):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                result_lines.append(f"[DIR]  {entry}/")
            else:
                try:
                    size = os.path.getsize(full_path)
                    result_lines.append(f"[FILE] {entry} ({size} bytes)")
                except:
                    result_lines.append(f"[FILE] {entry}")

        return "\n".join(result_lines) if result_lines else "(empty directory)"
    except Exception as e:
        return f"Error listing directory: {str(e)}"




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
    # Honor explicit disable flag to avoid MCP-induced latency
    if CLAUDE_DISABLE_MCP:
        return None

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


def _truncate(text: str, limit: int = 200) -> str:
    """Truncate long text for logs."""
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def _iter_text_from_content(content: Any):
    """Yield text fragments from Claude content payloads."""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                text = block.get('text', '')
                if text:
                    yield text
    elif isinstance(content, str) and content:
        yield content


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





async def stream_claude_api_with_tools(
    prompt: str,
    messages: List[Dict[str, Any]],
    trace: Optional[Callable[[str, str, int], None]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream response from Claude API with tool use capabilities.
    Implements an agentic loop that can execute tools and continue.
    """
    if not ANTHROPIC_AVAILABLE or not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API not configured. Set ANTHROPIC_API_KEY environment variable.")

    _emit_log(trace, "claude-api-tools.start", f"model={ANTHROPIC_MODEL}")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Convert messages to Anthropic format
    api_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        msg_content = msg.get("content", "")
        if role == "system":
            continue
        if role == "assistant":
            api_messages.append({"role": "assistant", "content": msg_content})
        else:
            api_messages.append({"role": "user", "content": msg_content})

    # Add current prompt
    if prompt and (not api_messages or api_messages[-1].get("content") != prompt):
        api_messages.append({"role": "user", "content": prompt})

    # Enhanced system message for tool use
    system_msg = """You are a helpful AI assistant with access to tools for interacting with the server filesystem and executing commands.

Available tools:
- read_file: Read file contents
- write_file: Write content to files
- bash: Execute shell commands
- glob: Find files by pattern
- grep: Search file contents
- list_directory: List directory contents

When you need to access files, run commands, or interact with the system, use these tools.
Always explain what you're doing and why.

""" + MCP_SYSTEM_PROMPT

    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content", "") + "\n\n" + system_msg
            break

    max_iterations = 20  # Prevent infinite loops
    iteration = 0

    try:
        while iteration < max_iterations:
            iteration += 1
            _emit_log(trace, "claude-api-tools.iteration", f"iteration={iteration}")

            # Make API call with tools
            # Retry logic for 529 overloaded errors
            max_retries = 3
            retry_delay = 2
            last_error = None

            for attempt in range(max_retries):
                try:
                    response = client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=8192,
                        system=system_msg,
                        messages=api_messages,
                        tools=CLAUDE_API_TOOLS,
                    )
                    break  # Success, exit retry loop
                except anthropic.APIStatusError as e:
                    if e.status_code == 529 and attempt < max_retries - 1:
                        _emit_log(trace, "claude-api-tools.retry", f"API overloaded, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        yield f"⏳ API busy, retrying in {retry_delay}s...\n"
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        last_error = e
                    else:
                        raise
            else:
                # All retries failed
                if last_error:
                    raise last_error

            # Process response blocks
            assistant_content = []
            has_tool_use = False

            for block in response.content:
                if block.type == "text":
                    # Stream text content
                    text = block.text
                    assistant_content.append({"type": "text", "text": text})
                    # Yield text in chunks for streaming effect
                    words = text.split(' ')
                    for i, word in enumerate(words):
                        if i > 0:
                            yield ' '
                        yield word

                elif block.type == "tool_use":
                    has_tool_use = True
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    # Show tool usage to user
                    yield f"\n\n🔧 **Using tool: {tool_name}**\n"
                    yield f"```\n{json.dumps(tool_input, indent=2)}\n```\n"

                    # Execute the tool
                    tool_result = execute_tool(tool_name, tool_input, trace)

                    # Show result preview
                    result_preview = tool_result[:500] + "..." if len(tool_result) > 500 else tool_result
                    yield f"\n📋 **Result:**\n```\n{result_preview}\n```\n\n"

                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": tool_input
                    })

                    # Add tool result to messages for next iteration
                    api_messages.append({"role": "assistant", "content": assistant_content})
                    api_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_result
                        }]
                    })
                    assistant_content = []  # Reset for next response

            # Check if we should continue or stop
            if response.stop_reason == "end_turn" or not has_tool_use:
                _emit_log(trace, "claude-api-tools.done", f"iterations={iteration}")
                break

        if iteration >= max_iterations:
            yield "\n\n⚠️ Maximum tool iterations reached."

    except Exception as e:
        _emit_log(trace, "claude-api-tools.error", str(e), level=logging.ERROR)
        raise




async def stream_claude_api(
    prompt: str,
    messages: List[Dict[str, Any]],
    trace: Optional[Callable[[str, str, int], None]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream response from Claude API with true token-by-token streaming.
    """
    if not ANTHROPIC_AVAILABLE or not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API not configured. Set ANTHROPIC_API_KEY environment variable.")

    _emit_log(trace, "claude-api.start", f"model={ANTHROPIC_MODEL}")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Convert messages to Anthropic format
    api_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        msg_content = msg.get("content", "")
        if role == "system":
            continue  # Handle system separately
        if role == "assistant":
            api_messages.append({"role": "assistant", "content": msg_content})
        else:
            api_messages.append({"role": "user", "content": msg_content})

    # Add current prompt if not already in messages
    if prompt and (not api_messages or api_messages[-1].get("content") != prompt):
        api_messages.append({"role": "user", "content": prompt})

    # Extract system message
    system_msg = MCP_SYSTEM_PROMPT
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content", "") + "\n\n" + system_msg
            break

    try:
        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=8192,
            system=system_msg,
            messages=api_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
        _emit_log(trace, "claude-api.done", "streaming complete")
    except Exception as e:
        _emit_log(trace, "claude-api.error", str(e), level=logging.ERROR)
        raise



async def run_claude_api_prompt(
    prompt: str,
    trace: Optional[Callable[[str, str, int], None]] = None,
) -> tuple[str, Optional[str]]:
    """
    Non-streaming Claude API call for tasks like follow-up generation.
    Returns (response_text, session_id) - session_id is None for API calls.
    """
    if not ANTHROPIC_AVAILABLE or not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API not configured")
    
    _emit_log(trace, "claude-api.nonstream.start", f"model={ANTHROPIC_MODEL}")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    try:
        # Retry logic for 529 errors
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=1024,  # Shorter for task responses
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except anthropic.APIStatusError as e:
                if e.status_code == 529 and attempt < max_retries - 1:
                    _emit_log(trace, "claude-api.nonstream.retry", f"API overloaded, retrying in {retry_delay}s")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
        
        # Extract text from response
        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text
        
        _emit_log(trace, "claude-api.nonstream.done", f"len={len(response_text)}")
        return response_text, None
        
    except Exception as e:
        _emit_log(trace, "claude-api.nonstream.error", str(e), level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")


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
    # elif model_id == "claude-api":
    #     # Non-streaming claude-api for tasks like follow-up generation (DISABLED)
    #     return await run_claude_api_prompt(prompt, trace=trace)
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


@app.get("/discussions")
async def list_discussions():
    """List active discussion states (debug endpoint)"""
    discussions_info = {}
    for chat_id, state in discussion_states.items():
        discussions_info[chat_id] = {
            "stage": state.stage.value,
            "mode": state.mode,
            "current_round": state.current_round,
            "max_rounds": state.max_rounds,
            "topic": state.topic[:100] + "..." if len(state.topic) > 100 else state.topic,
            "models": state.models,
            "history_count": len(state.discussion_history)
        }
    return {
        "active_discussions": discussions_info,
        "count": len(discussions_info)
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

        # Handle interactive discussion mode (SAFE: only affects interactive-discussion model)
        if model_id == "interactive-discussion":
            return await handle_interactive_discussion(request, body)

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
            # Limit history to last 10 messages to avoid "Argument list too long" errors
            # This provides enough context while staying under argument limits
            max_history_messages = 10
            recent_history = conversation_history[-max_history_messages:] if len(conversation_history) > max_history_messages else conversation_history

            history_parts = ["[Conversation history - last {} messages]".format(len(recent_history))]
            for entry in recent_history[:-1]:
                history_parts.append(entry)
            history_parts.append("")
            history_parts.append("[Current message]")
            history_parts.append(recent_history[-1])
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

        # Check if prompt is too large for command line args (>100KB to be safe)
        # This can happen with follow-up generation prompts that include full chat history
        prompt_size = len(final_message.encode('utf-8'))
        if prompt_size > 100_000:
            # Truncate to last 100KB to fit in command line arguments
            final_message_bytes = final_message.encode('utf-8')
            final_message = final_message_bytes[-100_000:].decode('utf-8', errors='ignore')
            trace(
                "prompt.truncated",
                f"Prompt too large ({prompt_size} bytes), truncated to 100KB",
                level=logging.WARNING,
            )

        msg_preview = (current_user_message[:60] + "...") if current_user_message and len(current_user_message) > 60 else (current_user_message or "file")
        trace(
            "prompt.ready",
            f"preview={msg_preview!r} model={model_id} session={'resume' if session_id else 'new'} files_text={len(file_contents)} files_bin={len(binary_paths)} len={len(final_message)}",
        )

        # Return in OpenAI format
        if body.stream:
            # Use incremental streaming for claude-cli and codex-cli
            if model_id == "claude-cli":
                return StreamingResponse(
                    stream_claude_incremental(
                        chat_id=chat_id,
                        session_id=session_id,
                        prompt=final_message,
                        history_prompt=history_prompt,
                        trace=trace,
                    ),
                    media_type="text/event-stream",
                )
            elif model_id == "codex-cli":
                return StreamingResponse(
                    stream_codex_incremental(
                        chat_id=chat_id,
                        session_id=session_id,
                        prompt=final_message,
                        history_prompt=history_prompt,
                        trace=trace,
                    ),
                    media_type="text/event-stream",
                )
            else:
                # Gemini and other models use generic streaming
                return StreamingResponse(
                    stream_chat_response(
                        model_id=model_id,
                        chat_id=chat_id,
                        session_id=session_id,
                        prompt=final_message,
                        history_prompt=history_prompt,
                        trace=trace,
                        messages=[m.dict() if hasattr(m, "dict") else m for m in body.messages] if body.messages else None,
                    ),
                    media_type="text/event-stream",
                )
        else:
            # Use codex-cli for follow-up generation (faster, uses OpenAI)
            if is_followup_prompt(final_message):
                trace("followup.codex", "routing follow-up to codex-cli")
                try:
                    response_text, _ = await run_codex_prompt(
                        prompt=final_message,
                        session_id=None,  # No session needed for follow-ups
                        timeout=30,  # Short timeout for follow-ups
                        trace=trace,
                        followup_prompt=True,
                    )
                    new_session_id = session_id
                except Exception as e:
                    trace("followup.error", f"codex failed: {e}, returning empty")
                    response_text = '{"follow_ups": []}'
                    new_session_id = session_id
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


async def stream_chat_response(
    model_id: str,
    chat_id: str,
    session_id: Optional[str],
    prompt: str,
    history_prompt: Optional[str],
    trace: Optional[Callable[[str, str, int], None]],
    messages: Optional[List[Dict[str, Any]]] = None,
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

    # For claude-api, use true streaming with tools (DISABLED)
    # if model_id == "claude-api":
    #     try:
    #         async for token in stream_claude_api_with_tools(prompt, messages or [], trace):
    #             yield f"data: {json.dumps(make_chunk(token))}\n\n"
    #         yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
    #         yield "data: [DONE]\n\n"
    #         return
    #     except Exception as e:
    #         error_msg = f"Error: {e}"
    #         _emit_log(trace, "stream.error", error_msg, level=logging.ERROR)
    #         yield f"data: {json.dumps(make_chunk(error_msg))}\n\n"
    #         yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
    #         yield "data: [DONE]\n\n"
    #         return

    # Initial thinking indicator for CLI models
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


# ============================================================================
# INTERACTIVE DISCUSSION MODULE - Safe addition, doesn't affect existing models
# ============================================================================

class DiscussionStage(Enum):
    SETUP = "setup"
    INITIAL = "initial"
    DISCUSSION = "discussion"  # Multi-round interactive stage
    ROUND1 = "round1"
    SYNTHESIS = "synthesis"
    IMPLEMENTATION = "implementation"  # Code implementation mode
    COMPLETE = "complete"

@dataclass
class DiscussionState:
    chat_id: str
    stage: DiscussionStage = DiscussionStage.SETUP
    topic: str = ""
    models: List[str] = None
    mode: str = "collaborate"  # collaborate or debate
    max_rounds: int = 999  # Effectively unlimited - user decides when to stop
    current_round: int = 0
    model_sessions: Dict[str, Optional[str]] = None
    discussion_history: List[Dict[str, str]] = None
    waiting_for_user: bool = False
    # Implementation mode fields
    implementation_plan: str = ""
    implementation_code: Dict[str, str] = None  # filename -> code content
    implementation_approved: bool = False
    rollback_commit: str = ""  # Git commit hash for rollback

    def __post_init__(self):
        if self.models is None:
            self.models = ["claude-cli", "codex-cli"]
        if self.model_sessions is None:
            self.model_sessions = {}
        if self.discussion_history is None:
            self.discussion_history = []
        if self.implementation_code is None:
            self.implementation_code = {}

    def get_model_name(self, model_id: str) -> str:
        """Get friendly model name"""
        return model_id.replace("-cli", "").title()

    def add_response(self, model: str, response: str, stage: str):
        """Add a model response to the discussion history"""
        self.discussion_history.append({
            "stage": stage,
            "model": model,
            "response": response,
            "timestamp": time.time()
        })

# Global discussion states (simple in-memory storage for dev)
discussion_states: Dict[str, DiscussionState] = {}

def get_discussion_state(chat_id: str) -> Optional[DiscussionState]:
    """Get discussion state for a chat"""
    return discussion_states.get(chat_id)

def set_discussion_state(state: DiscussionState):
    """Save discussion state"""
    discussion_states[state.chat_id] = state

def clear_discussion_state(chat_id: str):
    """Clear discussion state"""
    if chat_id in discussion_states:
        del discussion_states[chat_id]

def parse_discussion_intent(user_input: str) -> Dict[str, Any]:
    """Parse user intent for discussion configuration"""
    config = {
        "topic": user_input,
        "models": ["claude-cli", "codex-cli"],
        "rounds": 2,
        "mode": "collaborate"
    }

    input_lower = user_input.lower()

    # Detect mode
    if any(word in input_lower for word in ["debate", "argue", "disagree", "vs", "versus"]):
        config["mode"] = "debate"

    # Detect model preferences
    if "gemini" in input_lower:
        if "codex" in input_lower:
            config["models"] = ["gemini-cli", "codex-cli"]
        else:
            config["models"] = ["claude-cli", "gemini-cli"]

    # Extract core topic - only strip known command words if they appear as the FIRST word
    topic = user_input
    first_word_raw = input_lower.strip().split()[0] if input_lower.strip() else ""
    # Strip common punctuation from first word for matching (debate: → debate)
    first_word = first_word_raw.rstrip(':,;.!?')

    # Known command words that should be stripped from the beginning
    command_words = ["discuss", "debate", "analyze", "compare"]

    if first_word in command_words:
        # Remove the first word and use the rest as topic
        words = user_input.strip().split(None, 1)
        topic = words[1] if len(words) > 1 else user_input
    else:
        # Use entire input as topic
        topic = user_input

    logger.info(f"parse_discussion_intent | first_word='{first_word}' | topic='{topic[:100]}...'")

    config["topic"] = topic if topic else user_input
    return config


# ==========================================================================
# Implementation safety + rollback helpers
# ==========================================================================

def is_safe_path(path: Path) -> bool:
    """Validate target path against whitelist/blacklist rules."""
    try:
        resolved = path.expanduser().resolve()
    except Exception as exc:
        logger.warning(f"is_safe_path: failed to resolve {path}: {exc}")
        return False

    for blocked in BLOCKED_PATH_PREFIXES:
        if resolved == blocked or blocked in resolved.parents:
            return False

    for root in ALLOWED_WRITE_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue

    return False


def find_git_root(start: Path) -> Optional[Path]:
    """Find nearest git root for a path."""
    try:
        candidate = start.expanduser().resolve()
    except Exception:
        return None

    for parent in [candidate] + list(candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None


def create_rollback_commit(repo_root: Path) -> Optional[str]:
    """Create an empty commit as rollback checkpoint if repo is clean."""
    if not repo_root or not shutil.which("git"):
        return None

    def _run_git(args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
        )

    inside = _run_git(["rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0:
        logger.info(f"Rollback commit skipped: not a git repo at {repo_root}")
        return None

    status = _run_git(["status", "--porcelain"])
    if status.returncode != 0:
        logger.warning(f"Rollback commit status failed: {status.stderr.strip()}")
        return None

    if status.stdout.strip():
        logger.info("Rollback commit skipped: working tree dirty")
        return None

    commit_message = f"Implementation rollback checkpoint {int(time.time())}"
    commit = _run_git(["commit", "--allow-empty", "-m", commit_message])
    if commit.returncode != 0:
        logger.warning(f"Rollback commit failed: {commit.stderr.strip()}")
        return None

    head = _run_git(["rev-parse", "HEAD"])
    if head.returncode == 0:
        return head.stdout.strip()

    logger.warning("Rollback commit created but unable to read HEAD")
    return None


def reset_to_commit(repo_root: Path, commit_hash: str) -> bool:
    if not repo_root or not commit_hash or not shutil.which("git"):
        return False

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "reset", "--hard", commit_hash],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"Git rollback failed: {result.stderr.strip()}")
            return False
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(f"Git rollback exception: {exc}")
        return False


def backup_file(src: Path, backup_dir: Path) -> Optional[Path]:
    """Copy src into backup_dir preserving absolute structure."""
    try:
        resolved = src.expanduser().resolve()
        rel = resolved.relative_to(resolved.anchor)
        dest = backup_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if resolved.exists():
            shutil.copy2(resolved, dest)
        return dest
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to back up {src}: {exc}")
        return None


def write_file_with_backup(target: Path, content: str, backup_dir: Path) -> Tuple[Path, bool]:
    """Write file atomically with backup; returns (path, created)."""
    resolved = target.expanduser().resolve()
    created = not resolved.exists()
    if not created:
        backup_file(resolved, backup_dir)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="impl_", dir=str(resolved.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())

    os.replace(tmp_path, resolved)
    return resolved, created


def restore_backups(backup_dir: Path) -> List[Path]:
    restored: List[Path] = []
    if not backup_dir.exists():
        return restored

    for file in backup_dir.rglob("*"):
        if file.is_file():
            rel = file.relative_to(backup_dir)
            target = Path(os.sep) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            restored.append(target)
    return restored


def delete_created_files(paths: List[Path]) -> List[Path]:
    removed: List[Path] = []
    for p in paths:
        try:
            if p.exists():
                p.unlink()
                removed.append(p)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to delete {p}: {exc}")
    return removed


def check_service_health() -> Tuple[bool, str]:
    """Run configured health check command or HTTP probe."""
    if HEALTHCHECK_CMD:
        try:
            result = subprocess.run(
                HEALTHCHECK_CMD,
                shell=True,
                capture_output=True,
                text=True,
                timeout=HEALTHCHECK_TIMEOUT,
            )
            output = (result.stdout or "").strip() or (result.stderr or "").strip()
            return result.returncode == 0, output or f"Exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return False, "Health check command timed out"
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"Health check command failed: {exc}"

    if HEALTHCHECK_URL:
        try:
            with urllib.request.urlopen(HEALTHCHECK_URL, timeout=HEALTHCHECK_TIMEOUT) as resp:
                return 200 <= resp.status < 300, f"HTTP {resp.status}"
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"Health check HTTP failed: {exc}"

    return True, "No health check configured"

def build_discussion_history_prompt(state: DiscussionState) -> str:
    """Build a history prompt with full discussion context for session fallback"""
    history_parts = []
    history_parts.append(f"[Original Topic]\n{state.topic}\n")
    history_parts.append(f"\n[Discussion History - {state.current_round} rounds]\n")

    for entry in state.discussion_history[-10:]:  # Last 10 entries
        model_name = state.get_model_name(entry["model"])
        response = entry["response"]
        if len(response) > 800:
            response = response[:800] + "... [truncated]"
        history_parts.append(f"\n{model_name}: {response}\n")

    return "\n".join(history_parts)

def format_discussion_prompt(topic: str, mode: str, is_initial: bool = True, other_response: str = "", user_guidance: str = "", debate_position: str = "") -> str:
    """Format prompts for discussion"""
    if is_initial:
        if mode == "debate":
            position_instruction = f"\n\nYou must argue FOR: {debate_position}" if debate_position else "\n\nTake a clear position and provide your strongest arguments."
            return f"""You are participating in a debate about: "{topic}"{position_instruction} Be intellectually rigorous but respectful.

Your analysis:"""
        else:
            return f"""You are collaborating on analyzing: "{topic}"

Provide your expertise and perspective. Focus on constructive analysis and suggestions.

Your analysis:"""
    else:
        # Response to other model - ENHANCED for deeper engagement
        # If user provided guidance, incorporate it into the prompt
        if user_guidance:
            if mode == "debate":
                return f"""The user has provided this guidance: "{user_guidance}"

Your colleague's previous response:
"{other_response}"

Now respond to BOTH the user's guidance AND your colleague's points:
- Address the user's question or direction
- Challenge or support your colleague's relevant points
- Present your perspective on the user's guidance

Your response:"""
            else:
                return f"""The user has provided this guidance: "{user_guidance}"

Your colleague's previous response:
"{other_response}"

Please respond to BOTH the user's guidance AND build on your colleague's analysis:
- Address the user's question or direction
- Integrate your colleague's relevant points
- Add your own perspective and expertise

Your response:"""
        else:
            # No user guidance - models respond to each other
            if mode == "debate":
                return f"""Your colleague just argued:

"{other_response}"

Now respond to their points:
- Challenge any claims you disagree with (provide reasoning)
- Acknowledge points where they're correct
- Present counter-arguments or alternative perspectives
- Ask clarifying questions if needed

Your rebuttal:"""
            else:
                return f"""Your colleague shared this analysis:

"{other_response}"

Build on their contribution:
- Identify points you agree with and why
- Add complementary perspectives they may have missed
- Address any gaps or concerns you see
- Suggest how to combine both viewpoints

Your response:"""

async def handle_interactive_discussion(request: Request, body: ChatCompletionRequest):
    """Handle interactive discussion mode - SAFE: only called for interactive-discussion model"""
    trace_id, trace = _make_trace_logger()
    chat_id = get_chat_id(request, body.messages)

    # Get or create discussion state
    state = get_discussion_state(chat_id)
    user_input = body.messages[-1].content if body.messages else ""

    logger.info(f"[{trace}] interactive_discussion.request | chat_id={chat_id} input='{user_input[:50]}...' has_state={state is not None} stage={state.stage.value if state else 'none'}")

    if body.stream:
        return StreamingResponse(
            stream_interactive_discussion(
                chat_id=chat_id,
                user_input=user_input,
                state=state,
                trace=trace,
            ),
            media_type="text/event-stream",
        )
    else:
        # Simple non-streaming response for testing
        result = "Interactive discussion mode active. Use streaming for full experience."
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "interactive-discussion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop"
            }]
        }

async def stream_interactive_discussion(
    chat_id: str,
    user_input: str,
    state: Optional[DiscussionState],
    trace: Optional[Callable] = None,
) -> AsyncGenerator[str, None]:
    """Stream interactive discussion - MINIMAL VERSION for testing"""

    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    def make_chunk(content: str, finish: bool = False):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "interactive-discussion",
            "choices": [{
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None
            }]
        }

    try:
        # Handle setup stage
        if not state or state.stage == DiscussionStage.SETUP:
            # Parse discussion request
            config = parse_discussion_intent(user_input)

            # Create new state
            state = DiscussionState(
                chat_id=chat_id,
                topic=config["topic"],
                models=config["models"],
                mode=config["mode"]
            )
            set_discussion_state(state)

            # Show setup
            model_names = [state.get_model_name(m) for m in state.models]
            yield f"data: {json.dumps(make_chunk(f'🎭 **Interactive Discussion Setup**\n\n'))}\n\n"
            # Truncate topic for display (first 200 chars or first line)
            topic_display = state.topic.split('\n')[0][:200]
            if len(state.topic) > 200 or '\n' in state.topic:
                topic_display += "..."
            yield f"data: {json.dumps(make_chunk(f'**Topic:** {topic_display}\n'))}\n\n"
            participants_text = ' vs '.join(model_names)
            yield f"data: {json.dumps(make_chunk(f'**Participants:** {participants_text}\n'))}\n\n"
            yield f"data: {json.dumps(make_chunk(f'**Mode:** {state.mode.title()}\n\n'))}\n\n"

            # Move to initial stage
            state.stage = DiscussionStage.INITIAL
            set_discussion_state(state)

            yield f"data: {json.dumps(make_chunk('Type **\"start\"** to begin, or **\"cancel\"** to end.\n\n'))}\n\n"

        elif state.stage == DiscussionStage.INITIAL:
            user_command = user_input.lower().strip()

            if user_command == "cancel":
                clear_discussion_state(chat_id)
                yield f"data: {json.dumps(make_chunk('Discussion cancelled. 👋'))}\n\n"
            elif user_command == "start":
                # Run initial analysis
                yield f"data: {json.dumps(make_chunk(f'🎬 **Starting Discussion: {state.topic}**\n\n'))}\n\n"

                # Get responses from both models
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    icon = "🔵" if i == 0 else "🟡"

                    # For debate mode, try to assign opposing positions from topic
                    debate_position = ""
                    if state.mode == "debate" and " or " in state.topic.lower():
                        # Extract "X or Y" pattern and assign positions
                        parts = state.topic.lower().split(" or ", 1)
                        if len(parts) == 2:
                            # First model gets first option, second model gets second option
                            option1 = parts[0].split()[-1] if parts[0].split() else parts[0]  # Last word before "or"
                            option2 = parts[1].split()[0] if parts[1].split() else parts[1]   # First word after "or"
                            debate_position = option1 if i == 0 else option2

                    # Show which position this model will argue for
                    if debate_position:
                        yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} analyzing... (arguing for: {debate_position})**\n'))}\n\n"
                    else:
                        yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} analyzing...**\n'))}\n\n"

                    prompt = format_discussion_prompt(state.topic, state.mode, is_initial=True, debate_position=debate_position)
                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    state.add_response(model, response, "initial")
                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                # Move to discussion rounds for interactive exchange
                state.current_round = 1
                state.stage = DiscussionStage.DISCUSSION
                set_discussion_state(state)

                if state.max_rounds > 1:
                    yield f"data: {json.dumps(make_chunk(f'\n**Round {state.current_round} complete.** Type **\"continue\"** for round {state.current_round + 1}, provide your own guidance/question, **\"implement\"** to build an idea, **\"export\"** for summary, or **\"stop\"** to end.\n\n'))}\n\n"
                else:
                    state.stage = DiscussionStage.SYNTHESIS
                    set_discussion_state(state)
                    yield f"data: {json.dumps(make_chunk('\n**Discussion complete!** Type **\"export\"** for summary.\n\n'))}\n\n"
            else:
                yield f"data: {json.dumps(make_chunk('Please type **\"start\"** or **\"cancel\"**.'))}\n\n"

        elif state.stage == DiscussionStage.DISCUSSION:
            user_command = user_input.lower().strip()
            logger.info(f"[{trace}] discussion.command | chat_id={chat_id} cmd={user_command} stage={state.stage.value} round={state.current_round}")

            if user_command == "stop":
                state.stage = DiscussionStage.SYNTHESIS
                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk('**Discussion ended.** Type **\"export\"** for summary.\n\n'))}\n\n"

            elif user_command == "export":
                # User wants to export immediately from DISCUSSION stage
                logger.info(f"[{trace}] discussion.export | chat_id={chat_id} round={state.current_round} entries={len(state.discussion_history)}")
                # Generate the export with summary directly
                yield f"data: {json.dumps(make_chunk('📋 **Discussion Export**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Topic:** {state.topic}\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Mode:** {state.mode.title()}\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Rounds:** {state.current_round}\n\n'))}\n\n"

                yield f"data: {json.dumps(make_chunk('---\n\n## Round-by-Round Discussion\n\n'))}\n\n"

                for entry in state.discussion_history:
                    model_name = state.get_model_name(entry["model"])
                    response_text = entry["response"]
                    round_info = entry.get("stage", "")
                    yield f"data: {json.dumps(make_chunk(f'**{model_name}** ({round_info}):\n{response_text}\n\n'))}\n\n"

                # Generate AI summary using Claude
                yield f"data: {json.dumps(make_chunk('---\n\n## 🤖 AI-Generated Summary\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('*Analyzing discussion...*\n\n'))}\n\n"

                # Build summary prompt with full discussion history
                summary_parts = [f"Summarize this discussion about: {state.topic}\n\n"]
                summary_parts.append(f"Mode: {state.mode}\n")
                summary_parts.append(f"Total rounds: {state.current_round}\n\n")
                summary_parts.append("Discussion transcript:\n")

                for entry in state.discussion_history:
                    model_name = state.get_model_name(entry["model"])
                    response_text = entry["response"]
                    # Truncate very long responses for summary prompt
                    if len(response_text) > 1000:
                        response_text = response_text[:1000] + "... [truncated]"
                    summary_parts.append(f"{model_name}: {response_text}\n\n")

                summary_parts.append("""
Please provide:
1. **Key Points of Agreement**: What did both models agree on?
2. **Key Points of Disagreement**: Where did they differ?
3. **Evolution**: How did positions change across rounds?
4. **Conclusion**: What's the synthesized recommendation or outcome?

Keep it concise (3-5 paragraphs).""")

                summary_prompt = "".join(summary_parts)

                # Call Claude to generate summary
                try:
                    summary_response, _ = await call_model_prompt(
                        model_id="claude-cli",
                        prompt=summary_prompt,
                        session_id=None,  # Don't use discussion session
                        history_prompt=None,
                        chat_id=chat_id + "_summary",  # Different chat for summary
                        trace=trace,
                    )

                    yield f"data: {json.dumps(make_chunk(summary_response))}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps(make_chunk(f'*Summary generation failed: {str(e)}*'))}\n\n"

                yield f"data: {json.dumps(make_chunk('\n---\n\n*Export complete. Discussion ended.*\n\n'))}\n\n"
                clear_discussion_state(chat_id)

            elif user_command == "implement" or user_input.lower().startswith("implement:"):
                # User wants to implement a feature
                if user_command == "implement":
                    # User just typed "implement" - use the last follow-up or discussion topic as context
                    feature_description = "the idea discussed above"
                    logger.info(f"[{trace}] implementation.start | chat_id={chat_id} feature='from_context'")
                else:
                    feature_description = user_input[10:].strip()  # Remove "implement:" prefix
                    logger.info(f"[{trace}] implementation.start | chat_id={chat_id} feature='{feature_description}'")

                state.stage = DiscussionStage.IMPLEMENTATION
                state.implementation_plan = ""
                state.implementation_code = {}
                state.implementation_approved = False
                set_discussion_state(state)

                yield f"data: {json.dumps(make_chunk(f'🔧 **Implementation Mode**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Feature:** {feature_description}\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('**Stage:** Planning\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('Models are discussing how to implement this feature...\n\n'))}\n\n"

                # Both models collaborate on implementation plan
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    icon = "🔵" if i == 0 else "🟡"

                    yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} planning...**\n\n'))}\n\n"

                    # Create planning prompt with context from discussion if available
                    context_info = ""
                    if feature_description == "the idea discussed above" and state.discussion_history:
                        # Include recent discussion context
                        recent_entries = state.discussion_history[-4:]  # Last 4 entries
                        context_parts = ["Recent discussion context:\n"]
                        for entry in recent_entries:
                            model_name = state.get_model_name(entry["model"])
                            response = entry["response"][:300]  # First 300 chars
                            context_parts.append(f"{model_name}: {response}...\n")
                        context_info = "\n".join(context_parts) + "\n"

                    # Detect project path from discussion context
                    project_path_hint = ""
                    if state.discussion_history:
                        for entry in state.discussion_history:
                            response = entry["response"]
                            # Look for paths mentioned in discussion
                            if "/home/jay/projects/" in response:
                                import re
                                paths = re.findall(r'/home/jay/projects/[^\s]+', response)
                                if paths:
                                    project_path_hint = f"\nIMPORTANT: The project is located in: {paths[0]}\nAll file paths should be relative to this project directory.\n"
                                    break

                    planning_prompt = f"""{context_info}You are helping implement this feature: "{feature_description}"{project_path_hint}

Your task is to create an implementation plan. Analyze:
1. What files need to be modified or created (use paths relative to the project directory if mentioned)
2. What changes are needed in each file
3. Dependencies or prerequisites
4. Potential risks or challenges
5. How to test the implementation

Be specific about file paths and code changes. Keep the plan practical and implementable.

Your implementation plan:"""

                    # Get the other model's plan if available
                    history_prompt = ""
                    if i == 1 and state.implementation_plan:
                        history_prompt = f"\nYour colleague proposed:\n{state.implementation_plan}\n\nNow provide your analysis and suggestions:\n"
                        planning_prompt = history_prompt + planning_prompt

                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=planning_prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    # Store the plan
                    if not state.implementation_plan:
                        state.implementation_plan = f"{model_name}:\n{response}\n\n"
                    else:
                        state.implementation_plan += f"{model_name}:\n{response}\n\n"

                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk('\n**Plan complete!** Review the plan above.\n\nType **\"approve\"** to proceed with implementation, **\"revise: <feedback>\"** to modify the plan, or **\"cancel\"** to abort.\n\n'))}\n\n"

            elif user_command == "continue":
                # Check if we've hit max rounds
                if state.current_round >= state.max_rounds:
                    state.stage = DiscussionStage.SYNTHESIS
                    set_discussion_state(state)
                    yield f"data: {json.dumps(make_chunk(f'**Maximum rounds ({state.max_rounds}) reached.** Type **\"export\"** for summary.\n\n'))}\n\n"
                else:
                    # Run next round - models respond to each other (no user guidance)
                    state.current_round += 1
                    yield f"data: {json.dumps(make_chunk(f'\n🔄 **Round {state.current_round}** - Models responding to each other...\n\n'))}\n\n"

                    # Each model responds to the OTHER model's last response
                    for i, model in enumerate(state.models):
                        model_name = state.get_model_name(model)
                        other_model = state.models[1 - i]  # Get the other model
                        icon = "🔵" if i == 0 else "🟡"

                        # Find the other model's most recent response
                        other_response = None
                        for entry in reversed(state.discussion_history):
                            if entry["model"] == other_model:
                                other_response = entry["response"]
                                break

                        if not other_response:
                            continue

                        # Truncate other_response if too long (keep it under 2000 chars for context)
                        if len(other_response) > 2000:
                            other_response = other_response[:2000] + "\n\n[...response truncated for brevity...]"

                        yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} responding to {state.get_model_name(other_model)}...**\n'))}\n\n"

                        # Format prompt with the other model's response (no user guidance)
                        prompt = format_discussion_prompt(
                            state.topic,
                            state.mode,
                            is_initial=False,  # This is a follow-up
                            other_response=other_response
                        )

                        # Build history prompt for fallback if session is lost
                        history_prompt = build_discussion_history_prompt(state) + "\n\n" + prompt

                        response, session = await call_model_prompt(
                            model_id=model,
                            prompt=prompt,
                            session_id=get_session_id(model, chat_id),
                            history_prompt=history_prompt,
                            chat_id=chat_id,
                            trace=trace,
                        )

                        if session:
                            set_session_id(model, chat_id, session)

                        state.add_response(model, response, f"round_{state.current_round}")
                        yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                    # Offer to continue or stop
                    set_discussion_state(state)
                    if state.current_round >= state.max_rounds:
                        state.stage = DiscussionStage.SYNTHESIS
                        yield f"data: {json.dumps(make_chunk(f'\n**Discussion complete ({state.max_rounds} rounds).** Type **\"export\"** for summary.\n\n'))}\n\n"
                    else:
                        yield f"data: {json.dumps(make_chunk(f'\n**Round {state.current_round} complete.** Type **\"continue\"** for round {state.current_round + 1}, **\"export\"** for summary, or **\"stop\"** to end.\n\n'))}\n\n"
            else:
                # User provided free-form guidance instead of a command
                user_guidance = user_input  # Use the original input, not lowercased
                state.current_round += 1
                yield f"data: {json.dumps(make_chunk(f'\n💬 **Round {state.current_round}** - User guidance: "{user_guidance[:100]}{"..." if len(user_guidance) > 100 else ""}"\n\n'))}\n\n"

                # Both models respond to user's guidance
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    other_model = state.models[1 - i]
                    icon = "🔵" if i == 0 else "🟡"

                    # Find the other model's most recent response for context
                    other_response = None
                    for entry in reversed(state.discussion_history):
                        if entry["model"] == other_model:
                            other_response = entry["response"]
                            break

                    if not other_response:
                        other_response = "(No previous response from colleague)"

                    # Truncate other_response if too long
                    if len(other_response) > 2000:
                        other_response = other_response[:2000] + "\n\n[...response truncated for brevity...]"

                    yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} responding to user guidance...**\n'))}\n\n"

                    # Format prompt with user guidance AND other model's response
                    prompt = format_discussion_prompt(
                        state.topic,
                        state.mode,
                        is_initial=False,
                        other_response=other_response,
                        user_guidance=user_guidance  # Pass user's guidance
                    )

                    # Build history prompt for fallback if session is lost
                    history_prompt = build_discussion_history_prompt(state) + "\n\n" + prompt

                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=history_prompt,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    state.add_response(model, response, f"round_{state.current_round}_user_guided")
                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                # Offer to continue or stop
                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk(f'\n**Round {state.current_round} complete.** Type **\"continue\"** for round {state.current_round + 1}, provide guidance, **\"implement\"** to build an idea, **\"export\"** for summary, or **\"stop\"** to end.\n\n'))}\n\n"

        elif state.stage == DiscussionStage.IMPLEMENTATION:
            user_command = user_input.lower().strip()
            logger.info(f"[{trace}] implementation.command | chat_id={chat_id} cmd={user_command}")

            if user_command == "approve":
                # User approved the plan, generate code
                logger.info(f"[{trace}] implementation.approved | chat_id={chat_id}")
                state.implementation_approved = True
                set_discussion_state(state)

                yield f"data: {json.dumps(make_chunk('✅ **Plan approved!**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('**Stage:** Code Generation\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('Models are generating code...\n\n'))}\n\n"

                # Both models generate code
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    icon = "🔵" if i == 0 else "🟡"

                    yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} coding...**\n\n'))}\n\n"

                    # Create code generation prompt
                    code_prompt = f"""Based on this implementation plan:

{state.implementation_plan}

Generate the actual code changes. For each file:
1. Specify the full file path
2. Show the complete code or the specific changes needed

Format your response as:
```
FILE: /path/to/file.py
[complete file content or clear instructions for changes]
```

Focus on ONE key file that needs modification. Be precise and complete.

Your code:"""

                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=code_prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                    # Try to extract file path and code from response
                    # Simple pattern matching for FILE: declarations
                    import re
                    file_matches = re.findall(r'FILE:\s*(.+?)```(.+?)```', response, re.DOTALL)
                    for file_path, code_content in file_matches:
                        file_path = file_path.strip()
                        code_content = code_content.strip()
                        state.implementation_code[file_path] = code_content
                        logger.info(f"[{trace}] implementation.code_extracted | file={file_path} chars={len(code_content)}")

                set_discussion_state(state)

                if state.implementation_code:
                    yield f"data: {json.dumps(make_chunk(f'\n**Code generation complete!**\n\n'))}\n\n"
                    yield f"data: {json.dumps(make_chunk(f'Files to modify: {len(state.implementation_code)}\n\n'))}\n\n"
                    for filepath in state.implementation_code.keys():
                        yield f"data: {json.dumps(make_chunk(f'- {filepath}\n'))}\n\n"
                    yield f"data: {json.dumps(make_chunk('\n\nType **\"deploy\"** to apply changes, or **\"cancel\"** to abort.\n\n'))}\n\n"
                else:
                    yield f"data: {json.dumps(make_chunk('⚠️ **No code files were extracted.** The models may need clearer instructions.\n\nType **\"revise: provide code in FILE: format\"** to try again.\n\n'))}\n\n"

            elif user_command == "deploy":
                # Deploy the code changes
                logger.info(f"[{trace}] implementation.deploy | chat_id={chat_id} files={len(state.implementation_code)}")

                if not state.implementation_code:
                    yield f"data: {json.dumps(make_chunk('⚠️ No code files captured to deploy. Try **"approve"** again.\n\n'))}\n\n"
                else:
                    yield f"data: {json.dumps(make_chunk('🚀 **Deploying changes...**\n\n'))}\n\n"

                    # Pre-flight safety check
                    unsafe_files = []
                    resolved_paths: Dict[str, Path] = {}
                    for filepath in state.implementation_code.keys():
                        resolved = Path(filepath).expanduser()
                        resolved_paths[filepath] = resolved
                        if not is_safe_path(resolved):
                            unsafe_files.append(str(resolved))

                    if unsafe_files:
                        warning = "\n".join([f"- {p}" for p in unsafe_files])
                        yield f"data: {json.dumps(make_chunk(f'⛔ Blocked unsafe paths. Adjust the target files and retry:\n{warning}\n\n'))}\n\n"
                    else:
                        git_root = None
                        for resolved in resolved_paths.values():
                            git_root = find_git_root(resolved)
                            if git_root:
                                break

                        rollback_commit = create_rollback_commit(git_root) if git_root else None
                        state.rollback_commit = rollback_commit or ""
                        set_discussion_state(state)

                        if rollback_commit:
                            yield f"data: {json.dumps(make_chunk(f'💾 Rollback checkpoint created at {rollback_commit}\n\n'))}\n\n"
                        elif git_root:
                            yield f"data: {json.dumps(make_chunk('⚠️ Git rollback skipped (dirty tree or commit failed). Using file backups only.\n\n'))}\n\n"
                        else:
                            yield f"data: {json.dumps(make_chunk('ℹ️ No git repository detected. Using file backups for rollback.\n\n'))}\n\n"

                        backup_dir = Path(tempfile.mkdtemp(prefix="implementation_backup_"))
                        created_files: List[Path] = []
                        write_error: Optional[str] = None

                        for filepath, code in state.implementation_code.items():
                            target = resolved_paths[filepath]
                            yield f"data: {json.dumps(make_chunk(f'✏️ Writing {target}\n'))}\n\n"
                            try:
                                written, created = write_file_with_backup(target, code, backup_dir)
                                if created:
                                    created_files.append(written)
                                yield f"data: {json.dumps(make_chunk(f'✅ Wrote {written}\n'))}\n\n"
                            except Exception as exc:
                                write_error = f"Failed to write {target}: {exc}"
                                logger.exception(write_error)
                                break

                        if write_error:
                            yield f"data: {json.dumps(make_chunk('⚠️ Error during write. Rolling back...\n\n'))}\n\n"
                            restored = restore_backups(backup_dir)
                            removed = delete_created_files(created_files)
                            rollback_ok = reset_to_commit(git_root, state.rollback_commit) if state.rollback_commit else False
                            yield f"data: {json.dumps(make_chunk(f'Rolled back {len(restored)} files; removed {len(removed)} new files. Git rollback: {"ok" if rollback_ok else "skipped"}.\n\n'))}\n\n"
                            yield f"data: {json.dumps(make_chunk(write_error + '\n\n'))}\n\n"
                            clear_discussion_state(chat_id)
                        else:
                            # Health check after writes
                            health_ok, health_detail = check_service_health()
                            if not health_ok:
                                yield f"data: {json.dumps(make_chunk(f'❌ Health check failed: {health_detail}\nRolling back...\n\n'))}\n\n"
                                restored = restore_backups(backup_dir)
                                removed = delete_created_files(created_files)
                                rollback_ok = reset_to_commit(git_root, state.rollback_commit) if state.rollback_commit else False
                                yield f"data: {json.dumps(make_chunk(f'Rolled back {len(restored)} files; removed {len(removed)} new files. Git rollback: {"ok" if rollback_ok else "skipped"}.\n\n'))}\n\n"
                                clear_discussion_state(chat_id)
                            else:
                                yield f"data: {json.dumps(make_chunk(f'🩺 Health check passed: {health_detail}\n\n'))}\n\n"
                                yield f"data: {json.dumps(make_chunk(f'✅ **Implementation complete.** Backups stored at {backup_dir}.\n\n'))}\n\n"
                                clear_discussion_state(chat_id)

            elif user_input.lower().startswith("revise:"):
                # User wants to revise the plan
                feedback = user_input[7:].strip()
                logger.info(f"[{trace}] implementation.revise | chat_id={chat_id} feedback='{feedback[:50]}'")

                yield f"data: {json.dumps(make_chunk(f'🔄 **Revising plan based on feedback...**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Feedback:** {feedback}\n\n'))}\n\n"

                # Have models revise the plan
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    icon = "🔵" if i == 0 else "🟡"

                    yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} revising...**\n\n'))}\n\n"

                    revision_prompt = f"""The user provided this feedback on the implementation plan:

"{feedback}"

Previous plan:
{state.implementation_plan}

Revise the plan based on this feedback. Address the user's concerns and improve the approach.

Your revised plan:"""

                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=revision_prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    if i == 0:
                        state.implementation_plan = f"{model_name}:\n{response}\n\n"
                    else:
                        state.implementation_plan += f"{model_name}:\n{response}\n\n"

                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk('\n**Revised plan complete!**\n\nType **\"approve\"** to proceed, **\"revise: <more feedback>\"** to revise again, or **\"cancel\"** to abort.\n\n'))}\n\n"

            elif user_command == "cancel":
                logger.info(f"[{trace}] implementation.cancelled | chat_id={chat_id}")
                yield f"data: {json.dumps(make_chunk('❌ **Implementation cancelled.**\n\n'))}\n\n"
                clear_discussion_state(chat_id)

            else:
                yield f"data: {json.dumps(make_chunk(f'**Unknown command:** "{user_command}"\n\nValid commands: **approve**, **revise: <feedback>**, **deploy**, **cancel**\n\n'))}\n\n"

        elif state.stage == DiscussionStage.SYNTHESIS:
            user_command = user_input.lower().strip()
            logger.info(f"[{trace}] synthesis.command | chat_id={chat_id} cmd={user_command} stage={state.stage.value}")

            if user_command == "export":
                # Export with AI-generated summary
                logger.info(f"[{trace}] synthesis.export | chat_id={chat_id} round={state.current_round} entries={len(state.discussion_history)}")
                yield f"data: {json.dumps(make_chunk('📋 **Discussion Export**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Topic:** {state.topic}\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Mode:** {state.mode.title()}\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Rounds:** {state.current_round}\n\n'))}\n\n"

                yield f"data: {json.dumps(make_chunk('---\n\n## Round-by-Round Discussion\n\n'))}\n\n"

                for entry in state.discussion_history:
                    model_name = state.get_model_name(entry["model"])
                    response_text = entry["response"]
                    round_info = entry.get("stage", "")
                    yield f"data: {json.dumps(make_chunk(f'**{model_name}** ({round_info}):\n{response_text}\n\n'))}\n\n"

                # Generate AI summary using Claude
                yield f"data: {json.dumps(make_chunk('---\n\n## 🤖 AI-Generated Summary\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('*Analyzing discussion...*\n\n'))}\n\n"

                # Build summary prompt with full discussion history
                summary_parts = [f"Summarize this discussion about: {state.topic}\n\n"]
                summary_parts.append(f"Mode: {state.mode}\n")
                summary_parts.append(f"Total rounds: {state.current_round}\n\n")
                summary_parts.append("Discussion transcript:\n")

                for entry in state.discussion_history:
                    model_name = state.get_model_name(entry["model"])
                    response_text = entry["response"]
                    # Truncate very long responses for summary prompt
                    if len(response_text) > 1000:
                        response_text = response_text[:1000] + "... [truncated]"
                    summary_parts.append(f"{model_name}: {response_text}\n\n")

                summary_parts.append("""
Please provide:
1. **Key Points of Agreement**: What did both models agree on?
2. **Key Points of Disagreement**: Where did they differ?
3. **Evolution**: How did positions change across rounds?
4. **Conclusion**: What's the synthesized recommendation or outcome?

Keep it concise (3-5 paragraphs).""")

                summary_prompt = "".join(summary_parts)

                # Call Claude to generate summary
                try:
                    summary_response, _ = await call_model_prompt(
                        model_id="claude-cli",
                        prompt=summary_prompt,
                        session_id=None,  # Don't use discussion session
                        history_prompt=None,
                        chat_id=chat_id + "_summary",  # Different chat for summary
                        trace=trace,
                    )

                    yield f"data: {json.dumps(make_chunk(summary_response))}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps(make_chunk(f'*Summary generation failed: {str(e)}*'))}\n\n"

                yield f"data: {json.dumps(make_chunk('\n---\n\n*Export complete. Discussion ended.*\n\n'))}\n\n"
                clear_discussion_state(chat_id)
            else:
                # Unexpected input in SYNTHESIS stage
                logger.warning(f"[{trace}] synthesis.unexpected | chat_id={chat_id} input='{user_command}' expected='export'")
                yield f"data: {json.dumps(make_chunk(f'**Invalid command in synthesis stage:** "{user_command}"\n\nPlease type exactly **\"export\"** to generate summary.\n\n'))}\n\n"

    except Exception as e:
        error_msg = f"Discussion error: {str(e)}"
        _emit_log(trace, "interactive_discussion.error", str(e), level=logging.ERROR)
        yield f"data: {json.dumps(make_chunk(error_msg))}\n\n"

    yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BRIDGE_PORT, log_level="info")
