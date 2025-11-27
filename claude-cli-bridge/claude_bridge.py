#!/usr/bin/env python3
"""
Claude CLI Bridge for Open WebUI
Provides an OpenAI-compatible API that proxies requests to Claude CLI
Supports persistent sessions per chat thread

v1.8.0 - Dynamic MCP config loading from Claude CLI config file
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from typing import AsyncGenerator, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# File upload directory
UPLOAD_DIR = Path(tempfile.gettempdir()) / "claude_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Session storage directory
SESSION_DIR = Path(tempfile.gettempdir()) / "claude_sessions"
SESSION_DIR.mkdir(exist_ok=True)
SESSION_MAP_FILE = SESSION_DIR / "session_map.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Claude CLI Bridge", version="1.8.0")

# Configuration
CLAUDE_CLI_PATH = os.getenv("CLAUDE_CLI_PATH", "claude")
CLAUDE_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "600"))  # 10 minutes default
CLAUDE_PROJECT_PATH = os.getenv("CLAUDE_PROJECT_PATH", str(Path.home()))  # Project path for MCP config

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

# In-memory session map: chat_id -> claude_session_id
session_map: Dict[str, str] = {}


def load_session_map():
    """Load session map from disk."""
    global session_map
    try:
        if SESSION_MAP_FILE.exists():
            with open(SESSION_MAP_FILE, 'r') as f:
                session_map = json.load(f)
            logger.info(f"Loaded {len(session_map)} sessions from disk")
    except Exception as e:
        logger.error(f"Error loading session map: {e}")
        session_map = {}


def save_session_map():
    """Save session map to disk."""
    try:
        with open(SESSION_MAP_FILE, 'w') as f:
            json.dump(session_map, f)
    except Exception as e:
        logger.error(f"Error saving session map: {e}")


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
    owned_by: str = "anthropic"


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]


async def find_latest_session_id() -> Optional[str]:
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
                if re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', filename):
                    return filename
        return None
    except Exception as e:
        logger.error(f"Error finding session ID: {e}")
        return None


class SessionNotFoundError(Exception):
    """Raised when Claude CLI session is not found."""
    pass


def is_error_response(text: str) -> bool:
    """Check if response is an error (Claude CLI puts errors in stdout with exit code 0)"""
    error_patterns = [
        "No conversation found with session ID:",
        "Error:",
        "not a valid UUID",
    ]
    return any(pattern in text for pattern in error_patterns)


async def run_claude_prompt(prompt: str, session_id: Optional[str] = None, timeout: int = CLAUDE_TIMEOUT) -> tuple[str, Optional[str]]:
    """
    Run Claude CLI prompt and return (response, new_session_id).
    If session_id is provided, resumes that session.
    Raises SessionNotFoundError if session doesn't exist (caller should retry with history).
    """
    try:
        cmd = [CLAUDE_CLI_PATH]
        if session_id:
            cmd.extend(["--resume", session_id])
        # Add MCP config if available (dynamically loaded from ~/.claude.json)
        mcp_config = get_mcp_config()
        if mcp_config:
            cmd.extend(["--mcp-config", json.dumps(mcp_config)])
        cmd.extend(["-p", prompt, "--dangerously-skip-permissions"])

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

        # Check for errors (Claude CLI returns exit 0 even on errors, puts error in stdout)
        if process.returncode != 0 or is_error_response(response):
            error_msg = stderr_text or response or "Unknown error"
            # Raise specific exception for session not found - caller will retry with history
            if session_id and ("session" in error_msg.lower() or "not found" in error_msg.lower() or "No conversation found" in response):
                logger.warning(f"Session {session_id[:8]} not found")
                raise SessionNotFoundError(session_id)
            raise Exception(f"Claude CLI failed: {error_msg}")

        new_session_id = await find_latest_session_id() if not session_id else session_id
        return response, new_session_id

    except SessionNotFoundError:
        raise  # Re-raise for caller to handle
    except Exception as e:
        logger.error(f"Claude CLI error: {e}")
        raise


# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Claude CLI Bridge API",
        "version": app.version,
        "status": "running",
        "active_sessions": len(session_map)
    }


@app.get("/v1/models")
async def list_models() -> ModelList:
    """List available models (OpenAI-compatible)"""
    return ModelList(
        data=[
            Model(id="claude-cli", object="model", owned_by="anthropic")
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
async def delete_session(chat_id: str):
    """Delete a session mapping"""
    if chat_id in session_map:
        del session_map[chat_id]
        save_session_map()
        return {"status": "deleted", "chat_id": chat_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Handle chat completion requests (OpenAI-compatible)"""
    try:
        # Get chat ID and check session mapping
        chat_id = get_chat_id(request, body.messages)
        claude_session_id = session_map.get(chat_id)

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

        # Use current-only when session exists, history when no session
        prompt_parts = []
        if claude_session_id:
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

        # Prepend MCP system prompt for new sessions (resuming sessions already have context)
        if not claude_session_id:
            final_message = f"[SYSTEM]\n{MCP_SYSTEM_PROMPT}\n[END SYSTEM]\n\n{final_message}"

        # Single concise log line for request processing
        msg_preview = (current_user_message[:40] + "...") if current_user_message and len(current_user_message) > 40 else (current_user_message or "file")
        session_status = f"resume:{claude_session_id[:8]}" if claude_session_id else "new"
        files_info = f"+{len(binary_paths)}img" if binary_paths else ""
        logger.info(f"Request: {msg_preview} [{session_status}] {files_info}")

        # Return in OpenAI format
        if body.stream:
            # Streaming mode - shows thinking indicator while Claude works
            return StreamingResponse(
                stream_with_thinking(final_message, chat_id, claude_session_id, history_prompt),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming mode - wait for full response
            try:
                response_text, new_session_id = await run_claude_prompt(final_message, claude_session_id)
            except SessionNotFoundError:
                # Session expired - clear mapping and retry with history
                if chat_id in session_map:
                    del session_map[chat_id]
                    save_session_map()
                # Rebuild prompt with history and retry (add MCP system prompt since this is effectively a new session)
                fallback_prompt = history_prompt if history_prompt else final_message
                fallback_prompt = f"[SYSTEM]\n{MCP_SYSTEM_PROMPT}\n[END SYSTEM]\n\n{fallback_prompt}"
                logger.info(f"Retrying with history after session expired")
                response_text, new_session_id = await run_claude_prompt(fallback_prompt, session_id=None)

            if not response_text:
                response_text = "No response generated"
            # Save session mapping if this was a new session
            if not claude_session_id and new_session_id and chat_id:
                session_map[chat_id] = new_session_id
                save_session_map()
                logger.info(f"New session: {new_session_id[:8]}")
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.model,
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
        logger.error(f"Error processing chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(text: str) -> AsyncGenerator[str, None]:
    """Stream response in OpenAI SSE format, preserving newlines"""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    # Stream line by line to preserve markdown formatting
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Add newline back except for last line
        content = line + '\n' if i < len(lines) - 1 else line
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "claude-cli",
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)

    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "claude-cli",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_with_thinking(prompt: str, chat_id: str, session_id: Optional[str] = None, history_prompt: Optional[str] = None, timeout: int = CLAUDE_TIMEOUT) -> AsyncGenerator[str, None]:
    """Stream response after Claude finishes"""
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

    current_prompt = prompt
    current_session = session_id

    # Try up to 2 times (original + retry with history)
    for attempt in range(2):
        cmd = [CLAUDE_CLI_PATH]
        if current_session:
            cmd.extend(["--resume", current_session])
        # Add MCP config if available (dynamically loaded from ~/.claude.json)
        mcp_config = get_mcp_config()
        if mcp_config:
            cmd.extend(["--mcp-config", json.dumps(mcp_config)])
        cmd.extend(["-p", current_prompt, "--dangerously-skip-permissions"])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            yield f"data: {json.dumps(make_chunk('Request timed out after ' + str(timeout) + ' seconds'))}\n\n"
            yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
            yield "data: [DONE]\n\n"
            return

        response = stdout.decode().strip()

        # Check for errors in stdout (Claude CLI returns exit 0 even on errors)
        if process.returncode != 0 or is_error_response(response):
            error_msg = stderr.decode() if stderr else response
            # Check if session not found - retry with history
            if current_session and ("session" in error_msg.lower() or "not found" in error_msg.lower() or "No conversation found" in response):
                logger.warning(f"Session {current_session[:8]} not found, retrying with history")
                if history_prompt and attempt == 0:
                    if chat_id in session_map:
                        del session_map[chat_id]
                        save_session_map()
                    # Add MCP system prompt since this is effectively a new session
                    current_prompt = f"[SYSTEM]\n{MCP_SYSTEM_PROMPT}\n[END SYSTEM]\n\n{history_prompt}"
                    current_session = None
                    continue  # Retry
            if is_error_response(response):
                response = f"Error: {response}"

        # Save session mapping if this was a new session
        if not current_session and chat_id:
            new_session_id = await find_latest_session_id()
            if new_session_id:
                session_map[chat_id] = new_session_id
                save_session_map()
                logger.info(f"New session: {new_session_id[:8]}")

        # Stream the response line by line to preserve markdown formatting
        lines = response.split('\n')
        for i, line in enumerate(lines):
            content = line + '\n' if i < len(lines) - 1 else line
            yield f"data: {json.dumps(make_chunk(content))}\n\n"
            await asyncio.sleep(0.01)

        yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Fallback if both attempts failed
    yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
