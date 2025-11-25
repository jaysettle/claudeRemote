#!/usr/bin/env python3
"""
Claude CLI Bridge for Open WebUI
Provides an OpenAI-compatible API that proxies requests to Claude CLI
Supports persistent sessions per chat thread

v1.6.0 - Minimal logging
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

app = FastAPI(title="Claude CLI Bridge", version="1.6.0")

# Configuration
CLAUDE_CLI_PATH = os.getenv("CLAUDE_CLI_PATH", "claude")
CLAUDE_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "600"))  # 10 minutes default

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

        # Look for recently modified .jsonl files (sessions are stored as UUID.jsonl)
        cmd = ["find", str(projects_dir), "-name", "*.jsonl", "-mmin", "-2", "-type", "f"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()

        if stdout:
            files = [f for f in stdout.decode().strip().split('\n') if f]
            if files:
                filename = Path(files[0]).stem
                if re.match(r'^[a-f0-9-]{36}$', filename):
                    return filename
                # Try to find UUID in the path
                match = re.search(r'/([a-f0-9-]{36})\.jsonl$', files[0])
                if match:
                    return match.group(1)
        return None
    except Exception as e:
        logger.error(f"Error finding session ID: {e}")
        return None


async def run_claude_prompt(prompt: str, session_id: Optional[str] = None, timeout: int = CLAUDE_TIMEOUT) -> tuple[str, Optional[str]]:
    """
    Run Claude CLI prompt and return (response, new_session_id).
    If session_id is provided, resumes that session.
    """
    try:
        cmd = [CLAUDE_CLI_PATH]
        if session_id:
            cmd.extend(["--resume", session_id])
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

        stderr_text = stderr.decode() if stderr else ""
        if process.returncode != 0:
            error_msg = stderr_text or "Unknown error"
            # Retry without session if session not found
            if session_id and ("session" in error_msg.lower() or "not found" in error_msg.lower()):
                logger.warning(f"Session {session_id} not found, starting new")
                return await run_claude_prompt(prompt, session_id=None, timeout=timeout)
            raise Exception(f"Claude CLI failed: {error_msg}")

        response = stdout.decode().strip()
        new_session_id = await find_latest_session_id() if not session_id else session_id
        return response, new_session_id

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

        # Build the full prompt
        prompt_parts = []

        # Include conversation history if there's prior context
        if len(conversation_history) > 1:
            prompt_parts.append("[Conversation history]")
            # All messages except the last one (which is the current message)
            for entry in conversation_history[:-1]:
                prompt_parts.append(entry)
            prompt_parts.append("")
            prompt_parts.append("[Current message]")
            prompt_parts.append(conversation_history[-1])
        else:
            # Single message, no history needed
            prompt_parts.append(current_user_message or "")

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

        # Single concise log line for request processing
        msg_preview = (current_user_message[:40] + "...") if current_user_message and len(current_user_message) > 40 else (current_user_message or "file")
        session_status = f"resume:{claude_session_id[:8]}" if claude_session_id else "new"
        files_info = f"+{len(binary_paths)}img" if binary_paths else ""
        logger.info(f"Request: {msg_preview} [{session_status}] {files_info}")

        # Return in OpenAI format
        if body.stream:
            # Streaming mode - shows thinking indicator while Claude works
            return StreamingResponse(
                stream_with_thinking(final_message, chat_id, claude_session_id),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming mode - wait for full response
            response_text, new_session_id = await run_claude_prompt(final_message, claude_session_id)
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
    """Stream response in OpenAI SSE format"""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    words = text.split()
    for i, word in enumerate(words):
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "claude-cli",
            "choices": [{
                "index": 0,
                "delta": {"content": word + " " if i < len(words) - 1 else word},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.02)

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


async def stream_with_thinking(prompt: str, chat_id: str, session_id: Optional[str] = None, timeout: int = CLAUDE_TIMEOUT) -> AsyncGenerator[str, None]:
    """Stream response with thinking indicator while Claude works"""
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

    # Build Claude CLI command
    cmd = [CLAUDE_CLI_PATH]
    if session_id:
        cmd.extend(["--resume", session_id])
    cmd.extend(["-p", prompt, "--dangerously-skip-permissions"])

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Show thinking indicator while waiting
    yield f"data: {json.dumps(make_chunk('⏳ '))}\n\n"

    dots = 0
    while True:
        try:
            # Check if process is done (wait 2 seconds between dots)
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=2.0
            )
            # Process finished - got the response
            response = stdout.decode().strip()
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                # Check if it's a session not found error
                if session_id and ("session" in error_msg.lower() or "not found" in error_msg.lower()):
                    logger.warning(f"Session {session_id[:8]} not found")
                response = f"Error: {error_msg}"

            # Save session mapping if this was a new session
            if not session_id and chat_id:
                new_session_id = await find_latest_session_id()
                if new_session_id:
                    session_map[chat_id] = new_session_id
                    save_session_map()
                    logger.info(f"New session: {new_session_id[:8]}")

            # Clear thinking indicator and send response
            yield f"data: {json.dumps(make_chunk(chr(8) * (dots + 3)))}\n\n"  # Backspaces

            # Stream the actual response
            words = response.split()
            for i, word in enumerate(words):
                yield f"data: {json.dumps(make_chunk(word + (' ' if i < len(words) - 1 else '')))}\n\n"
                await asyncio.sleep(0.02)

            break

        except asyncio.TimeoutError:
            # Still running - add a dot
            dots += 1
            yield f"data: {json.dumps(make_chunk('.'))}\n\n"

            if dots * 2 > timeout:
                process.kill()
                yield f"data: {json.dumps(make_chunk(' (timed out)'))}\n\n"
                break

    yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
