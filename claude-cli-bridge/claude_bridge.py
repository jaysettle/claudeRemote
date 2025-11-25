#!/usr/bin/env python3
"""
Claude CLI Bridge for Open WebUI
Provides an OpenAI-compatible API that proxies requests to Claude CLI via tmux
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from typing import AsyncGenerator, Optional, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# File upload directory
UPLOAD_DIR = Path(tempfile.gettempdir()) / "claude_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Claude CLI Bridge", version="1.0.0")

# Configuration
CLAUDE_CLI_PATH = os.getenv("CLAUDE_CLI_PATH", "claude")
TMUX_SESSION_PREFIX = "claude_cli_"
CLAUDE_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "300"))  # 5 minutes default


# Pydantic models for OpenAI-compatible API
class ImageUrl(BaseModel):
    url: str

class ContentPart(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]  # Can be string or list of content parts


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None


def save_base64_file(data_url: str) -> Optional[str]:
    """Save a base64 data URL to a temp file and return the path"""
    try:
        # Parse data URL: data:image/png;base64,xxxxx
        if not data_url.startswith('data:'):
            return None

        # Extract mime type and base64 data
        header, b64_data = data_url.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]

        # Determine file extension
        ext_map = {
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'application/pdf': '.pdf',
            'text/plain': '.txt',
            'text/markdown': '.md',
            'application/json': '.json',
            'text/csv': '.csv',
        }
        ext = ext_map.get(mime_type, '.bin')

        # Save to temp file
        file_id = str(uuid.uuid4())[:8]
        file_path = UPLOAD_DIR / f"upload_{file_id}{ext}"

        with open(file_path, 'wb') as f:
            f.write(base64.b64decode(b64_data))

        logger.info(f"Saved uploaded file to {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving base64 file: {e}")
        return None


def process_message_content(content: Union[str, List[ContentPart]]) -> tuple[str, List[str]]:
    """Process message content and extract text and file paths"""
    if isinstance(content, str):
        return content, []

    text_parts = []
    file_paths = []

    for part in content:
        if isinstance(part, dict):
            part_type = part.get('type', '')
            if part_type == 'text':
                text_parts.append(part.get('text', ''))
            elif part_type == 'image_url':
                image_url = part.get('image_url', {})
                url = image_url.get('url', '') if isinstance(image_url, dict) else ''
                if url.startswith('data:'):
                    file_path = save_base64_file(url)
                    if file_path:
                        file_paths.append(file_path)
        elif hasattr(part, 'type'):
            if part.type == 'text' and part.text:
                text_parts.append(part.text)
            elif part.type == 'image_url' and part.image_url:
                url = part.image_url.url
                if url.startswith('data:'):
                    file_path = save_base64_file(url)
                    if file_path:
                        file_paths.append(file_path)

    return ' '.join(text_parts), file_paths


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "claude-cli"


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]


class TmuxSession:
    """Manages tmux sessions for Claude CLI"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_name = f"{TMUX_SESSION_PREFIX}{session_id}"
        self.is_first_request = True

    async def create_session(self) -> bool:
        """Create a new tmux session with Claude CLI"""
        try:
            # Check if session already exists
            result = await asyncio.create_subprocess_exec(
                "tmux", "has-session", "-t", self.session_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            if result.returncode == 0:
                logger.info(f"Tmux session {self.session_name} already exists")
                self.is_first_request = False
                return True

            # Create new session
            cmd = [
                "tmux", "new-session", "-d", "-s", self.session_name,
                CLAUDE_CLI_PATH, "--dangerously-skip-permissions"
            ]

            logger.info(f"Creating tmux session: {' '.join(cmd)}")
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            if result.returncode != 0:
                stderr = await result.stderr.read()
                raise Exception(f"Failed to create tmux session: {stderr.decode()}")

            # Give Claude a moment to initialize
            await asyncio.sleep(3)

            self.is_first_request = True
            return True

        except Exception as e:
            logger.error(f"Error creating tmux session: {e}")
            return False

    async def send_input(self, text: str) -> bool:
        """Send text input to the tmux session"""
        try:
            # If not first request, use --continue flag
            if not self.is_first_request:
                # Send continuation command
                cmd = [
                    "tmux", "send-keys", "-t", self.session_name,
                    f"{CLAUDE_CLI_PATH} --continue --dangerously-skip-permissions", "Enter"
                ]
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await result.wait()
                await asyncio.sleep(1)

            # Send the actual input text (literal mode)
            cmd = ["tmux", "send-keys", "-t", self.session_name, "-l", text]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            if result.returncode != 0:
                stderr = await result.stderr.read()
                raise Exception(f"Failed to send input text: {stderr.decode()}")

            # Send Enter key separately
            cmd = ["tmux", "send-keys", "-t", self.session_name, "Enter"]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            if result.returncode != 0:
                stderr = await result.stderr.read()
                raise Exception(f"Failed to send Enter: {stderr.decode()}")

            self.is_first_request = False
            return True

        except Exception as e:
            logger.error(f"Error sending input to tmux: {e}")
            return False

    async def capture_output(self, timeout: int = CLAUDE_TIMEOUT) -> str:
        """Capture output from the tmux session"""
        try:
            start_time = time.time()

            # First, capture initial output
            cmd = ["tmux", "capture-pane", "-t", self.session_name, "-p"]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            initial_output = stdout.decode()

            # Wait for output to CHANGE (indicates Claude is processing)
            last_output = initial_output
            output_changed = False

            while time.time() - start_time < timeout:
                await asyncio.sleep(1)

                cmd = ["tmux", "capture-pane", "-t", self.session_name, "-p"]
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                current_output = stdout.decode()

                if current_output != initial_output:
                    output_changed = True

                # Only check for stability after output has changed
                if output_changed:
                    if current_output == last_output:
                        # Output stable - Claude is done
                        return current_output

                last_output = current_output

            return last_output

        except Exception as e:
            logger.error(f"Error capturing tmux output: {e}")
            return ""

    async def close(self):
        """Close the tmux session"""
        try:
            cmd = ["tmux", "kill-session", "-t", self.session_name]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            logger.info(f"Closed tmux session {self.session_name}")
        except Exception as e:
            logger.error(f"Error closing tmux session: {e}")


# Session management
active_sessions: Dict[str, TmuxSession] = {}


def get_or_create_session(session_id: Optional[str] = None) -> TmuxSession:
    """Get existing session or create a new one"""
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in active_sessions:
        active_sessions[session_id] = TmuxSession(session_id)

    return active_sessions[session_id]


# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Claude CLI Bridge API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/v1/models")
async def list_models() -> ModelList:
    """List available models (OpenAI-compatible)"""
    return ModelList(
        data=[
            Model(
                id="claude-cli",
                object="model",
                owned_by="anthropic"
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests (OpenAI-compatible)"""
    try:
        # Extract the user's message and any file attachments
        user_message = None
        file_paths = []

        for msg in reversed(request.messages):
            if msg.role == "user":
                text, files = process_message_content(msg.content)
                user_message = text
                file_paths.extend(files)
                break

        if not user_message and not file_paths:
            raise HTTPException(status_code=400, detail="No user message found")

        # If files were uploaded, add them to the message
        if file_paths:
            file_references = "\n".join([f"[Attached file: {fp}]" for fp in file_paths])
            if user_message:
                user_message = f"{user_message}\n\n{file_references}\n\nPlease read and analyze the attached file(s)."
            else:
                user_message = f"{file_references}\n\nPlease read and analyze the attached file(s)."
            logger.info(f"Processing request with {len(file_paths)} file(s)")
        else:
            logger.info(f"Processing request: {user_message[:100] if user_message else 'empty'}...")

        # Get or create session (using a default session for simplicity)
        session = get_or_create_session("default")

        # Create session if needed
        if not await session.create_session():
            raise HTTPException(status_code=500, detail="Failed to create Claude session")

        # Send input to Claude
        if not await session.send_input(user_message):
            raise HTTPException(status_code=500, detail="Failed to send input to Claude")

        # Capture output
        output = await session.capture_output()

        if not output:
            raise HTTPException(status_code=500, detail="No response from Claude")

        # Clean up the output (remove prompts and formatting)
        response_text = clean_claude_output(output)

        # Return in OpenAI format
        if request.stream:
            return StreamingResponse(
                stream_response(response_text),
                media_type="text/event-stream"
            )
        else:
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
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
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(user_message.split()) + len(response_text.split())
                }
            }

    except Exception as e:
        logger.error(f"Error processing chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(text: str) -> AsyncGenerator[str, None]:
    """Stream response in OpenAI SSE format"""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    # Stream the text word by word
    words = text.split()
    for i, word in enumerate(words):
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "claude-cli",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": word + " " if i < len(words) - 1 else word
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)  # Small delay for streaming effect

    # Send final chunk
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "claude-cli",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def clean_claude_output(output: str) -> str:
    """Clean up Claude CLI output to extract just the response"""
    import re
    lines = output.split('\n')

    response_lines = []
    in_response = False
    skip_until_next_section = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            if in_response and response_lines:
                response_lines.append('')  # Preserve paragraph breaks
            continue

        # Skip box drawing characters and UI chrome
        if any(c in stripped for c in ['╭', '╮', '╯', '╰', '│', '─', '━', '⏵', '├', '└']):
            continue

        # Skip prompts and commands
        if stripped.startswith('>') or stripped.startswith('$'):
            continue
        if '/usr/local/bin/claude' in stripped or '--dangerously-skip-permissions' in stripped:
            continue

        # Skip tool invocations and their output
        if re.match(r'^(Write|Read|Edit|Bash|Glob|Grep|Task|TodoWrite)\s*\(', stripped):
            skip_until_next_section = True
            continue

        # Skip status messages
        if re.match(r'^Retrieved \d+ sources?', stripped):
            continue
        if re.match(r'^\[Pasted text #\d+', stripped):
            continue

        # Skip tips
        if stripped.lower().startswith('tip:'):
            continue

        # Skip status/info lines
        if any(x in stripped.lower() for x in ['bypass permissions', 'auto-update failed',
               'shift+tab', 'ctrl-g', 'thinking off', 'welcome back', 'opus 4.5',
               'tips for getting', 'recent activity', 'no recent activity',
               'run /init', 'processing', 'esc to interrupt', '/terminal-setup',
               'wrote', 'created', 'updated', 'deleted', 'working as expected',
               '--continue', '--resume', 'resume a conversation']):
            continue

        # Skip lines that are just decoration
        if re.match(r'^[▐▛█▜▌▘▝\s]+$', stripped):
            continue

        # Claude's actual responses start with ● or ⎿
        if stripped.startswith('●') or stripped.startswith('⎿'):
            in_response = True
            skip_until_next_section = False
            # Remove the bullet and leading space
            clean_line = re.sub(r'^[●⎿]\s*', '', stripped)
            # Skip if it's a tool invocation
            if not re.match(r'^(Write|Read|Edit|Bash|Glob|Grep|Task|TodoWrite)\s*\(', clean_line):
                response_lines.append(clean_line)
        elif in_response and not skip_until_next_section:
            # Continue capturing multi-line responses
            # Stop if we hit another prompt
            if stripped.startswith('>'):
                break
            # Skip tool-related output
            if not re.match(r'^(Write|Read|Edit|Bash|Glob|Grep|Task|TodoWrite)\s*\(', stripped):
                response_lines.append(stripped)

    # Join and clean up
    response = '\n'.join(response_lines).strip()

    # Remove trailing empty lines
    while response.endswith('\n\n'):
        response = response[:-1]

    # If still empty, try to extract any meaningful text
    if not response:
        # Fallback: look for any text that's not a command or status
        for line in lines:
            stripped = line.strip()
            if stripped and not any(x in stripped for x in ['●', '⎿', '>', '$', '/', 'claude', 'Write(', 'Read(', 'Bash(', 'Tip:']):
                if len(stripped) > 20 and not stripped.startswith('['):
                    response = stripped
                    break

    return response if response else "No response generated"


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up sessions on shutdown"""
    logger.info("Shutting down, closing all sessions...")
    for session in active_sessions.values():
        await session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
