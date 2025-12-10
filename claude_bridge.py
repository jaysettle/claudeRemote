#!/usr/bin/env python3
"""
Claude/Codex CLI Bridge for Open WebUI
Provides an OpenAI-compatible API that proxies requests to CLI agents
Supports persistent sessions per chat thread

v1.12.2-dev - Refactored modular architecture
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

# Import configuration
from config import (
    VERSION, BRIDGE_PORT, SUPPORTED_MODELS, ANTHROPIC_AVAILABLE,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MCP_SYSTEM_PROMPT
)

# Import models
from models import (
    Message, ChatCompletionRequest, Model, ModelList
)

# Import session management
from session_manager import (
    load_session_map, get_session_id, set_session_id, 
    clear_session_id, get_all_sessions
)

# Import utilities
from utils.file_utils import process_message_content
from utils.mcp_loader import get_mcp_config
from utils.helpers import make_trace_logger, emit_log as _emit_log

# Import CLI adapters
from cli_adapters.claude_adapter import (
    run_claude_prompt, stream_claude_incremental, SessionNotFoundError
)
from cli_adapters.codex_adapter import (
    run_codex_prompt, stream_codex_incremental, CodexSessionNotFoundError,
    is_followup_prompt
)
from cli_adapters.gemini_adapter import (
    run_gemini_prompt, GeminiSessionNotFoundError
)
from cli_adapters.vibe_adapter import (
    run_vibe_prompt, VibeSessionNotFoundError
)

# Import discussion handling
from discussions.stream_handler import handle_interactive_discussion

# Import tools (for Claude API mode)
from tools import CLAUDE_API_TOOLS, execute_tool

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Claude CLI Bridge", version=VERSION)

# Load sessions on startup
load_session_map()


# ============================================================================
# Helper Functions
# ============================================================================

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
    elif model_id == "devstral-small-2":
        try:
            return await run_vibe_prompt(prompt, session_id, trace=trace)
        except VibeSessionNotFoundError:
            clear_session_id(model_id, chat_id)
            fallback_prompt = history_prompt if history_prompt else prompt
            _emit_log(trace, "vibe.retry.history", "retrying without resume after session miss")
            return await run_vibe_prompt(fallback_prompt, session_id=None, trace=trace)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_id}")


async def stream_chat_response(
    model_id: str,
    chat_id: str,
    session_id: Optional[str],
    prompt: str,
    history_prompt: Optional[str],
    trace: Optional[Callable],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[str, None]:
    """Generic streaming for non-incremental models (Gemini, Claude API)"""
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
        
        # Update session if changed
        if new_session_id and new_session_id != session_id:
            set_session_id(model_id, chat_id, new_session_id)
            _emit_log(trace, "session.map.updated", f"model={model_id} chat_id={chat_id} session={new_session_id}")
        
        # Stream line by line to preserve formatting
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
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Claude CLI Bridge API",
        "version": VERSION,
        "status": "running",
        "active_sessions": sum(len(m) for m in get_all_sessions().values())
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
    return {"sessions": get_all_sessions()}


@app.get("/mcp")
async def get_mcp_status():
    """Show current MCP configuration (debug endpoint)"""
    from config import CLAUDE_PROJECT_PATH
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
    from discussions.state import discussion_states
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
        if chat_id in get_all_sessions().get(mid, {}):
            clear_session_id(mid, chat_id)
            removed_models.append(mid)

    if removed_models:
        return {"status": "deleted", "chat_id": chat_id, "models": removed_models}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Handle chat completion requests (OpenAI-compatible)"""
    trace_id, trace = make_trace_logger()
    try:
        model_id = body.model or "claude-cli"
        if model_id not in SUPPORTED_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_id}")

        # Handle interactive discussion mode
        if model_id == "interactive-discussion":
            return await handle_interactive_discussion(request, body)

        # Get chat ID and check session mapping
        chat_id = get_chat_id(request, body.messages)
        session_id = get_session_id(model_id, chat_id)
        
        # Detect prior sessions for other models (model switch scenario)
        other_models = [
            mid for mid in SUPPORTED_MODELS.keys()
            if mid != model_id and chat_id in get_all_sessions().get(mid, {})
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
            max_history_messages = 10
            recent_history = conversation_history[-max_history_messages:] if len(conversation_history) > max_history_messages else conversation_history

            history_parts = ["[Conversation history - last {} messages]".format(len(recent_history))]
            for entry in recent_history[:-1]:
                history_parts.append(entry)
            history_parts.append("")
            history_parts.append("[Current message]")
            history_parts.append(recent_history[-1])
            history_prompt = "\n".join(history_parts)

        # Use current-only when session exists, history when no session
        prompt_parts = []
        if model_id in ["gemini-cli", "devstral-small-2"]:
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

        # Prepend system prompt for new sessions
        if not session_id:
            final_message = f"[SYSTEM]\n{MCP_SYSTEM_PROMPT}\n[END SYSTEM]\n\n{final_message}"

        # Check if prompt is too large for command line args (>100KB)
        prompt_size = len(final_message.encode('utf-8'))
        if prompt_size > 100_000:
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
            # Non-streaming response
            # Use codex-cli for follow-up generation (faster)
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


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BRIDGE_PORT)
