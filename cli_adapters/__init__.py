"""CLI Adapters for different AI models"""
from .claude_adapter import (
    run_claude_prompt, find_latest_claude_session_id, 
    stream_claude_incremental, SessionNotFoundError
)
from .codex_adapter import (
    run_codex_prompt, find_latest_codex_session_id,
    stream_codex_incremental, CodexSessionNotFoundError,
    normalize_codex_response, is_followup_prompt, extract_followups
)
from .gemini_adapter import (
    run_gemini_prompt, find_latest_gemini_session_id,
    GeminiSessionNotFoundError
)

__all__ = [
    'run_claude_prompt', 'find_latest_claude_session_id', 'stream_claude_incremental',
    'SessionNotFoundError', 'run_codex_prompt', 'find_latest_codex_session_id',
    'stream_codex_incremental', 'CodexSessionNotFoundError', 'normalize_codex_response',
    'is_followup_prompt', 'extract_followups', 'run_gemini_prompt', 
    'find_latest_gemini_session_id', 'GeminiSessionNotFoundError'
]
