"""
CLI Streaming Module for Claude CLI Bridge
Handles streaming responses from various CLI tools (Claude, Codex, Gemini)

NOTE: Due to the size and complexity of the streaming functions (~1500+ lines),
      these functions remain in the main claude_bridge.py file for now.
      A future refactoring iteration should extract them into this module.

Functions that should eventually be moved here:
- normalize_codex_response()
- is_followup_prompt()
- extract_followups()
- is_error_response()
- run_claude_prompt()
- run_codex_prompt()
- run_gemini_prompt()
- stream_claude_api_with_tools()
- stream_claude_api()
- run_claude_api_prompt()
- call_model_prompt()
- stream_codex_incremental()
- stream_claude_incremental()
- stream_chat_response()
- stream_interactive_discussion()
- handle_interactive_discussion()
"""

import json
import logging
from typing import List

logger = logging.getLogger(__name__)


# ============================================================================
# Response Processing Utilities
# ============================================================================

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


# All async streaming functions remain in claude_bridge.py for now
# Future refactoring should move them here
