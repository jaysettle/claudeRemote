#!/usr/bin/env python3
"""
Configuration and Constants for Claude CLI Bridge
Centralizes all environment variables, paths, and configuration settings
"""

import os
import tempfile
from pathlib import Path
from typing import List
import logging

# ============================================================================
# Version and Basic Configuration
# ============================================================================

VERSION = "1.12.2-dev"
BRIDGE_PORT = int(os.getenv("CLAUDE_BRIDGE_PORT", "8000"))

# ============================================================================
# File and Session Storage
# ============================================================================

# File upload directory (env override for dev/prod separation)
UPLOAD_DIR = Path(os.getenv("CLAUDE_UPLOAD_DIR", Path(tempfile.gettempdir()) / "claude_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Session storage directory (env override for dev/prod separation)
SESSION_DIR = Path(os.getenv("CLAUDE_SESSION_DIR", Path(tempfile.gettempdir()) / "claude_sessions"))
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SESSION_MAP_FILE = SESSION_DIR / "session_map.json"

# ============================================================================
# CLI Paths and Timeouts
# ============================================================================

CLAUDE_CLI_PATH = os.getenv("CLAUDE_CLI_PATH", "claude")
CLAUDE_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "600"))  # 10 minutes default
CLAUDE_PROJECT_PATH = os.getenv("CLAUDE_PROJECT_PATH", str(Path.home()))  # Project path for MCP config
CLAUDE_DISABLE_MCP = os.getenv("CLAUDE_DISABLE_MCP", "1") == "1"  # Default: disable MCP to avoid slow startup

CODEX_CLI_PATH = os.getenv("CODEX_CLI_PATH", str(Path.home() / ".npm-global" / "bin" / "codex"))
CODEX_TIMEOUT = int(os.getenv("CODEX_TIMEOUT", str(CLAUDE_TIMEOUT)))

GEMINI_CLI_PATH = os.getenv("GEMINI_CLI_PATH", "/usr/local/bin/gemini")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", str(CLAUDE_TIMEOUT)))

# ============================================================================
# MCP Configuration
# ============================================================================

MCP_CONFIG_CACHE_TTL = 60  # Reload config every 60 seconds

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

# ============================================================================
# Supported Models
# ============================================================================

SUPPORTED_MODELS = {
    "claude-cli": {"owned_by": "anthropic"},
    # "claude-api": {"owned_by": "anthropic-api"},  # DISABLED
    "codex-cli": {"owned_by": "openai-codex"},
    "gemini-cli": {"owned_by": "google-gemini"},
    "devstral-small-2": {"owned_by": "mistral-ai"},
    "interactive-discussion": {"owned_by": "collaborative"},
}

# ============================================================================
# Anthropic API Configuration
# ============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Check if anthropic is available
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# ============================================================================
# Implementation Safety Configuration
# ============================================================================

def _parse_allowed_roots(raw: str) -> List[Path]:
    """Parse colon-separated list of allowed root paths"""
    roots: List[Path] = []
    for part in raw.split(os.pathsep):
        if not part:
            continue
        try:
            roots.append(Path(part).expanduser().resolve())
        except Exception as exc:
            logging.warning(f"Failed to parse allowed root '{part}': {exc}")
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

# ============================================================================
# Logging Configuration
# ============================================================================

# Default to DEBUG so stream tracing is visible when tailing logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
