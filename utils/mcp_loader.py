#!/usr/bin/env python3
"""
MCP Configuration Loader
Dynamically loads MCP config from Claude CLI's config file
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import CLAUDE_PROJECT_PATH, CLAUDE_DISABLE_MCP, MCP_CONFIG_CACHE_TTL

logger = logging.getLogger(__name__)

# Cache for MCP config (reloaded periodically)
_mcp_config_cache = None
_mcp_config_cache_time = 0


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
