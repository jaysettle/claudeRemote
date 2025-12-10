#!/usr/bin/env python3
"""
Session Management for Claude CLI Bridge
Handles persistent session mappings between chat IDs and CLI session IDs
"""

import json
import logging
from typing import Dict, Optional

from config import SESSION_MAP_FILE, SUPPORTED_MODELS

logger = logging.getLogger(__name__)

# In-memory session map: model_id -> {chat_id -> session_id}
session_map: Dict[str, Dict[str, str]] = {model: {} for model in SUPPORTED_MODELS.keys()}


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


def get_all_sessions() -> Dict[str, Dict[str, str]]:
    """Get all session mappings (for debug endpoint)."""
    return session_map
