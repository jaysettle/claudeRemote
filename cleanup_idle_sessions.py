#!/usr/bin/env python3
"""
Claude CLI Session Cleanup Script
Removes idle Claude CLI sessions after 2 hours of inactivity
Safe to run with --resume since sessions can be restored
"""

import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
import os

# Configuration
SESSION_DIR = Path(os.getenv("CLAUDE_SESSION_DIR", Path("/tmp/claude_sessions")))
SESSION_MAP_FILE = SESSION_DIR / "session_map.json"
IDLE_THRESHOLD_HOURS = 2  # Close sessions idle for 2+ hours
DRY_RUN = False  # Set to True to see what would be cleaned without actually doing it

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_session_last_activity(session_id: str) -> float:
    """
    Get the last activity time for a Claude CLI session.
    Checks modification time of the session JSONL file.
    Returns timestamp or 0 if session file not found.
    """
    # Claude CLI stores sessions in ~/.claude/projects/<project-name>/
    home = Path.home()

    # Check common session locations
    session_paths = [
        home / ".claude" / "projects" / "-home-jay-claude-cli-bridge" / f"{session_id}.jsonl",
        home / ".claude" / "projects" / "-home-jay-claude-cli-bridge-dev" / f"{session_id}.jsonl",
        home / ".claude" / "projects" / "-home-jay" / f"{session_id}.jsonl",
    ]

    for session_path in session_paths:
        if session_path.exists():
            return session_path.stat().st_mtime

    # Session file doesn't exist - might be already cleaned up
    return 0


def get_gemini_session_last_activity(session_id: str) -> float:
    """
    Get the last activity time for a Gemini CLI session.
    Gemini stores sessions in ~/.gemini/tmp/*/chats/session-*.json
    """
    home = Path.home()
    gemini_tmp = home / ".gemini" / "tmp"

    if not gemini_tmp.exists():
        return 0

    # Search for session file in all subdirectories
    for chat_dir in gemini_tmp.glob("*/chats"):
        session_file = chat_dir / f"session-{session_id}.json"
        if session_file.exists():
            return session_file.stat().st_mtime

    return 0


def get_codex_session_last_activity(session_id: str) -> float:
    """
    Get the last activity time for a Codex CLI session.
    Returns timestamp or 0 if session file not found.
    """
    # Codex sessions are stored in ~/.codex/ or similar
    # Adjust path based on actual Codex CLI storage location
    home = Path.home()

    # Check common Codex session locations
    codex_paths = [
        home / ".codex" / "sessions" / f"{session_id}.jsonl",
        home / ".codex" / f"{session_id}.jsonl",
    ]

    for codex_path in codex_paths:
        if codex_path.exists():
            return codex_path.stat().st_mtime

    return 0


def load_session_map() -> dict:
    """Load the session map from disk."""
    if not SESSION_MAP_FILE.exists():
        logger.warning(f"Session map not found: {SESSION_MAP_FILE}")
        return {}

    try:
        with open(SESSION_MAP_FILE, 'r') as f:
            data = json.load(f)

        # Handle legacy format (single dict) vs new format (per-model dicts)
        if data and all(isinstance(v, str) for v in data.values()):
            # Legacy: {"chat_id": "session_id"}
            return {"claude-cli": data}
        else:
            # New: {"model_id": {"chat_id": "session_id"}}
            return data

    except Exception as e:
        logger.error(f"Error loading session map: {e}")
        return {}


def save_session_map(session_map: dict):
    """Save the updated session map to disk."""
    try:
        with open(SESSION_MAP_FILE, 'w') as f:
            json.dump(session_map, f, indent=2)
        logger.info(f"Saved updated session map to {SESSION_MAP_FILE}")
    except Exception as e:
        logger.error(f"Error saving session map: {e}")


def cleanup_idle_sessions():
    """Main cleanup function."""
    logger.info("=" * 80)
    logger.info("Claude CLI Session Cleanup - Starting")
    logger.info(f"Threshold: {IDLE_THRESHOLD_HOURS} hours")
    logger.info(f"Dry run: {DRY_RUN}")
    logger.info("=" * 80)

    # Load current session map
    session_map = load_session_map()
    if not session_map:
        logger.info("No sessions found to clean up")
        return

    # Calculate idle threshold timestamp
    idle_threshold = time.time() - (IDLE_THRESHOLD_HOURS * 3600)
    idle_threshold_dt = datetime.fromtimestamp(idle_threshold)
    logger.info(f"Sessions idle since before {idle_threshold_dt} will be cleaned")

    # Track cleanup statistics
    stats = {
        "total_sessions": 0,
        "idle_sessions": 0,
        "active_sessions": 0,
        "missing_sessions": 0,
        "cleaned_sessions": 0,
    }

    # Process each model's sessions
    cleaned_map = {}

    for model_id, chat_sessions in session_map.items():
        logger.info(f"\nProcessing {model_id} sessions ({len(chat_sessions)} chats)")
        cleaned_map[model_id] = {}

        for chat_id, session_id in chat_sessions.items():
            stats["total_sessions"] += 1

            # Get last activity based on model type
            if model_id == "gemini-cli":
                last_activity = get_gemini_session_last_activity(session_id)
            elif model_id == "codex-cli":
                last_activity = get_codex_session_last_activity(session_id)
            else:  # claude-cli and others
                last_activity = get_session_last_activity(session_id)

            if last_activity == 0:
                # Session file doesn't exist - already cleaned up or never created
                logger.warning(f"  ⚠️  {model_id}/{chat_id[:8]}: Session {session_id[:8]} file not found")
                stats["missing_sessions"] += 1
                # Don't keep this mapping since session doesn't exist
                continue

            last_activity_dt = datetime.fromtimestamp(last_activity)
            idle_hours = (time.time() - last_activity) / 3600

            if last_activity < idle_threshold:
                # Session is idle
                logger.info(
                    f"  🧹 {model_id}/{chat_id[:8]}: Session {session_id[:8]} idle for "
                    f"{idle_hours:.1f}h (last: {last_activity_dt})"
                )
                stats["idle_sessions"] += 1

                if not DRY_RUN:
                    # Session will be removed from map
                    # --resume will create a new session when needed
                    stats["cleaned_sessions"] += 1
                    logger.info(f"     ✂️  Removed mapping (can be restored with --resume)")
                else:
                    logger.info(f"     [DRY RUN] Would remove this mapping")
                    # In dry run, keep the mapping so we can see what would happen
                    cleaned_map[model_id][chat_id] = session_id
            else:
                # Session is active
                logger.debug(
                    f"  ✅ {model_id}/{chat_id[:8]}: Session {session_id[:8]} active "
                    f"(idle {idle_hours:.1f}h, last: {last_activity_dt})"
                )
                stats["active_sessions"] += 1
                cleaned_map[model_id][chat_id] = session_id

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Cleanup Summary:")
    logger.info(f"  Total sessions checked: {stats['total_sessions']}")
    logger.info(f"  Active sessions (kept): {stats['active_sessions']}")
    logger.info(f"  Idle sessions (>{IDLE_THRESHOLD_HOURS}h): {stats['idle_sessions']}")
    logger.info(f"  Missing session files: {stats['missing_sessions']}")

    if DRY_RUN:
        logger.info(f"  [DRY RUN] Would remove: {stats['idle_sessions'] + stats['missing_sessions']} mappings")
    else:
        logger.info(f"  Cleaned up: {stats['cleaned_sessions']} mappings")
        # Save updated session map (with idle sessions removed)
        save_session_map(cleaned_map)

    logger.info("=" * 80)
    logger.info("Cleanup complete!")


if __name__ == "__main__":
    import sys

    # Check for dry-run flag
    if "--dry-run" in sys.argv:
        DRY_RUN = True
        logger.info("Running in DRY RUN mode (no changes will be made)")

    # Check for custom threshold
    if "--hours" in sys.argv:
        idx = sys.argv.index("--hours")
        if idx + 1 < len(sys.argv):
            try:
                IDLE_THRESHOLD_HOURS = float(sys.argv[idx + 1])
            except ValueError:
                logger.error("Invalid hours value")
                sys.exit(1)

    cleanup_idle_sessions()
