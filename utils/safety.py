#!/usr/bin/env python3
"""
Safety and Health Check Utilities
Path validation, git rollback, and service health checks
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

from config import (
    ALLOWED_WRITE_ROOTS, BLOCKED_PATH_PREFIXES,
    HEALTHCHECK_CMD, HEALTHCHECK_URL, HEALTHCHECK_TIMEOUT
)

logger = logging.getLogger(__name__)


def is_safe_path(path: Path) -> bool:
    """Validate target path against whitelist/blacklist rules."""
    try:
        resolved = path.expanduser().resolve()
    except Exception as exc:
        logger.warning(f"is_safe_path: failed to resolve {path}: {exc}")
        return False

    for blocked in BLOCKED_PATH_PREFIXES:
        if resolved == blocked or blocked in resolved.parents:
            return False

    for root in ALLOWED_WRITE_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue

    return False

def find_git_root(start: Path) -> Optional[Path]:
    """Find nearest git root for a path."""
    try:
        candidate = start.expanduser().resolve()
    except Exception:
        return None

    for parent in [candidate] + list(candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None

def create_rollback_commit(repo_root: Path) -> Optional[str]:
    """Create an empty commit as rollback checkpoint if repo is clean."""
    if not repo_root or not shutil.which("git"):
        return None

    def _run_git(args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
        )

    inside = _run_git(["rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0:
        logger.info(f"Rollback commit skipped: not a git repo at {repo_root}")
        return None

    status = _run_git(["status", "--porcelain"])
    if status.returncode != 0:
        logger.warning(f"Rollback commit status failed: {status.stderr.strip()}")
        return None

    if status.stdout.strip():
        logger.info("Rollback commit skipped: working tree dirty")
        return None

    commit_message = f"Implementation rollback checkpoint {int(time.time())}"
    commit = _run_git(["commit", "--allow-empty", "-m", commit_message])
    if commit.returncode != 0:
        logger.warning(f"Rollback commit failed: {commit.stderr.strip()}")
        return None

    head = _run_git(["rev-parse", "HEAD"])
    if head.returncode == 0:
        return head.stdout.strip()

    logger.warning("Rollback commit created but unable to read HEAD")
    return None

def reset_to_commit(repo_root: Path, commit_hash: str) -> bool:
    if not repo_root or not commit_hash or not shutil.which("git"):
        return False

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "reset", "--hard", commit_hash],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"Git rollback failed: {result.stderr.strip()}")
            return False
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(f"Git rollback exception: {exc}")
        return False

def backup_file(src: Path, backup_dir: Path) -> Optional[Path]:
    """Copy src into backup_dir preserving absolute structure."""
    try:
        resolved = src.expanduser().resolve()
        rel = resolved.relative_to(resolved.anchor)
        dest = backup_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if resolved.exists():
            shutil.copy2(resolved, dest)
        return dest
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to back up {src}: {exc}")
        return None

def write_file_with_backup(target: Path, content: str, backup_dir: Path) -> Tuple[Path, bool]:
    """Write file atomically with backup; returns (path, created)."""
    resolved = target.expanduser().resolve()
    created = not resolved.exists()
    if not created:
        backup_file(resolved, backup_dir)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="impl_", dir=str(resolved.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())

    os.replace(tmp_path, resolved)
    return resolved, created

def restore_backups(backup_dir: Path) -> List[Path]:
    restored: List[Path] = []
    if not backup_dir.exists():
        return restored

    for file in backup_dir.rglob("*"):
        if file.is_file():
            rel = file.relative_to(backup_dir)
            target = Path(os.sep) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            restored.append(target)
    return restored

def delete_created_files(paths: List[Path]) -> List[Path]:
    removed: List[Path] = []
    for p in paths:
        try:
            if p.exists():
                p.unlink()
                removed.append(p)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to delete {p}: {exc}")
    return removed

def check_service_health() -> Tuple[bool, str]:
    """Run configured health check command or HTTP probe."""
    if HEALTHCHECK_CMD:
        try:
            result = subprocess.run(
                HEALTHCHECK_CMD,
                shell=True,
                capture_output=True,
                text=True,
                timeout=HEALTHCHECK_TIMEOUT,
            )
            output = (result.stdout or "").strip() or (result.stderr or "").strip()
            return result.returncode == 0, output or f"Exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return False, "Health check command timed out"
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"Health check command failed: {exc}"

    if HEALTHCHECK_URL:
        try:
            with urllib.request.urlopen(HEALTHCHECK_URL, timeout=HEALTHCHECK_TIMEOUT) as resp:
                return 200 <= resp.status < 300, f"HTTP {resp.status}"
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"Health check HTTP failed: {exc}"

    return True, "No health check configured"
