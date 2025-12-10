#!/usr/bin/env python3
"""
Shell and Filesystem Tools
Implements bash, glob, grep, and list_directory tools
"""

import os
import subprocess
import glob as glob_module
from typing import Dict, Any


def _tool_bash(input: Dict[str, Any]) -> str:
    """Execute bash command."""
    command = input.get("command", "")
    timeout = input.get("timeout", 60)

    # Safety check - block certain dangerous commands
    dangerous_patterns = ["rm -rf /", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
    for dp in dangerous_patterns:
        if dp in command:
            return f"Error: Potentially dangerous command blocked: {command}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.expanduser("~")
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n--- stderr ---\n"
            output += result.stderr

        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"

        # Truncate if too long
        if len(output) > 50000:
            output = output[:50000] + "\n... (output truncated)"

        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"

def _tool_glob(input: Dict[str, Any]) -> str:
    """Find files matching glob pattern."""
    pattern = input.get("pattern", "")
    base_path = os.path.expanduser(input.get("path", "~"))

    try:
        full_pattern = os.path.join(base_path, pattern)
        matches = glob_module.glob(full_pattern, recursive=True)

        # Sort by modification time (newest first)
        matches.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)

        # Limit results
        if len(matches) > 100:
            matches = matches[:100]
            return "\n".join(matches) + f"\n... ({len(matches)} results shown, more available)"

        return "\n".join(matches) if matches else "No files found matching pattern"
    except Exception as e:
        return f"Error in glob: {str(e)}"

def _tool_grep(input: Dict[str, Any]) -> str:
    """Search for pattern in files."""
    pattern = input.get("pattern", "")
    path = os.path.expanduser(input.get("path", "."))
    file_pattern = input.get("file_pattern", "*")

    try:
        # Use grep command for efficiency
        cmd = f"grep -rn --include='{file_pattern}' '{pattern}' '{path}' 2>/dev/null | head -100"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern '{pattern}'"

        return output
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error in grep: {str(e)}"

def _tool_list_directory(input: Dict[str, Any]) -> str:
    """List directory contents."""
    path = os.path.expanduser(input.get("path", "~"))

    if not os.path.exists(path):
        return f"Error: Path not found: {path}"

    if not os.path.isdir(path):
        return f"Error: Not a directory: {path}"

    try:
        entries = os.listdir(path)
        result_lines = []

        for entry in sorted(entries):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                result_lines.append(f"[DIR]  {entry}/")
            else:
                try:
                    size = os.path.getsize(full_path)
                    result_lines.append(f"[FILE] {entry} ({size} bytes)")
                except:
                    result_lines.append(f"[FILE] {entry}")

        return "\n".join(result_lines) if result_lines else "(empty directory)"
    except Exception as e:
        return f"Error listing directory: {str(e)}"




# Cache for MCP config (reloaded periodically)
_mcp_config_cache = None
_mcp_config_cache_time = 0
MCP_CONFIG_CACHE_TTL = 60  # Reload config every 60 seconds
