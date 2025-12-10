#!/usr/bin/env python3
"""
File Operations Tools
Implements read_file and write_file tools
"""

import os
from typing import Dict, Any


def _tool_read_file(input: Dict[str, Any]) -> str:
    """Read file contents."""
    path = os.path.expanduser(input.get("path", ""))
    offset = input.get("offset", 1)
    limit = input.get("limit", 500)

    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    if os.path.isdir(path):
        return f"Error: Path is a directory, not a file: {path}"

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Apply offset and limit
        start = max(0, offset - 1)
        end = start + limit
        selected_lines = lines[start:end]

        # Format with line numbers
        result_lines = []
        for i, line in enumerate(selected_lines, start=start+1):
            result_lines.append(f"{i:6d}\t{line.rstrip()}")

        result = "\n".join(result_lines)
        if len(lines) > end:
            result += f"\n... ({len(lines) - end} more lines)"

        return result
    except Exception as e:
        return f"Error reading file: {str(e)}"

def _tool_write_file(input: Dict[str, Any]) -> str:
    """Write content to file."""
    path = os.path.expanduser(input.get("path", ""))
    content = input.get("content", "")

    # Safety check - don't allow writing to certain paths
    dangerous_paths = ["/etc/", "/usr/", "/bin/", "/sbin/", "/boot/", "/root/"]
    for dp in dangerous_paths:
        if path.startswith(dp):
            return f"Error: Cannot write to protected path: {path}"

    try:
        # Create directory if needed
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"
