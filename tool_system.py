"""
Tool system for Claude API integration
Provides agentic capabilities through tool definitions and execution
"""

import glob as glob_module
import logging
import os
import subprocess
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Tool Definitions
# ============================================================================

CLAUDE_API_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Use this to view file contents, code, configs, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed). Optional.",
                    "default": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Optional, defaults to 500.",
                    "default": 500
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "bash",
        "description": "Execute a bash command on the server. Use for running scripts, git commands, system operations, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 60.",
                    "default": 60
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern. Use to discover files in the filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern like '**/*.py' or 'src/**/*.ts'"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in. Defaults to home directory.",
                    "default": "~"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "grep",
        "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to current directory.",
                    "default": "."
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files, e.g., '*.py'",
                    "default": "*"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "list_directory",
        "description": "List files and directories in a path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to home directory.",
                    "default": "~"
                }
            },
            "required": []
        }
    }
]


# ============================================================================
# Tool Execution
# ============================================================================

def execute_tool(tool_name: str, tool_input: Dict[str, Any], trace: Optional[Callable] = None) -> str:
    """Execute a tool and return the result as a string."""
    # Import here to avoid circular dependency
    from utils import _emit_log

    _emit_log(trace, f"tool.{tool_name}", f"input={tool_input}")

    try:
        if tool_name == "read_file":
            return _tool_read_file(tool_input)
        elif tool_name == "write_file":
            return _tool_write_file(tool_input)
        elif tool_name == "bash":
            return _tool_bash(tool_input)
        elif tool_name == "glob":
            return _tool_glob(tool_input)
        elif tool_name == "grep":
            return _tool_grep(tool_input)
        elif tool_name == "list_directory":
            return _tool_list_directory(tool_input)
        else:
            return f"Error: Unknown tool '{tool_name}'"
    except Exception as e:
        from utils import _emit_log
        _emit_log(trace, f"tool.{tool_name}.error", str(e), level=logging.ERROR)
        return f"Error executing {tool_name}: {str(e)}"


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
