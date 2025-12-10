#!/usr/bin/env python3
"""
Tool Executor
Dispatches tool calls to appropriate implementations
"""

import logging
from typing import Any, Callable, Dict, Optional

from .file_tools import _tool_read_file, _tool_write_file
from .shell_tools import _tool_bash, _tool_glob, _tool_grep, _tool_list_directory

logger = logging.getLogger(__name__)


def _emit_log(log_fn: Optional[Callable[[str, str, int], None]], stage: str, message: str, level: int = logging.INFO):
    """Emit a log line using the trace logger if provided."""
    if log_fn:
        log_fn(stage, message, level)
    else:
        logger.log(level, f"{stage} | {message}")


def execute_tool(tool_name: str, tool_input: Dict[str, Any], trace: Optional[Callable] = None) -> str:
    """Execute a tool and return the result as a string."""
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
        _emit_log(trace, f"tool.{tool_name}.error", str(e), level=logging.ERROR)
        return f"Error executing {tool_name}: {str(e)}"
