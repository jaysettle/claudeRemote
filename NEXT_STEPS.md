# Next Steps - Updating Main File

## Current Status

✅ All modules created and tested - imports work correctly
⏳ Main file (`claude_bridge.py`) still needs to be updated to use new modules

## Step-by-Step Guide

### Step 1: Backup Original File
```bash
cd /home/jay/claude-cli-bridge-dev
cp claude_bridge.py claude_bridge.py.backup
echo "Backup created: claude_bridge.py.backup"
```

### Step 2: Update Imports in Main File

Replace the current imports and configuration sections with:

```python
#!/usr/bin/env python3
"""
Claude/Codex CLI Bridge for Open WebUI
Provides an OpenAI-compatible API that proxies requests to CLI agents
Supports persistent sessions per chat thread

v1.12.2-dev - Refactored modular architecture
"""

import asyncio
import json
import logging
import os
import re
import shlex
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

# Import our modules
import config
from models import (
    ChatCompletionRequest,
    Message,
    Model,
    ModelList,
    DiscussionStage,
    DiscussionState,
    SessionNotFoundError,
    CodexSessionNotFoundError,
    GeminiSessionNotFoundError,
)
from session_management import (
    get_session_id,
    set_session_id,
    clear_session_id,
    get_discussion_state,
    set_discussion_state,
    clear_discussion_state,
    session_map,
)
from tool_system import CLAUDE_API_TOOLS, execute_tool
from utils import (
    get_mcp_config,
    get_chat_id,
    process_message_content,
    find_latest_claude_session_id,
    find_latest_codex_session_id,
    find_latest_gemini_session_id,
    check_service_health,
)
from utils.mcp_loader import MCP_SYSTEM_PROMPT
from utils.helpers import _make_trace_logger, _emit_log, _safe_cmd, _truncate, _iter_text_from_content, _parse_uuid
from discussion_system import (
    parse_discussion_intent,
    build_discussion_history_prompt,
    format_discussion_prompt,
)
from cli_streaming import (
    normalize_codex_response,
    is_followup_prompt,
    extract_followups,
    is_error_response,
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Claude CLI Bridge", version=config.VERSION)

# Optional: Anthropic API for true streaming (claude-api model)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
```

### Step 3: Remove Duplicate Code

After adding the imports above, remove these sections from the main file:
1. All the configuration constants (lines ~40-122) - now in config.py
2. Tool definitions and implementations (lines ~123-453) - now in tool_system.py
3. MCP config loading (lines ~457-530) - now in utils/mcp_loader.py
4. Session management functions (lines ~584-640) - now in session_management.py
5. Pydantic models (lines ~643-780) - now in models.py
6. File processing functions (lines ~688-768) - now in utils/file_utils.py
7. Helper functions (_make_trace_logger, _emit_log, etc. lines ~535-582) - now in utils/helpers.py
8. parse_discussion_intent, format_discussion_prompt (lines ~2296-2608) - now in discussion_system.py
9. normalize_codex_response and related (lines ~876-953) - now in cli_streaming.py

### Step 4: Update References

Search and replace these references throughout the remaining code:

**Config values** - Add `config.` prefix:
- `CLAUDE_CLI_PATH` → `config.CLAUDE_CLI_PATH`
- `CLAUDE_TIMEOUT` → `config.CLAUDE_TIMEOUT`
- `CODEX_CLI_PATH` → `config.CODEX_CLI_PATH`
- `GEMINI_CLI_PATH` → `config.GEMINI_CLI_PATH`
- `ANTHROPIC_API_KEY` → `config.ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL` → `config.ANTHROPIC_MODEL`
- `SUPPORTED_MODELS` → `config.SUPPORTED_MODELS`
- `UPLOAD_DIR` → `config.UPLOAD_DIR`
- etc.

**Functions now in utils**:
- Already prefixed correctly in your code if imported properly

**Models**:
- Already imported at top, no changes needed

### Step 5: Test the Updated File

```bash
# Test that the file has valid Python syntax
cd /home/jay/claude-cli-bridge-dev
python3 -m py_compile claude_bridge.py
echo "✅ Syntax check passed"

# Try importing it
python3 -c "import claude_bridge; print('✅ Module imports successfully')"

# Check if FastAPI app initializes
python3 -c "from claude_bridge import app; print(f'✅ FastAPI app: {app.title} v{app.version}')"
```

### Step 6: Start the Service (Test)

```bash
# Test run (Ctrl+C to stop)
cd /home/jay/claude-cli-bridge-dev
python3 claude_bridge.py
```

Expected output:
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

### Step 7: Test Endpoints

In another terminal:

```bash
# Health check
curl http://192.168.3.142:9000/
# Should return: {"message":"Claude CLI Bridge API","version":"1.12.2-dev",...}

# Models list
curl http://192.168.3.142:9000/v1/models
# Should return JSON with 4 models

# Sessions
curl http://192.168.3.142:9000/sessions
# Should return session map with 23 Claude sessions

# MCP status
curl http://192.168.3.142:9000/mcp
# Should return MCP configuration
```

### Step 8: Test Chat Completion

```bash
curl -X POST http://192.168.3.142:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-cli",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "stream": false
  }'
```

Should return a chat completion response.

### Step 9: Restart Dev Service

If all tests pass:

```bash
# Restart the dev service to use the refactored code
sudo systemctl restart claude-bridge-dev.service

# Check status
sudo systemctl status claude-bridge-dev.service

# Watch logs
sudo journalctl -u claude-bridge-dev.service -f
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
- Make sure you're in `/home/jay/claude-cli-bridge-dev/`
- Check that all module files exist
- Verify Python path includes current directory

### Attribute Errors

If you get `AttributeError: module 'config' has no attribute 'X'`:
- Check that the config value is defined in config.py
- Verify you're using the correct module prefix

### Circular Import Errors

If you get circular import issues:
- Large async functions intentionally left in main file
- Don't try to import main file functions into modules

## Alternative: Automated Script

Create a script to automate the update:

```bash
# Coming soon - automated refactoring script
# Will handle all imports and reference updates automatically
```

## Rollback Plan

If anything breaks:

```bash
# Restore backup
cd /home/jay/claude-cli-bridge-dev
cp claude_bridge.py.backup claude_bridge.py

# Restart service
sudo systemctl restart claude-bridge-dev.service
```

## Success Criteria

✅ File syntax is valid
✅ All imports work
✅ FastAPI app initializes
✅ All endpoints respond correctly
✅ Chat completions work
✅ Session persistence works
✅ No errors in logs

## After Success

1. Update production (`/home/jay/claude-cli-bridge/`)
2. Restart production service
3. Monitor logs for any issues
4. Consider Phase 2 refactoring (extract large streaming functions)

---

**Status**: Ready to proceed with main file update
**Estimated Time**: 30-60 minutes for careful update and testing
**Risk**: Low (backup available, modular structure tested)
