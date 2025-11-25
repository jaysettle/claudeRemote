# Claude CLI Bridge for Open WebUI

A bridge service that connects [Open WebUI](https://github.com/open-webui/open-webui) to Claude CLI, enabling voice-controlled interactions with Claude through a web interface.

## Overview

This project creates an OpenAI-compatible API layer that proxies requests from Open WebUI to Claude CLI running on a Linux server. This allows you to use Open WebUI's excellent interface (including voice input) with Claude.

## Architecture

```
Client Device (Voice/Text Input)
    |
    v
Open WebUI (Docker) - Port 3001
    |
    v
Claude Bridge (Python/FastAPI) - Port 8000
    |
    v
Claude CLI (Native)
```

## Components

### Claude CLI Bridge (`claude-cli-bridge/`)
- **claude_bridge.py**: FastAPI server providing OpenAI-compatible endpoints
- **docker-compose.yml**: Docker configuration for Open WebUI
- **Dockerfile**: Container build for the bridge (optional)
- **deploy.sh / deploy-windows.ps1**: Deployment scripts

## Features

- OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`)
- Session management via tmux
- File upload support (images, PDFs, text files)
- Streaming responses
- Works with Open WebUI's voice input

## Requirements

- Linux server with:
  - Python 3.8+
  - Docker & Docker Compose
  - tmux
  - Claude CLI (authenticated)
- Client device with web browser

## Quick Start

1. Clone this repository to your server
2. Install dependencies: `pip install -r claude-cli-bridge/requirements.txt`
3. Ensure Claude CLI is installed and authenticated
4. Start the bridge: `python3 claude-cli-bridge/claude_bridge.py`
5. Start Open WebUI: `docker-compose up -d`
6. Access Open WebUI at `http://your-server:3001`
7. Select "claude-cli" from the model dropdown

## Configuration

Set environment variables:
- `CLAUDE_CLI_PATH`: Path to Claude CLI binary (default: `/usr/local/bin/claude`)
- `CLAUDE_TIMEOUT`: Response timeout in seconds (default: 300)

## Security Note

This project uses `--dangerously-skip-permissions` flag for Claude CLI. Only use in trusted environments.

## Related Projects

- [Open WebUI](https://github.com/open-webui/open-webui) - The web interface used for this integration
- [Claude CLI](https://github.com/anthropics/claude-cli) - Anthropic's CLI tool for Claude

## License

MIT
