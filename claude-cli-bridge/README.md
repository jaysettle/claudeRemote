# Claude CLI Bridge for Open WebUI

This project integrates Claude CLI with Open WebUI, allowing you to use voice dictation and a web interface to interact with Claude.

## Architecture

```
┌─────────────────┐
│   Open WebUI    │ (Web Interface + Voice Input)
│   Port 3000     │
└────────┬────────┘
         │ OpenAI-compatible API
         │
┌────────▼────────┐
│  Claude Bridge  │ (FastAPI Service)
│   Port 8000     │
└────────┬────────┘
         │ tmux + CLI
         │
┌────────▼────────┐
│   Claude CLI    │ (--dangerously-skip-permissions)
│                 │
└─────────────────┘
```

## Features

- **OpenAI-Compatible API**: Works seamlessly with Open WebUI
- **Tmux Session Management**: Maintains persistent Claude CLI sessions
- **Automatic Continuation**: Handles first-time vs continued conversations
- **Streaming Support**: Real-time response streaming (optional)
- **Voice Input**: Use Apple Voice Dictate through Open WebUI

## Quick Start

### Option 1: Docker Compose (Recommended)

1. Build and start both services:
```bash
docker-compose up -d
```

2. Access Open WebUI at `http://localhost:3001`

3. Select "claude-cli" as your model

**Note**: Using port 3001 to avoid conflicts with existing Open WebUI installations.

### Option 2: Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Claude CLI is installed:
```bash
curl -fsSL https://raw.githubusercontent.com/anthropics/claude-cli/main/install.sh | sh
```

3. Run the bridge:
```bash
python claude_bridge.py
```

4. Run Open WebUI separately and configure it to use:
   - API Base URL: `http://localhost:8001/v1`
   - API Key: `dummy-key` (any value works)

## Deployment to Linux Server (YOUR_SERVER_IP)

### Prerequisites

SSH into your Linux server:
```bash
ssh YOUR_USERNAME@YOUR_SERVER_IP
# Password: YOUR_PASSWORD
```

### Steps

1. **Copy files to server**:
```bash
# From your Windows machine
scp -r claude-cli-bridge YOUR_USERNAME@YOUR_SERVER_IP:~/
```

2. **SSH into server and setup**:
```bash
ssh YOUR_USERNAME@YOUR_SERVER_IP

cd ~/claude-cli-bridge

# Ensure Claude CLI is authenticated
claude --version
# If not authenticated, run: claude auth login

# Install Docker if not already installed
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Build and run
sudo docker-compose up -d

# Check logs
sudo docker-compose logs -f
```

3. **Access from your network**:
   - Open WebUI: `http://YOUR_SERVER_IP:3001`
   - Claude Bridge API: `http://YOUR_SERVER_IP:8001`

   **Note**: Using ports 3001 and 8001 to avoid conflicts with existing services.

### With Tailscale

If using Tailscale, you can access from anywhere:
```bash
# Install Tailscale on the server (if not already)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --auth-key=tskey-auth-knNhMovwsV11CNTRL-H4b4icj1dpUd3vS8V3SQpUWtz13YNrNWe

# Find your Tailscale IP
tailscale ip -4
```

Then access via: `http://[tailscale-ip]:3001`

## Configuration

### Environment Variables

**Claude Bridge** (`claude_bridge.py`):
- `CLAUDE_CLI_PATH`: Path to Claude CLI binary (default: `claude`)
- `CLAUDE_TIMEOUT`: Timeout for Claude responses in seconds (default: `300`)

**Open WebUI**:
- `OPENAI_API_BASE_URLS`: Set to `http://claude-bridge:8000/v1`
- `OPENAI_API_KEYS`: Any dummy value (e.g., `dummy-key`)

## Usage

1. Open Open WebUI in your browser
2. Select "claude-cli" from the model dropdown
3. Type or use voice dictation to send requests
4. Claude CLI will process your request and return responses

### Voice Input with Apple Devices

1. Enable Voice Dictate on your Apple device
2. Navigate to Open WebUI in Safari
3. Click in the input field
4. Use the microphone button or keyboard shortcut to dictate
5. Send your message to Claude

## How It Works

1. **First Request**:
   - Creates a new tmux session
   - Launches `claude --dangerously-skip-permissions`
   - Sends your input

2. **Subsequent Requests**:
   - Reuses existing tmux session
   - Runs `claude --continue --dangerously-skip-permissions`
   - Maintains conversation context

3. **Response Capture**:
   - Monitors tmux pane output
   - Detects when Claude finishes responding
   - Cleans and formats the response
   - Returns in OpenAI-compatible format

## Troubleshooting

### Bridge not responding
```bash
# Check if service is running
sudo docker-compose ps

# Check logs
sudo docker-compose logs claude-bridge
```

### Claude CLI not authenticated
```bash
# SSH into the bridge container
sudo docker exec -it claude-cli-bridge bash

# Authenticate
claude auth login
```

### Tmux sessions stuck
```bash
# List sessions
tmux ls

# Kill specific session
tmux kill-session -t claude_cli_default

# Kill all Claude sessions
tmux ls | grep claude_cli | cut -d: -f1 | xargs -I {} tmux kill-session -t {}
```

### Open WebUI can't connect
1. Check if both containers are on the same network:
```bash
sudo docker network inspect claude-cli-bridge_openwebui-network
```

2. Test API directly:
```bash
curl http://localhost:8001/v1/models
```

## Security Notes

⚠️ **IMPORTANT**: This uses `--dangerously-skip-permissions` flag which bypasses Claude's safety checks. Use only in trusted environments.

## License

MIT License - See LICENSE file for details

## Contributing

Pull requests welcome! Please ensure:
- Code is well-documented
- Tests pass
- Follows existing code style

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Claude CLI documentation
3. Open an issue on GitHub
