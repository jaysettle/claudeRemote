# CLAUDE.md - Development Environment

**YOU ARE IN THE DEVELOPMENT ENVIRONMENT** - This is a safe sandbox for experimentation.

This server (jasLinux) runs both production and development Claude CLI Bridge instances. This file documents the dev environment and teaches you how to safely experiment.

---

## Table of Contents

1. [Environment Awareness](#environment-awareness)
2. [Available Models](#available-models)
3. [What You CAN Do](#what-you-can-do-safely)
4. [What You MUST NOT Do](#what-you-must-not-do)
5. [Architecture Overview](#architecture-overview)
6. [Port Allocation](#port-allocation)
7. [Service Management](#service-management)
8. [Systemd Service Configuration](#systemd-service-configuration)
9. [What Has Worked](#what-has-worked-lessons-learned)
10. [Tailscale Funnel Guide](#tailscale-funnel-guide)
11. [Creating New Web Services](#creating-new-web-services)
12. [File Locations](#file-locations)
13. [Docker Patterns](#docker-patterns)
14. [Firewall Configuration](#firewall-configuration)
15. [Troubleshooting](#troubleshooting)

---

## Environment Awareness

### You Are Here: DEVELOPMENT
```
Directory: /home/jay/claude-cli-bridge-dev/
Service:   claude-bridge-dev.service
Ports:     9000 (bridge), 9001 (Open WebUI)
Version:   1.12.2-dev
Models:    claude-cli, claude-api, codex-cli, gemini-cli
```

### Production (DO NOT MODIFY)
```
Directory: /home/jay/claude-cli-bridge/
Service:   claude-bridge.service
Ports:     8000 (bridge), 3001 (Open WebUI)
Version:   1.8.0
Models:    claude-cli only
```

### Quick Status Check
```bash
# Dev status
curl -s http://localhost:9000/ | jq .version
# Should return: "1.12.1-dev"

# Prod status (for comparison only)
curl -s http://localhost:8000/ | jq .version
# Should return: "1.8.0"

# List models
curl -s http://localhost:9000/v1/models | jq '.data[].id'
```

---

## Available Models

The dev environment supports **4 models** (vs production which only has claude-cli):

### Model Comparison

| Model | Streaming | File Access | Session Persistence | MCP Tools | Notes |
|-------|-----------|-------------|---------------------|-----------|-------|
| **claude-cli** | Simulated (line-by-line) | Via CLI Read tool | Yes (`--resume`) | Yes (hass-mcp, google-drive) | Production-ready |
| **claude-api** | True (token-by-token) | Via built-in tools | No | No (uses built-in tools) | Direct Anthropic API |
| **codex-cli** | Simulated (line-by-line) | Via CLI tools | Yes (`--resume`) | Limited | OpenAI Codex CLI |
| **gemini-cli** | Simulated (line-by-line) | Via CLI tools | Yes (`--resume`) | Limited | Google Gemini CLI |

### claude-api Tools

The `claude-api` model has built-in agentic capabilities with these tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line numbers (offset/limit supported) |
| `write_file` | Write/create files (protected paths blocked: /etc, /usr, /bin, /sbin, /boot, /root) |
| `bash` | Execute shell commands (dangerous commands blocked: rm -rf /, mkfs, dd if=, etc.) |
| `glob` | Find files by pattern (e.g., `**/*.py`) |
| `grep` | Search file contents with regex |
| `list_directory` | List files in a directory |

**Tool Safety Features:**
- Protected paths cannot be written to
- Dangerous bash commands are blocked
- 60-second timeout on bash commands
- Output truncated at 50KB

### History Handling by Model

| Model | When Session Exists | When No Session |
|-------|---------------------|-----------------|
| **claude-cli** | Current message only | Full history |
| **claude-api** | Always full history | Always full history |
| **codex-cli** | Current message only | Full history |
| **gemini-cli** | Always full history | Always full history |

### Model Selection in Open WebUI

In Open WebUI dev (port 9001), you can select models from the dropdown:
- `claude-cli` - CLI-based, with MCP tools
- `claude-api` - Direct API, true streaming, built-in tools
- `codex-cli` - OpenAI Codex CLI
- `gemini-cli` - Google Gemini CLI

---

## What You CAN Do (Safely)

### In This Dev Environment:
- Modify `/home/jay/claude-cli-bridge-dev/claude_bridge.py`
- Restart `claude-bridge-dev.service`
- Restart `open-webui-claude-dev` container
- Add new UFW rules for dev ports (9000-9999 range recommended)
- Create new Docker containers on unused ports
- Add new Tailscale Funnel routes on new ports (8443, 10443, etc.)
- Experiment with MCP configurations in `.claude/settings.json`
- Delete dev session files in `/tmp/claude_sessions_dev/`
- Delete dev uploads in `/tmp/claude_uploads_dev/`

### System-Wide (with care):
- Install npm packages globally (`npm install -g`)
- Install Python packages (`pip install`)
- Create new systemd services (use unique names)
- Add UFW rules for new ports

---

## What You MUST NOT Do

### Never Touch Production:
- DO NOT modify `/home/jay/claude-cli-bridge/` (production directory)
- DO NOT restart `claude-bridge.service` (production bridge)
- DO NOT restart `open-webui-claude` container (production UI)
- DO NOT delete `open-webui-claude-data` volume (user data!)
- DO NOT change ports 8000 or 3001

### System Safety:
- DO NOT open ports to public internet without Tailscale
- DO NOT delete `~/.claude.json` (shared MCP config)
- DO NOT delete `~/.config/google-drive-mcp/` (OAuth tokens)
- DO NOT remove UFW rules for ports 22, 3001, 8000
- DO NOT stop Docker daemon
- DO NOT reboot without user consent

---

## Architecture Overview

```
                    PRODUCTION                          DEVELOPMENT
                    ==========                          ===========

Internet --> Tailscale Funnel                    Tailscale Funnel
             (port 443)                          (port 8443)
                 |                                    |
                 v                                    v
         Open WebUI Docker                    Open WebUI Docker
         (port 3001)                          (port 9001)
         open-webui-claude                    open-webui-claude-dev
                 |                                    |
                 v                                    v
         Claude Bridge                        Claude Bridge
         (port 8000)                          (port 9000)
         claude-bridge.service                claude-bridge-dev.service
         ~/claude-cli-bridge/                 ~/claude-cli-bridge-dev/
         v1.8.0                               v1.12.1-dev
                 |                                    |
                 v                                    v
           claude-cli only              claude-cli, claude-api,
                                        codex-cli, gemini-cli
                 |                                    |
                 +----------------+-------------------+
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
              Claude CLI    Codex CLI     Gemini CLI        Anthropic API
         (/usr/local/bin/   (~/.npm-global/  (/usr/local/    (direct for
            claude)         bin/codex)       bin/gemini)     claude-api)
                    |             |             |                  |
                    +-------------+-------------+                  |
                                  |                                |
                                  v                                v
                           Claude API                      Claude API
                                  |                       (with tools)
                      +-----------+-----------+
                      v                       v
                hass-mcp                google-drive
           (Home Assistant)           (Google Drive)
```

### Dev-Only: claude-api Direct Path
```
Open WebUI --> Claude Bridge --> Anthropic API (direct)
                    |                    |
                    |                    v
                    |              Built-in Tools:
                    |              - read_file
                    |              - write_file
                    |              - bash
                    |              - glob
                    |              - grep
                    |              - list_directory
                    |                    |
                    v                    v
              True streaming      Tool execution
              (token-by-token)    (local server)
```

---

## Port Allocation

### Reserved - DO NOT USE
| Port | Service | Environment |
|------|---------|-------------|
| 22 | SSH | System |
| 3001 | Open WebUI | **PRODUCTION** |
| 8000 | Claude Bridge | **PRODUCTION** |
| 8001 | Location Map | Production |

### Development Ports
| Port | Service | Status |
|------|---------|--------|
| 9000 | Claude Bridge Dev | **IN USE** |
| 9001 | Open WebUI Dev | **IN USE** |
| 9002-9099 | Available for dev services | Free |

### Tailscale Funnel Ports
| Port | Service | URL |
|------|---------|-----|
| 443 | Production Open WebUI | https://jaslinux.tail23d264.ts.net |
| 8443 | Dev Open WebUI | https://jaslinux.tail23d264.ts.net:8443 |
| 10443+ | Available for new services | Free |

---

## Service Management

### Dev Services
```bash
# Claude Bridge Dev
sudo systemctl status claude-bridge-dev.service
sudo systemctl restart claude-bridge-dev.service
sudo journalctl -u claude-bridge-dev.service -f

# Open WebUI Dev
docker restart open-webui-claude-dev
docker logs -f open-webui-claude-dev

# Health Checks
curl http://localhost:9000/
curl http://localhost:9001/health
```

### All Services Status
```bash
# Quick overview
echo "=== Production ===" && \
curl -s http://localhost:8000/ | jq -r '"Bridge: \(.version) - \(.status)"' && \
curl -s http://localhost:3001/health | jq -r '"WebUI: \(.status)"' && \
echo "=== Development ===" && \
curl -s http://localhost:9000/ | jq -r '"Bridge: \(.version) - \(.status)"' && \
curl -s http://localhost:9001/health | jq -r '"WebUI: \(.status)"'
```

---

## Systemd Service Configuration

### Dev Service: `/etc/systemd/system/claude-bridge-dev.service`

```ini
[Unit]
Description=Claude CLI Bridge Service (DEV)
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=jay
WorkingDirectory=/home/jay/claude-cli-bridge-dev

# CLI paths
Environment=CLAUDE_CLI_PATH=/usr/local/bin/claude
Environment=CODEX_CLI_PATH=/home/jay/.npm-global/bin/codex
Environment=GEMINI_CLI_PATH=/usr/local/bin/gemini

# Dev isolation
Environment=CLAUDE_PROJECT_PATH=/home/jay/claude-cli-bridge-dev
Environment=CLAUDE_SESSION_DIR=/tmp/claude_sessions_dev
Environment=CLAUDE_UPLOAD_DIR=/tmp/claude_uploads_dev
Environment=CLAUDE_BRIDGE_PORT=9000

# Claude API (for claude-api model with true streaming)
Environment=ANTHROPIC_API_KEY=sk-ant-api03-...
Environment=ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Use venv Python (has anthropic package)
ExecStart=/home/jay/claude-cli-bridge-dev/venv/bin/python /home/jay/claude-cli-bridge-dev/claude_bridge.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_CLI_PATH` | `/usr/local/bin/claude` | Path to Claude CLI binary |
| `CODEX_CLI_PATH` | `~/.npm-global/bin/codex` | Path to Codex CLI binary |
| `GEMINI_CLI_PATH` | `/usr/local/bin/gemini` | Path to Gemini CLI binary |
| `CLAUDE_TIMEOUT` | `600` | Response timeout (seconds) |
| `CLAUDE_PROJECT_PATH` | `$HOME` | MCP config lookup path |
| `CLAUDE_SESSION_DIR` | `/tmp/claude_sessions` | Session storage |
| `CLAUDE_UPLOAD_DIR` | `/tmp/claude_uploads` | File upload storage |
| `CLAUDE_BRIDGE_PORT` | `8000` | API port |
| `ANTHROPIC_API_KEY` | (none) | Required for claude-api model |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Model for claude-api |

### Updating the Service

```bash
# Edit service file
sudo nano /etc/systemd/system/claude-bridge-dev.service

# Reload systemd
sudo systemctl daemon-reload

# Restart service
sudo systemctl restart claude-bridge-dev.service

# Check status
sudo systemctl status claude-bridge-dev.service
```

---

## What Has Worked (Lessons Learned)

### Claude CLI Bridge - Key Discoveries

1. **Session Persistence with --resume**
   - Claude CLI supports `--resume <session_id>` for continuing conversations
   - Session IDs must be UUID format (not agent-* format)
   - Sessions stored in `~/.claude/projects/-home-jay-claude-cli-bridge/`

2. **MCP Tools with -p Flag**
   - Non-interactive mode (`-p`) requires explicit `--mcp-config` JSON
   - Config is JSON-escaped and passed on command line
   - Bridge dynamically loads from `~/.claude.json`

3. **Streaming Responses**
   - CLI models: Must stream line-by-line to preserve markdown formatting
   - claude-api: True token-by-token streaming (no buffering needed)

4. **--dangerously-skip-permissions**
   - Required for non-interactive execution
   - Claude will not prompt for file access, tool use, etc.

### Claude API (v1.11.0+) - Key Discoveries

1. **Tool Use with Anthropic API**
   - Tools defined as JSON schema, passed to `messages.create()`
   - Response includes `tool_use` blocks when model wants to use a tool
   - Agentic loop: call API → check for tool_use → execute → send result → repeat

2. **529 Overloaded Errors**
   - Anthropic API returns 529 when overloaded
   - Implemented retry logic with exponential backoff (2s → 4s → 8s)
   - Max 3 retries before giving up

3. **Safety Features for Tools**
   - Protected paths: Cannot write to /etc, /usr, /bin, /sbin, /boot, /root
   - Dangerous commands blocked: rm -rf /, mkfs, dd if=, > /dev/, fork bomb
   - Bash timeout: 60 seconds default

4. **Venv Python Required**
   - `anthropic` package not in system Python
   - Service must use `/home/jay/claude-cli-bridge-dev/venv/bin/python`
   - Install: `pip install anthropic` in venv

### Open WebUI Integration

1. **Headers are Critical**
   - `ENABLE_FORWARD_USER_INFO_HEADERS=true` required
   - `X-OpenWebUI-Chat-Id` header maps conversations to Claude sessions

2. **TTS Requires Separate Config**
   - Chat API goes to Claude Bridge (OPENAI_API_BASE_URLS)
   - TTS goes to real OpenAI (AUDIO_TTS_OPENAI_API_BASE_URL)
   - Different env vars, different endpoints

3. **Subpath Hosting Does NOT Work**
   - Tried `/dev` path for dev environment - fails after login
   - SvelteKit frontend expects root path
   - **Solution**: Use separate ports (8443, 10443) instead

### Tailscale Funnel

1. **Port-Based Routing Works**
   ```bash
   # Each port gets its own URL
   tailscale funnel --bg 3001           # https://host.ts.net/
   tailscale funnel --bg --https=8443 9001  # https://host.ts.net:8443/
   ```

2. **Path-Based Routing Does NOT Work for SPAs**
   - Can configure `/dev` path, but SPA redirects break
   - Always use port-based separation for web apps

3. **Background Mode is Persistent**
   - `--bg` flag keeps funnel running after disconnect
   - Survives reboots (systemd-managed)

### Docker Patterns

1. **Volume Naming**
   - Production: `open-webui-claude-data`
   - Development: `open-webui-claude-dev-data`
   - Never share volumes between prod/dev

2. **Environment Isolation**
   - Each container has its own env vars
   - Dev points to port 9000, prod to 8000

---

## Tailscale Funnel Guide

### Creating a New Public URL

For a new web service on port 9002:

```bash
# 1. Add UFW rules
sudo ufw allow from 192.168.3.0/24 to any port 9002  # LAN access
sudo ufw allow from 100.64.0.0/10 to any port 9002   # Tailscale access

# 2. Start your service on port 9002

# 3. Create Tailscale Funnel on a unique HTTPS port
sudo tailscale funnel --bg --https=9443 9002

# 4. Verify
sudo tailscale funnel status
```

**Result**: Service accessible at `https://jaslinux.tail23d264.ts.net:9443/`

### Available Funnel Ports
- 443: Reserved for production
- 8443: Reserved for dev Open WebUI
- 9443, 10443, 11443...: Available for new services

### Funnel Commands
```bash
# View current configuration
sudo tailscale funnel status

# Add new funnel
sudo tailscale funnel --bg --https=<external-port> <internal-port>

# Remove funnel
sudo tailscale funnel --https=<external-port> off

# Add path-based route (for APIs, not SPAs)
sudo tailscale funnel --bg --set-path=/api 8080
```

### IMPORTANT: Port vs Path
- **Web Apps (SPA)**: Use port-based (`https://host:8443/`)
- **APIs**: Can use path-based (`https://host/api/`)
- **Reason**: SPAs redirect to `/` after login, breaking subpath hosting

---

## Creating New Web Services

### Template: New Docker Web Service

```bash
# 1. Choose a port (check port allocation above)
NEW_PORT=9002
FUNNEL_PORT=9443
SERVICE_NAME=my-new-service

# 2. Add firewall rules
sudo ufw allow from 192.168.3.0/24 to any port $NEW_PORT
sudo ufw allow from 100.64.0.0/10 to any port $NEW_PORT

# 3. Run Docker container
docker run -d --name $SERVICE_NAME \
  -p $NEW_PORT:8080 \
  --restart unless-stopped \
  your-image:tag

# 4. Add Tailscale Funnel
sudo tailscale funnel --bg --https=$FUNNEL_PORT $NEW_PORT

# 5. Verify
curl http://localhost:$NEW_PORT/
curl https://jaslinux.tail23d264.ts.net:$FUNNEL_PORT/
```

### Template: New Python Web Service

```bash
# 1. Create directory
mkdir ~/my-new-service
cd ~/my-new-service

# 2. Create service file
sudo tee /etc/systemd/system/my-new-service.service << EOF
[Unit]
Description=My New Service
After=network.target

[Service]
Type=simple
User=jay
WorkingDirectory=/home/jay/my-new-service
ExecStart=/usr/bin/python3 /home/jay/my-new-service/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 3. Enable and start
sudo systemctl daemon-reload
sudo systemctl enable my-new-service
sudo systemctl start my-new-service

# 4. Add firewall and funnel (see above)
```

---

## File Locations

### Development Environment
```
/home/jay/claude-cli-bridge-dev/
├── claude_bridge.py      # Dev bridge code v1.12.1-dev (MODIFY THIS)
├── CLAUDE.md             # This file
├── venv/                 # Python venv with anthropic package
│   └── bin/python        # Used by systemd service
└── .claude/
    └── settings.json     # Project-scoped MCP config

/etc/systemd/system/
└── claude-bridge-dev.service

/tmp/claude_sessions_dev/
└── session_map.json      # Dev session mappings (per-model)

/tmp/claude_uploads_dev/
└── *.png, *.jpg, etc.    # Dev uploaded files
```

### Shared Configuration
```
~/.claude.json            # MCP servers (shared by prod and dev)
~/.npm-global/            # Global npm packages (codex CLI here)
/usr/local/bin/claude     # Claude CLI binary
/usr/local/bin/gemini     # Gemini CLI binary
```

### Venv Setup (for claude-api)
```bash
cd /home/jay/claude-cli-bridge-dev
python3 -m venv venv
source venv/bin/activate
pip install anthropic fastapi uvicorn pydantic
```

---

## Docker Patterns

### Development Container Pattern
```bash
docker run -d --name <name>-dev \
  -p <dev-port>:8080 \
  -e CONFIG_VAR=dev-value \
  -v <name>-dev-data:/app/data \
  --restart unless-stopped \
  image:tag
```

### Current Dev Container
```bash
# Recreate if needed
docker stop open-webui-claude-dev && docker rm open-webui-claude-dev
docker run -d --name open-webui-claude-dev \
  -p 9001:8080 \
  -e OPENAI_API_BASE_URLS=http://192.168.3.142:9000/v1 \
  -e OPENAI_API_KEYS=dummy-key \
  -e ENABLE_FORWARD_USER_INFO_HEADERS=true \
  -e AUDIO_TTS_ENGINE=openai \
  -e AUDIO_TTS_OPENAI_API_BASE_URL=https://api.openai.com/v1 \
  -e AUDIO_TTS_OPENAI_API_KEY=<key> \
  -e AUDIO_TTS_MODEL=tts-1 \
  -e AUDIO_TTS_VOICE=alloy \
  -v open-webui-claude-dev-data:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

---

## Firewall Configuration

### Current Rules
```bash
sudo ufw status numbered

# Production
# 3001: Open WebUI (LAN + Tailscale)
# 8000: Claude Bridge (LAN + Docker)

# Development
# 9000: Dev Bridge (LAN + Docker)
# 9001: Dev Open WebUI (LAN + Tailscale)
```

### Adding Rules for New Service
```bash
# LAN access
sudo ufw allow from 192.168.3.0/24 to any port <port>

# Docker access (if needed)
sudo ufw allow from 172.16.0.0/12 to any port <port>

# Tailscale access
sudo ufw allow from 100.64.0.0/10 to any port <port>
```

---

## Troubleshooting

### Dev Bridge Not Responding
```bash
sudo systemctl status claude-bridge-dev.service
sudo journalctl -u claude-bridge-dev.service -n 50
sudo systemctl restart claude-bridge-dev.service
```

### Dev Open WebUI Issues
```bash
docker logs open-webui-claude-dev --tail 50
docker restart open-webui-claude-dev
```

### Tailscale Funnel Not Working
```bash
sudo tailscale funnel status
# Check if funnel is enabled in Tailscale admin console
# https://login.tailscale.com/admin/machines
```

### Port Already in Use
```bash
sudo ss -tlnp | grep <port>
# Kill the process or choose a different port
```

---

*Last Updated: 2025-12-01*
*Environment: Development (claude-cli-bridge-dev)*
*Version: 1.12.2-dev*
*Models: claude-cli, claude-api, codex-cli, gemini-cli*
*This file guides Claude CLI when working in this dev environment.*
