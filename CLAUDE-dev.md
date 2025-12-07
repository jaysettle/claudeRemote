# CLAUDE.md - Development Environment

**YOU ARE IN THE DEVELOPMENT ENVIRONMENT** - This is a safe sandbox for experimentation.

This server (jasLinux) runs both production and development Claude CLI Bridge instances. This file documents the dev environment and teaches you how to safely experiment.

---

## Table of Contents

1. [Environment Awareness](#environment-awareness)
2. [What You CAN Do](#what-you-can-do-safely)
3. [What You MUST NOT Do](#what-you-must-not-do)
4. [Architecture Overview](#architecture-overview)
5. [Port Allocation](#port-allocation)
6. [Service Management](#service-management)
7. [What Has Worked](#what-has-worked-lessons-learned)
8. [Tailscale Funnel Guide](#tailscale-funnel-guide)
9. [Creating New Web Services](#creating-new-web-services)
10. [File Locations](#file-locations)
11. [Docker Patterns](#docker-patterns)
12. [Firewall Configuration](#firewall-configuration)
13. [Troubleshooting](#troubleshooting)

---

## Environment Awareness

### You Are Here: DEVELOPMENT
```
Directory: /home/jay/claude-cli-bridge-dev/
Service:   claude-bridge-dev.service
Ports:     9000 (bridge), 9001 (Open WebUI)
Version:   1.8.1-dev
```

### Production (DO NOT MODIFY)
```
Directory: /home/jay/claude-cli-bridge/
Service:   claude-bridge.service
Ports:     8000 (bridge), 3001 (Open WebUI)
Version:   1.8.0
```

### Quick Status Check
```bash
# Dev status
curl -s http://localhost:9000/ | jq .version
# Should return: "1.8.1-dev"

# Prod status (for comparison only)
curl -s http://localhost:8000/ | jq .version
# Should return: "1.8.0"
```

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
                 |                                    |
                 +----------------+-------------------+
                                  v
                            Claude CLI
                      (/usr/local/bin/claude)
                                  |
                                  v
                           Claude API
                                  |
                      +-----------+-----------+
                      v                       v
                hass-mcp                google-drive
           (Home Assistant)           (Google Drive)
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
   - Must stream line-by-line to preserve markdown formatting
   - Chunk by newlines, not arbitrary byte boundaries

4. **--dangerously-skip-permissions**
   - Required for non-interactive execution
   - Claude will not prompt for file access, tool use, etc.

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
├── claude_bridge.py      # Dev bridge code (MODIFY THIS)
├── CLAUDE.md             # This file
└── .claude/
    └── settings.json     # Project-scoped MCP config

/etc/systemd/system/
└── claude-bridge-dev.service

/tmp/claude_sessions_dev/
└── session_map.json      # Dev session mappings

/tmp/claude_uploads_dev/
└── *.png, *.jpg, etc.    # Dev uploaded files
```

### Shared Configuration
```
~/.claude.json            # MCP servers (shared by prod and dev)
~/.npm-global/            # Global npm packages
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

*Last Updated: 2025-11-27*
*Environment: Development (claude-cli-bridge-dev)*
*This file guides Claude CLI when working in this dev environment.*
