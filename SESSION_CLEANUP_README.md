# Claude Session Cleanup System

Automatically removes idle Claude/Codex/Gemini CLI sessions after 2 hours of inactivity.

## Why This Matters

Your server health check showed **20+ active Claude API connections**. These are from persistent sessions that remain mapped even when idle. This cleanup system:

- ✅ Closes idle sessions after 2 hours
- ✅ Reduces active API connections
- ✅ Frees up session resources
- ✅ Safe to use with `--resume` (sessions can be restored)

## Files

- **`cleanup_idle_sessions.py`** - Main cleanup script
- **`install_cleanup.sh`** - Interactive installation script
- **`/etc/systemd/system/claude-session-cleanup.service`** - Systemd service (option 1)
- **`/etc/systemd/system/claude-session-cleanup.timer`** - Systemd timer (option 1)

## Quick Start

### Option 1: Automatic Installation

```bash
cd /home/jay/claude-cli-bridge-dev
./install_cleanup.sh
```

Choose between:
1. **Systemd timer** (recommended) - Runs hourly, survives reboots
2. **Cron job** - Simpler, user-level only

### Option 2: Manual Testing

```bash
# Dry run (see what would be cleaned)
python3 cleanup_idle_sessions.py --dry-run

# Actually clean up idle sessions
python3 cleanup_idle_sessions.py

# Custom idle threshold (e.g., 4 hours)
python3 cleanup_idle_sessions.py --hours 4
```

## How It Works

1. **Reads session map** from `/tmp/claude_sessions_dev/session_map.json`
2. **Checks last activity** by looking at session file modification times:
   - Claude CLI: `~/.claude/projects/*/session-id.jsonl`
   - Gemini CLI: `~/.gemini/tmp/*/chats/session-*.json`
   - Codex CLI: `~/.codex/sessions/session-id.jsonl`
3. **Removes mappings** for sessions idle > 2 hours
4. **Preserves active sessions** (used within last 2 hours)

## Session Restoration

When a cleaned session is resumed:

```bash
# Claude CLI automatically creates a new session
claude --resume <old-session-id> -p "Continue working"
```

The bridge will:
1. Detect the session ID is not in the map
2. Fall back to sending full conversation history
3. Create a new session with the resumed context
4. Update the map with the new session ID

**No conversation history is lost!**

## Configuration

Edit `cleanup_idle_sessions.py` to customize:

```python
# Idle threshold (default: 2 hours)
IDLE_THRESHOLD_HOURS = 2

# Session directory (default: /tmp/claude_sessions)
SESSION_DIR = Path(os.getenv("CLAUDE_SESSION_DIR", Path("/tmp/claude_sessions")))

# Dry run mode (default: False)
DRY_RUN = False
```

## Monitoring

### Systemd Timer (Option 1)

```bash
# Check timer status
sudo systemctl status claude-session-cleanup.timer

# View next scheduled run
sudo systemctl list-timers claude-session-cleanup.timer

# View cleanup logs
sudo journalctl -u claude-session-cleanup.service -f

# Run manually
sudo systemctl start claude-session-cleanup.service
```

### Cron Job (Option 2)

```bash
# View crontab
crontab -l

# View logs
tail -f /tmp/claude_cleanup.log

# Run manually
python3 /home/jay/claude-cli-bridge-dev/cleanup_idle_sessions.py
```

## Example Output

```
================================================================================
Claude CLI Session Cleanup - Starting
Threshold: 2 hours
Dry run: False
================================================================================
Sessions idle since before 2025-12-07 09:17:02 will be cleaned

Processing claude-cli sessions (3 chats)
  🧹 claude-cli/dace53ab: Session 0856b851 idle for 16.2h (last: 2025-12-06 19:07:56)
     ✂️  Removed mapping (can be restored with --resume)
  🧹 claude-cli/60b863e1: Session a33529f7 idle for 13.8h (last: 2025-12-06 21:26:36)
     ✂️  Removed mapping (can be restored with --resume)
  ✅ claude-cli/f4a9c2e8: Session b1c8d3f9 active (idle 0.5h, last: 2025-12-07 10:47:23)

Processing codex-cli sessions (0 chats)
Processing gemini-cli sessions (0 chats)

================================================================================
Cleanup Summary:
  Total sessions checked: 3
  Active sessions (kept): 1
  Idle sessions (>2h): 2
  Missing session files: 0
  Cleaned up: 2 mappings
================================================================================
```

## Production Deployment

To use in production (port 8000), update the service file:

```bash
# Edit /etc/systemd/system/claude-session-cleanup.service
Environment=CLAUDE_SESSION_DIR=/tmp/claude_sessions  # Production directory

# Or create separate service for production
sudo cp /etc/systemd/system/claude-session-cleanup.service \
        /etc/systemd/system/claude-session-cleanup-prod.service

# Edit the prod service to use /tmp/claude_sessions
```

## Troubleshooting

### Sessions not being cleaned

Check that session files exist:
```bash
ls -lah ~/.claude/projects/-home-jay-claude-cli-bridge-dev/
```

### Want to clean more aggressively

Run with a shorter threshold:
```bash
python3 cleanup_idle_sessions.py --hours 1  # 1 hour instead of 2
```

### Systemd timer not running

```bash
# Check if enabled
sudo systemctl is-enabled claude-session-cleanup.timer

# Enable if not
sudo systemctl enable claude-session-cleanup.timer
sudo systemctl start claude-session-cleanup.timer
```

## Safety

- ✅ **Safe to run** - Only removes session mappings, not actual session files
- ✅ **Reversible** - Sessions can be resumed with `--resume`
- ✅ **Non-destructive** - Full conversation history preserved in bridge
- ✅ **Tested** - Dry-run mode available for testing

## Impact

**Before cleanup:**
- 20+ active Claude API connections
- Session map grows indefinitely
- Memory usage increases over time

**After cleanup (running hourly):**
- Only active sessions remain connected
- Session map stays small
- Minimal memory footprint
- Old sessions auto-cleaned every hour

---

**Author:** Jay Settle
**Version:** 1.0.0
**Date:** 2025-12-07
