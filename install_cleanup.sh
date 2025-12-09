#!/bin/bash
# Installation script for Claude session cleanup automation

echo "Claude Session Cleanup - Installation Script"
echo "=============================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Don't run this as root/sudo. It will prompt for password when needed."
    exit 1
fi

echo "Choose installation method:"
echo "  1) Systemd timer (recommended - runs hourly, survives reboots)"
echo "  2) Cron job (simpler, but user-level only)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "Installing systemd timer..."
    echo ""

    # Copy service and timer files
    sudo cp /tmp/claude-session-cleanup.service /etc/systemd/system/
    sudo cp /tmp/claude-session-cleanup.timer /etc/systemd/system/

    # Reload systemd
    sudo systemctl daemon-reload

    # Enable and start the timer
    sudo systemctl enable claude-session-cleanup.timer
    sudo systemctl start claude-session-cleanup.timer

    echo ""
    echo "✅ Systemd timer installed and started!"
    echo ""
    echo "Check status with:"
    echo "  sudo systemctl status claude-session-cleanup.timer"
    echo "  sudo systemctl list-timers claude-session-cleanup.timer"
    echo ""
    echo "View logs with:"
    echo "  sudo journalctl -u claude-session-cleanup.service -f"
    echo ""
    echo "Test manually with:"
    echo "  sudo systemctl start claude-session-cleanup.service"
    echo ""

elif [ "$choice" = "2" ]; then
    echo ""
    echo "Installing cron job..."
    echo ""

    # Add cron job (runs every hour at :05 past the hour)
    (crontab -l 2>/dev/null; echo "# Claude session cleanup - runs hourly"; echo "5 * * * * /usr/bin/python3 /home/jay/claude-cli-bridge-dev/cleanup_idle_sessions.py >> /tmp/claude_cleanup.log 2>&1") | crontab -

    echo ""
    echo "✅ Cron job installed!"
    echo ""
    echo "Cron will run cleanup every hour at :05 past the hour"
    echo ""
    echo "Check crontab with:"
    echo "  crontab -l"
    echo ""
    echo "View logs with:"
    echo "  tail -f /tmp/claude_cleanup.log"
    echo ""
    echo "Test manually with:"
    echo "  python3 /home/jay/claude-cli-bridge-dev/cleanup_idle_sessions.py"
    echo ""

else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "Done!"
