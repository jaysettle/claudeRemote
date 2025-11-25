#!/bin/bash
# Setup script for Claude Bridge auto-start on boot

echo "Setting up Claude Bridge auto-start..."

# Copy service file
sudo cp ~/claude-cli-bridge/claude-bridge.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable docker
sudo systemctl enable claude-bridge.service

# Start the bridge service now
sudo systemctl start claude-bridge.service

# Set up Tailscale HTTPS (serves port 3001 over HTTPS)
sudo tailscale serve --bg https / http://localhost:3001

# Show status
echo ""
echo "=== Status ==="
sudo systemctl status claude-bridge.service --no-pager
echo ""
docker ps | grep webui
echo ""
echo "=== Tailscale HTTPS URL ==="
tailscale serve status

echo ""
echo "Setup complete! Access via your Tailscale HTTPS URL"
