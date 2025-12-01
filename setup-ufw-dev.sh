#!/bin/bash
# UFW rules for dev environment

# Allow dev ports on LAN
ufw allow from 192.168.3.0/24 to any port 9000
ufw allow from 192.168.3.0/24 to any port 9001

# Allow Docker to reach dev bridge
ufw allow from 172.16.0.0/12 to any port 9000

# Allow Tailscale to reach dev Open WebUI
ufw allow from 100.64.0.0/10 to any port 9001

echo "UFW rules added for dev environment"
ufw status | grep 900
