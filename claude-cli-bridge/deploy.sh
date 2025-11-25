#!/bin/bash

# Claude CLI Bridge Deployment Script
# Deploys to Linux server at YOUR_SERVER_IP

set -e

SERVER_USER="YOUR_USERNAME"
SERVER_IP="YOUR_SERVER_IP"
REMOTE_DIR="~/claude-cli-bridge"

echo "🚀 Deploying Claude CLI Bridge to $SERVER_USER@$SERVER_IP"

# Check if we can reach the server
echo "📡 Testing connection..."
if ! ssh -o ConnectTimeout=5 $SERVER_USER@$SERVER_IP "echo 'Connection successful'"; then
    echo "❌ Cannot connect to server. Please check:"
    echo "  1. Server is running"
    echo "  2. SSH is configured"
    echo "  3. Network connection"
    exit 1
fi

# Create remote directory
echo "📁 Creating remote directory..."
ssh $SERVER_USER@$SERVER_IP "mkdir -p $REMOTE_DIR"

# Copy files to server
echo "📤 Uploading files..."
scp -r ./* $SERVER_USER@$SERVER_IP:$REMOTE_DIR/

# Execute setup on server
echo "⚙️  Setting up on server..."
ssh $SERVER_USER@$SERVER_IP << 'ENDSSH'
    cd ~/claude-cli-bridge

    # Check if Claude CLI is installed
    if ! command -v claude &> /dev/null; then
        echo "📦 Installing Claude CLI..."
        curl -fsSL https://raw.githubusercontent.com/anthropics/claude-cli/main/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Check if Claude is authenticated
    echo "🔐 Checking Claude CLI authentication..."
    if ! claude --version &> /dev/null; then
        echo "⚠️  Claude CLI needs authentication"
        echo "   Please run: claude auth login"
    fi

    # Install Docker and Docker Compose if needed
    if ! command -v docker &> /dev/null; then
        echo "🐳 Installing Docker..."
        sudo apt-get update
        sudo apt-get install -y docker.io docker-compose
        sudo systemctl enable docker
        sudo systemctl start docker
        sudo usermod -aG docker $USER
        echo "⚠️  Please log out and back in for Docker permissions to take effect"
    fi

    # Stop existing containers if running
    echo "🛑 Stopping existing containers..."
    sudo docker-compose down 2>/dev/null || true

    # Build and start containers
    echo "🏗️  Building containers..."
    sudo docker-compose build

    echo "🚀 Starting services..."
    sudo docker-compose up -d

    # Wait for services to start
    echo "⏳ Waiting for services to start..."
    sleep 5

    # Check status
    echo "📊 Service status:"
    sudo docker-compose ps

    # Get the server IP
    SERVER_IP=$(hostname -I | awk '{print $1}')

    echo ""
    echo "✅ Deployment complete!"
    echo ""
    echo "🌐 Access URLs:"
    echo "   Open WebUI:    http://$SERVER_IP:3001"
    echo "   Claude Bridge: http://$SERVER_IP:8001"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Open http://$SERVER_IP:3001 in your browser"
    echo "   2. Create an account or log in"
    echo "   3. Select 'claude-cli' from the model dropdown"
    echo "   4. Start chatting!"
    echo ""
    echo "🔍 View logs:"
    echo "   sudo docker-compose logs -f"
    echo ""

ENDSSH

echo "✅ Deployment script completed!"
