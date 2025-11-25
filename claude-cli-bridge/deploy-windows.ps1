# Claude CLI Bridge Deployment Script (Windows)
# Deploys to Linux server at YOUR_SERVER_IP

param(
    [string]$ServerUser = "YOUR_USERNAME",
    [string]$ServerIP = "YOUR_SERVER_IP"
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Deploying Claude CLI Bridge to $ServerUser@$ServerIP" -ForegroundColor Green

# Check if SSH is available
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "❌ SSH not found. Please install OpenSSH." -ForegroundColor Red
    exit 1
}

# Check if we can reach the server
Write-Host "📡 Testing connection..." -ForegroundColor Cyan
try {
    ssh -o ConnectTimeout=5 $ServerUser@$ServerIP "echo 'Connection successful'" | Out-Null
    Write-Host "✅ Connection successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot connect to server. Please check:" -ForegroundColor Red
    Write-Host "  1. Server is running" -ForegroundColor Yellow
    Write-Host "  2. SSH is configured" -ForegroundColor Yellow
    Write-Host "  3. Network connection" -ForegroundColor Yellow
    exit 1
}

# Get the current directory
$LocalDir = Get-Location

# Create remote directory
Write-Host "📁 Creating remote directory..." -ForegroundColor Cyan
ssh $ServerUser@$ServerIP "mkdir -p ~/claude-cli-bridge"

# Copy files to server
Write-Host "📤 Uploading files..." -ForegroundColor Cyan
scp -r "$LocalDir\*" "${ServerUser}@${ServerIP}:~/claude-cli-bridge/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to upload files" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Files uploaded successfully" -ForegroundColor Green

# Execute setup on server
Write-Host "⚙️  Setting up on server..." -ForegroundColor Cyan

$SetupScript = @'
cd ~/claude-cli-bridge

# Check if Claude CLI is installed
if ! command -v claude &> /dev/null; then
    echo "📦 Installing Claude CLI..."
    curl -fsSL https://raw.githubusercontent.com/anthropics/claude-cli/main/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check Claude version
echo "🔍 Claude CLI version:"
~/.local/bin/claude --version || echo "⚠️  Claude CLI not properly installed"

# Install Docker if needed
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker $USER
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
sudo docker-compose down 2>/dev/null || true

# Build and start
echo "🏗️  Building containers..."
sudo docker-compose build

echo "🚀 Starting services..."
sudo docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 5

# Check status
echo "📊 Service status:"
sudo docker-compose ps

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Access URLs:"
echo "   Open WebUI:    http://$SERVER_IP:3001"
echo "   Claude Bridge: http://$SERVER_IP:8001"
echo ""
'@

ssh $ServerUser@$ServerIP $SetupScript

Write-Host ""
Write-Host "✅ Deployment completed!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access Open WebUI at: http://${ServerIP}:3001" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Open http://${ServerIP}:3001 in your browser"
Write-Host "   2. Create an account or log in"
Write-Host "   3. Select 'claude-cli' from the model dropdown"
Write-Host "   4. Start chatting with voice input!"
Write-Host ""
Write-Host "🔍 To view logs, run:" -ForegroundColor Yellow
Write-Host "   ssh $ServerUser@$ServerIP 'cd ~/claude-cli-bridge && sudo docker-compose logs -f'"
Write-Host ""
