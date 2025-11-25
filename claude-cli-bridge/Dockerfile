FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tmux \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Claude CLI
RUN curl -fsSL https://raw.githubusercontent.com/anthropics/claude-cli/main/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY claude_bridge.py .

# Expose port
EXPOSE 8000

# Set environment variables
ENV CLAUDE_CLI_PATH=/root/.local/bin/claude
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "claude_bridge.py"]
