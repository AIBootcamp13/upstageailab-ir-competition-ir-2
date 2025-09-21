#!/bin/bash
# Quick Container SSH Connection Script

SSH_PORT=2222
CONTAINER_USER="vscode"
CONTAINER_HOST="localhost"

echo "🔗 Connecting to container..."
echo "   Host: $CONTAINER_HOST:$SSH_PORT"
echo "   User: $CONTAINER_USER"
echo "   SSH host keys are persisted - no host key warnings!"
echo ""

# SSH connection with options to avoid host key checking (for development)
ssh -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -p $SSH_PORT \
    $CONTAINER_USER@$CONTAINER_HOST