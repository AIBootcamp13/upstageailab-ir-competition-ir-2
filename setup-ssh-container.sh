#!/bin/bash
# SSH Key Setup and Container Connection Script
# This script helps manage SSH keys and provides easy container connection

set -e

CONTAINER_NAME="information-retrieval-rag-dev-container"
SSH_PORT=2222

echo "🔑 SSH Key Setup for Container"
echo "================================"

# Check if SSH keys exist on host
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "⚠️  No SSH key found on host (~/.ssh/id_rsa)"
    echo "   Generating new SSH key pair..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" -C "container-dev-key"
    echo "✅ SSH key pair generated"
fi

# Ensure SSH directory permissions are correct
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub

echo "📋 SSH Key Status:"
echo "   Host SSH key: $(ls -la ~/.ssh/id_rsa)"
echo "   Public key: $(ls -la ~/.ssh/id_rsa.pub)"

# Check if container is running
if docker ps | grep -q "$CONTAINER_NAME"; then
    echo ""
    echo "🐳 Container Status: RUNNING"
    echo "   Container: $CONTAINER_NAME"
    echo "   SSH Port: $SSH_PORT"

# Check if container is running
if docker ps | grep -q "$CONTAINER_NAME"; then
    echo ""
    echo "🐳 Container Status: RUNNING"
    echo "   Container: $CONTAINER_NAME"
    echo "   SSH Port: $SSH_PORT"

    # Check SSH host keys in container
    echo ""
    echo "🔍 Checking SSH host keys..."
    if docker exec "$CONTAINER_NAME" ls /etc/ssh/ssh_host_*_key >/dev/null 2>&1; then
        echo "✅ SSH host keys exist (persisted)"
    else
        echo "⚠️  SSH host keys missing - generating..."
        docker exec "$CONTAINER_NAME" sudo ssh-keygen -A
        echo "✅ SSH host keys generated"
    fi

    # Copy SSH public key to container (if not already done)
    echo ""
    echo "🔄 Setting up SSH key in container..."
    docker exec "$CONTAINER_NAME" mkdir -p /home/vscode/.ssh
    docker cp ~/.ssh/id_rsa.pub "$CONTAINER_NAME:/tmp/host_key.pub" 2>/dev/null || echo "Host key not found, skipping copy"
    docker exec "$CONTAINER_NAME" sh -c "cat /tmp/host_key.pub >> /home/vscode/.ssh/authorized_keys 2>/dev/null || echo 'No host key to add'"
    docker exec "$CONTAINER_NAME" chown vscode:vscode /home/vscode/.ssh/authorized_keys
    docker exec "$CONTAINER_NAME" chmod 600 /home/vscode/.ssh/authorized_keys

    echo "✅ SSH key configured in container"

    # Test SSH connection
    echo ""
    echo "🧪 Testing SSH connection..."
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT vscode@localhost "echo 'SSH connection successful!'"; then
        echo "✅ SSH connection test passed"
    else
        echo "❌ SSH connection test failed"
        exit 1
    fi

    echo ""
    echo "🚀 Quick Connect Commands:"
    echo "   ssh -p $SSH_PORT vscode@localhost"
    echo "   Or use: ./connect-container.sh"

else
    echo ""
    echo "🐳 Container Status: NOT RUNNING"
    echo "   Start the container first:"
    echo "   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d"
    echo ""
    echo "   Then run this script again to setup SSH keys"
fi

echo ""
echo "💡 Tips:"
echo "   - SSH keys are automatically mounted from host (~/.ssh)"
echo "   - Container SSH port: $SSH_PORT"
echo "   - Username: vscode"
echo "   - To reset container: docker-compose down && docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d"