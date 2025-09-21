#!/bin/bash
# SSH Setup Status and Troubleshooting Script

CONTAINER_NAME="information-retrieval-rag-dev-container"
SSH_PORT=2222

echo "🔍 SSH Setup Status Check"
echo "=========================="
echo ""

# Check if container is running
if docker ps | grep -q "$CONTAINER_NAME"; then
    echo "✅ Container Status: RUNNING"
    echo "   Name: $CONTAINER_NAME"
    echo "   SSH Port: $SSH_PORT"
    echo ""

    # Check SSH host keys
    echo "🔑 SSH Host Keys Status:"
    docker exec "$CONTAINER_NAME" sh -c "
        if [ -f /etc/ssh/ssh_host_rsa_key ]; then
            echo '   ✅ RSA key: EXISTS'
        else
            echo '   ❌ RSA key: MISSING'
        fi
        if [ -f /etc/ssh/ssh_host_ed25519_key ]; then
            echo '   ✅ Ed25519 key: EXISTS'
        else
            echo '   ❌ Ed25519 key: MISSING'
        fi
    "
    echo ""

    # Check SSH daemon
    echo "🔧 SSH Service Status:"
    if docker exec "$CONTAINER_NAME" pgrep -f sshd >/dev/null; then
        echo "   ✅ SSH daemon: RUNNING"
    else
        echo "   ❌ SSH daemon: NOT RUNNING"
    fi
    echo ""

    # Check user SSH directory
    echo "👤 User SSH Setup:"
    docker exec "$CONTAINER_NAME" sh -c "
        if [ -d /home/vscode/.ssh ]; then
            echo '   ✅ SSH directory: EXISTS'
            ls -la /home/vscode/.ssh/
        else
            echo '   ❌ SSH directory: MISSING'
        fi
    "
    echo ""

    # Check volume mounts
    echo "💾 Volume Mounts:"
    docker inspect "$CONTAINER_NAME" | grep -A 5 "Mounts" | head -10
    echo ""

    # Test SSH connection
    echo "🧪 SSH Connection Test:"
    if timeout 5 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p $SSH_PORT vscode@localhost "echo 'Connection successful'" >/dev/null 2>&1; then
        echo "   ✅ SSH connection: WORKING"
    else
        echo "   ❌ SSH connection: FAILED"
        echo "   💡 Try: ./setup-ssh-container.sh"
    fi

else
    echo "❌ Container Status: NOT RUNNING"
    echo "   Start with: docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d"
fi

echo ""
echo "📋 Troubleshooting Tips:"
echo "   1. Check container logs: docker-compose logs dev-container"
echo "   2. Re-setup SSH: ./setup-ssh-container.sh"
echo "   3. Check SSH config: docker exec $CONTAINER_NAME cat /etc/ssh/sshd_config"
echo "   4. Test manually: ssh -p $SSH_PORT vscode@localhost"
echo ""
echo "🔄 After container rebuild:"
echo "   - SSH host keys are persisted via named volume"
echo "   - No host key warnings on reconnection"
echo "   - SSH keys are mounted from host (~/.ssh)"