#!/bin/bash
# Alternative SSH Setup using Named Volumes
# This creates a persistent volume for SSH keys

VOLUME_NAME="dev-container-ssh-keys"
CONTAINER_NAME="information-retrieval-rag-dev-container"

echo "🔑 Setting up SSH keys with named volume..."
echo "==========================================="

# Create named volume if it doesn't exist
if ! docker volume ls | grep -q "$VOLUME_NAME"; then
    echo "📦 Creating named volume: $VOLUME_NAME"
    docker volume create "$VOLUME_NAME"
else
    echo "📦 Named volume already exists: $VOLUME_NAME"
fi

# Copy current SSH keys to the volume (one-time setup)
echo ""
echo "🔄 Copying SSH keys to volume..."
docker run --rm \
    -v ~/.ssh:/host-ssh:ro \
    -v $VOLUME_NAME:/container-ssh \
    alpine:latest \
    sh -c "cp -r /host-ssh/* /container-ssh/ 2>/dev/null || true"

echo "✅ SSH keys copied to named volume"
echo ""
echo "📝 To use named volume instead of host mounting:"
echo "   1. Comment out the host mount in docker-compose.dev.yml:"
echo "      # - ~/.ssh:/home/vscode/.ssh:ro"
echo "   2. Uncomment the named volume:"
echo "      - ssh-keys:/home/vscode/.ssh"
echo "   3. Restart container: docker-compose down && docker-compose up -d"
echo ""
echo "💡 Benefits of named volume:"
echo "   - Independent of host SSH keys"
echo "   - Persistent across container resets"
echo "   - Can be backed up/managed separately"