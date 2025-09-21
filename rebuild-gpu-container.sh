#!/bin/bash

echo "🔧 Rebuilding container with GPU support..."

# Stop any running containers
docker-compose down

# Rebuild the container
docker-compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache

echo "✅ Container rebuilt with GPU support!"
echo ""
echo "🚀 To start the container with GPU access:"
echo "   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d"
echo ""
echo "🔍 To test GPU access inside the container:"
echo "   docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec dev-container bash"
echo "   Then run: nvidia-smi"
echo "   And: python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""