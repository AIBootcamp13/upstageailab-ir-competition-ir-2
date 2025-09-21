#!/bin/bash
# Test script to verify the development environment setup

echo "Testing Docker environment setup..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop or the Docker service."
    exit 1
fi

echo "✅ Docker is running"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found"
    exit 1
fi

echo "✅ docker-compose.yml found"

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."

# Check Elasticsearch
if curl -s http://localhost:9200 >/dev/null; then
    echo "✅ Elasticsearch is running"
else
    echo "❌ Elasticsearch is not accessible"
fi

# Check Kibana
if curl -s http://localhost:5601 >/dev/null; then
    echo "✅ Kibana is running"
else
    echo "❌ Kibana is not accessible"
fi

# Check Redis
if redis-cli -p 6379 ping >/dev/null; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not accessible"
fi

# Check Dev Container
if docker-compose ps | grep -q "dev-container.*Up"; then
    echo "✅ Dev container is running"
else
    echo "❌ Dev container is not running"
fi

echo "📋 Service status summary:"
docker-compose ps

echo "🎉 Setup verification complete!"