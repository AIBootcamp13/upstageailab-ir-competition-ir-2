#!/bin/bash
# Setup script to add Elastic Stack binaries to PATH

# Get the absolute path of the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Add Elasticsearch binaries to PATH
if [ -d "$PROJECT_ROOT/elasticsearch-8.9.0/bin" ]; then
    export PATH="$PROJECT_ROOT/elasticsearch-8.9.0/bin:$PATH"
    echo "Added Elasticsearch binaries to PATH"
fi

# Add Kibana binaries to PATH (if Kibana is downloaded)
if [ -d "$PROJECT_ROOT/.local_kibana/kibana-8.15.0/bin" ]; then
    export PATH="$PROJECT_ROOT/.local_kibana/kibana-8.15.0/bin:$PATH"
    echo "Added Kibana binaries to PATH"
fi

# Redis is already installed system-wide, so it should be in PATH
if command -v redis-server >/dev/null 2>&1; then
    echo "Redis is already available in PATH"
else
    echo "Warning: Redis not found in PATH"
fi

echo "Elastic Stack binaries setup complete."
echo "You can now run: elasticsearch, kibana, redis-server"