#!/usr/bin/env bash
set -euo pipefail

# Simple helper to download and run Elasticsearch Linux tar locally for development
# Usage: ./scripts/start-elasticsearch.sh [version]
# Defaults to version 8.9.0

ES_VERSION="${1:-8.9.0}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/elasticsearch-$ES_VERSION"

if [ -d "$DIST_DIR" ]; then
  echo "Using existing $DIST_DIR"
else
  TAR_NAME="elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
  URL="https://artifacts.elastic.co/downloads/elasticsearch/${TAR_NAME}"
  echo "Downloading Elasticsearch $ES_VERSION from $URL"
  curl -fSL "$URL" -o "/tmp/$TAR_NAME"
  tar -xzf "/tmp/$TAR_NAME" -C "$ROOT_DIR"
  rm "/tmp/$TAR_NAME"
fi

cd "$DIST_DIR"

# Create a minimal dev config override
CONF_FILE="$DIST_DIR/config/elasticsearch.yml"
if ! grep -q "discovery.type" "$CONF_FILE" 2>/dev/null; then
  cat >> "$CONF_FILE" <<EOF
# Local development overrides
network.host: 127.0.0.1
http.port: 9200
discovery.type: single-node
# Disable security for local dev (DO NOT use in production)
xpack.security.enabled: false
path.data: ${DIST_DIR}/data
path.logs: ${DIST_DIR}/logs
EOF
  echo "Appended minimal dev settings to $CONF_FILE"
else
  echo "elasticsearch.yml already contains discovery.type or network.host; leaving it intact"
fi

echo "Starting Elasticsearch (logs -> $DIST_DIR/logs)"
mkdir -p "$DIST_DIR/logs"
nohup bash -c "exec ./bin/elasticsearch" > "$DIST_DIR/logs/es.stdout.log" 2> "$DIST_DIR/logs/es.stderr.log" &
ES_PID=$!
echo "Elasticsearch started with PID $ES_PID"
echo "Give it some seconds and then check: curl -sS http://127.0.0.1:9200"
