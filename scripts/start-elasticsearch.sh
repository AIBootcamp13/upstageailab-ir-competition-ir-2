#!/usr/bin/env bash
set -euo pipefail

# Simple helper to download and run Elasticsearch Linux tar locally for development
# Usage: ./scripts/start-elasticsearch.sh [--foreground|--systemd] [version]
# Defaults to version 8.9.0

FOREGROUND=0
PREBUILT=0
ES_VERSION_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground|--systemd)
      FOREGROUND=1
      shift
      ;;
    --prebuilt)
      PREBUILT=1
      shift
      ;;
    *)
      if [ -z "$ES_VERSION_ARG" ]; then
        ES_VERSION_ARG="$1"
      fi
      shift
      ;;
  esac
done

ES_VERSION="${ES_VERSION_ARG:-8.9.0}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/elasticsearch-$ES_VERSION"

# If prebuilt requested, prefer system package or distro-managed service
if [ "$PREBUILT" -eq 1 ]; then
  if command -v elasticsearch >/dev/null 2>&1; then
    echo "Using system 'elasticsearch' binary"
    ES_BIN="$(command -v elasticsearch)"
  else
    echo "Trying to install Elasticsearch via package manager (requires sudo)"
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y elasticsearch || true
    elif command -v yum >/dev/null 2>&1; then
      sudo yum install -y elasticsearch || true
    fi
    if command -v elasticsearch >/dev/null 2>&1; then
      ES_BIN="$(command -v elasticsearch)"
    fi
  fi
fi

# If no system binary available, fall back to local tarball distribution
if [ -z "${ES_BIN:-}" ]; then
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
  ES_BIN="$DIST_DIR/bin/elasticsearch"
else
  # system binary may expect to be run as a service; assume systemd managed
  echo "System Elasticsearch binary detected at $ES_BIN"
fi

cd "$DIST_DIR"

# Create a minimal dev config override if needed
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

mkdir -p "$DIST_DIR/logs" "$DIST_DIR/run"

mkdir -p "$DIST_DIR/logs" "$DIST_DIR/run"

if [ "$FOREGROUND" -eq 1 ]; then
  echo "Starting Elasticsearch in foreground (suitable for systemd)"
  exec "$ES_BIN"
else
  echo "Starting Elasticsearch in background (logs -> $DIST_DIR/logs)"
  nohup bash -c "exec $ES_BIN" > "$DIST_DIR/logs/es.stdout.log" 2> "$DIST_DIR/logs/es.stderr.log" &
  ES_PID=$!
  echo "$ES_PID" > "$DIST_DIR/run/elasticsearch.pid"
  echo "Elasticsearch started with PID $ES_PID"

  # Verify service is up (wait up to 60s)
  echo "Waiting for Elasticsearch to respond on http://127.0.0.1:9200"
  for i in $(seq 1 60); do
    if curl -sS --fail http://127.0.0.1:9200 >/dev/null 2>&1; then
      echo "Elasticsearch is up (after ${i}s)"
      break
    fi
    sleep 1
  done
  if ! curl -sS --fail http://127.0.0.1:9200 >/dev/null 2>&1; then
    echo "Warning: Elasticsearch did not respond within 60 seconds; check $DIST_DIR/logs for details"
    exit 1
  fi
fi
