#!/usr/bin/env bash
set -euo pipefail

# Simple helper to download and run Redis locally for development
# Usage: ./scripts/start-redis.sh [--foreground|--systemd] [version]
# Defaults to version 7.2.0

FOREGROUND=0
VER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground|--systemd)
      FOREGROUND=1
      shift
      ;;
    *)
      if [ -z "$VER" ]; then
        VER="$1"
      fi
      shift
      ;;
  esac
done

REDIS_VERSION="${VER:-7.2.0}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/redis-$REDIS_VERSION"

if [ -d "$DIST_DIR" ]; then
  echo "Using existing $DIST_DIR"
else
  TAR_NAME="redis-$REDIS_VERSION.tar.gz"
  URL="http://download.redis.io/releases/$TAR_NAME"
  echo "Downloading Redis $REDIS_VERSION from $URL"
  curl -fSL "$URL" -o "/tmp/$TAR_NAME"
  tar -xzf "/tmp/$TAR_NAME" -C "$ROOT_DIR"
  rm "/tmp/$TAR_NAME"
  cd "$DIST_DIR"
  echo "Building Redis (requires make and gcc)"
  make -j$(nproc)
fi

cd "$DIST_DIR"
mkdir -p "$DIST_DIR/data" "$DIST_DIR/logs" "$DIST_DIR/run"

if [ "$FOREGROUND" -eq 1 ]; then
  echo "Starting redis-server in foreground"
  exec ./src/redis-server --dir "$DIST_DIR/data"
else
  echo "Starting redis-server in background (logs -> $DIST_DIR/logs)"
  nohup ./src/redis-server --dir "$DIST_DIR/data" > "$DIST_DIR/logs/redis.stdout.log" 2> "$DIST_DIR/logs/redis.stderr.log" &
  REDIS_PID=$!
  echo "$REDIS_PID" > "$DIST_DIR/run/redis.pid"
  echo "Redis started with PID $REDIS_PID"

  # Simple verification: wait up to 20s for PING
  echo "Waiting for Redis to accept connections on port 6379"
  for i in $(seq 1 20); do
    if command -v redis-cli >/dev/null 2>&1; then
      if redis-cli -p 6379 ping >/dev/null 2>&1; then
        echo "Redis is responding (after ${i}s)"
        break
      fi
    fi
    sleep 1
  done
  if command -v redis-cli >/dev/null 2>&1; then
    if ! redis-cli -p 6379 ping >/dev/null 2>&1; then
      echo "Warning: Redis did not respond within 20s; check logs in $DIST_DIR/logs"
      exit 1
    fi
  else
    echo "Note: redis-cli not found; can't verify with PING. Install redis-tools if you want verification."
  fi
fi
