#!/usr/bin/env bash
set -euo pipefail

# Simple helper to download and run Redis locally for development
# Usage: ./scripts/start-redis.sh [version]
# Defaults to version 7.2.0

REDIS_VERSION="${1:-7.2.0}"
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
echo "Starting redis-server with data dir $DIST_DIR/data"
mkdir -p "$DIST_DIR/data" "$DIST_DIR/logs"
nohup ./src/redis-server --dir "$DIST_DIR/data" > "$DIST_DIR/logs/redis.stdout.log" 2> "$DIST_DIR/logs/redis.stderr.log" &
REDIS_PID=$!
echo "Redis started with PID $REDIS_PID"
echo "Connect with: redis-cli -p 6379"
