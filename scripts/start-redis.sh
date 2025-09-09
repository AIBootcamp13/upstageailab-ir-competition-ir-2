#!/usr/bin/env bash
set -euo pipefail

# Simple helper to download/run Redis locally for development
# Usage: ./scripts/start-redis.sh [--foreground|--systemd] [--prebuilt] [version]
# Defaults to version 7.2.0

FOREGROUND=0
PREBUILT=0
NO_INSTALL=0
VER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground|--systemd)
      FOREGROUND=1; shift ;;
    --prebuilt)
      PREBUILT=1; shift ;;
    --no-install)
      NO_INSTALL=1; shift ;;
    *)
      if [ -z "$VER" ]; then VER="$1"; fi
      shift
      ;;
  esac
done

REDIS_VERSION="${VER:-7.2.0}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/redis-$REDIS_VERSION"

# If a Redis instance is already responding on localhost:6379, skip start/download.
if python3 - <<'PY' > /dev/null 2>&1
import socket
try:
  s=socket.create_connection(('127.0.0.1',6379),1)
  s.sendall(b'*1\r\n$4\r\nPING\r\n')
  resp=s.recv(1024)
  if b'PONG' in resp:
    print('PONG')
    raise SystemExit(0)
except SystemExit:
  raise
except:
  raise SystemExit(1)
PY
then
  echo "Redis already responding on 127.0.0.1:6379 — skipping start/download"
  exit 0
fi

# If prebuilt requested, prefer system redis-server binary or package manager
if [ "$PREBUILT" -eq 1 ]; then
  if command -v redis-server >/dev/null 2>&1; then
    echo "Using system provided redis-server binary"
    REDIS_BIN="$(command -v redis-server)"
  else
    if [ "$NO_INSTALL" -eq 1 ]; then
      echo "--no-install set; not attempting to install redis packages and falling back to build-from-source if available"
    else
      echo "redis-server not found on PATH; attempting to install via package manager (requires sudo)"
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y redis-server || true
      elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y redis || true
      else
        echo "No supported package manager found; falling back to building from source"
      fi
    fi
    if command -v redis-server >/dev/null 2>&1; then
      REDIS_BIN="$(command -v redis-server)"
    fi
  fi
fi

# If no prebuilt or no system binary available, build from source if needed
if [ -z "${REDIS_BIN:-}" ]; then
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
  REDIS_BIN="$DIST_DIR/src/redis-server"
fi

cd "$ROOT_DIR"
mkdir -p "$DIST_DIR/data" "$DIST_DIR/logs" "$DIST_DIR/run"

if [ "$FOREGROUND" -eq 1 ]; then
  echo "Starting redis-server in foreground using $REDIS_BIN"
  exec "$REDIS_BIN" --dir "$DIST_DIR/data"
else
  echo "Starting redis-server in background using $REDIS_BIN (logs -> $DIST_DIR/logs)"
  nohup "$REDIS_BIN" --dir "$DIST_DIR/data" > "$DIST_DIR/logs/redis.stdout.log" 2> "$DIST_DIR/logs/redis.stderr.log" &
  REDIS_PID=$!
  echo "$REDIS_PID" > "$DIST_DIR/run/redis.pid"
  echo "Redis started with PID $REDIS_PID"

  # Simple verification: wait up to 20s for PING using redis-cli or small python check
  echo "Waiting for Redis to accept connections on port 6379"
  for i in $(seq 1 20); do
    if command -v redis-cli >/dev/null 2>&1; then
      if redis-cli -p 6379 ping >/dev/null 2>&1; then
        echo "Redis is responding (after ${i}s)"
        break
      fi
    else
      # try a minimal TCP PING using python
  if python3 - <<'PY' > /dev/null 2>&1
import socket
try:
    s=socket.create_connection(('127.0.0.1',6379),1)
    s.sendall(b'*1\r\n$4\r\nPING\r\n')
    resp=s.recv(1024)
    if b'PONG' in resp:
        print('PONG')
except:
    raise SystemExit(1)
PY
      then
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
    # final python check
  if ! python3 - <<'PY' > /dev/null 2>&1
import socket
try:
    s=socket.create_connection(('127.0.0.1',6379),1)
    s.sendall(b'*1\r\n$4\r\nPING\r\n')
    resp=s.recv(1024)
    if b'PONG' not in resp:
        raise SystemExit(1)
except:
    raise SystemExit(1)
PY
    then
      echo "Warning: Redis did not respond within 20s; check logs in $DIST_DIR/logs"
      exit 1
    fi
  fi
fi
