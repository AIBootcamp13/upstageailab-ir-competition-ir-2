#!/usr/bin/env bash
set -euo pipefail

# Smoke test: start elasticsearch and redis (background), run a couple of checks, then stop them.
# Usage: ./scripts/smoke-test.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Flags
NO_CLEANUP=0
NO_INSTALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cleanup)
      NO_CLEANUP=1; shift ;;
    --no-install)
      NO_INSTALL=1; shift ;;
    *)
      shift ;;
  esac
done

echo "Starting Elasticsearch..."
if [ "$NO_INSTALL" -eq 1 ]; then
  bash "$ROOT_DIR/scripts/start-elasticsearch.sh" --prebuilt --no-install || true
else
  bash "$ROOT_DIR/scripts/start-elasticsearch.sh" --prebuilt || true
fi
echo "Starting Redis..."
if [ "$NO_INSTALL" -eq 1 ]; then
  bash "$ROOT_DIR/scripts/start-redis.sh" --prebuilt --no-install || true
else
  bash "$ROOT_DIR/scripts/start-redis.sh" --prebuilt || true
fi

ES_DIR="$ROOT_DIR/elasticsearch-8.9.0"
REDIS_DIR="$ROOT_DIR/redis-7.2.0"

echo "Wait a few seconds for services to initialize..."
sleep 5

echo "Checking Elasticsearch HTTP API on http://127.0.0.1:9200"
if curl -sS --fail http://127.0.0.1:9200 >/dev/null 2>&1; then
  echo "Elasticsearch responded"
else
  echo "Elasticsearch did not respond"
  exit 1
fi

echo "Checking Redis PING"
if command -v redis-cli >/dev/null 2>&1; then
  redis-cli -p 6379 ping || { echo "Redis ping failed"; exit 1; }
else
  python3 - <<'PY'
import socket
try:
    s=socket.create_connection(('127.0.0.1',6379),2)
    s.sendall(b'*1\r\n$4\r\nPING\r\n')
    resp=s.recv(1024)
    if b'PONG' not in resp:
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
  echo "Redis responded (via python TCP check)"
fi

echo "Smoke test passed. Cleaning up..."

# If NO_CLEANUP is set, skip stopping services
if [ "$NO_CLEANUP" -eq 1 ]; then
  echo "--no-cleanup set; leaving services running"
  exit 0
fi

# Stop services by PID files if present
if [ -f "$ES_DIR/run/elasticsearch.pid" ]; then
  ES_PID="$(cat "$ES_DIR/run/elasticsearch.pid" 2>/dev/null || echo "")"
  if [ -n "$ES_PID" ] && kill -0 "$ES_PID" >/dev/null 2>&1; then
    kill "$ES_PID" || true
  else
    echo "Elasticsearch PID $ES_PID not running, skipping kill"
  fi
fi
if [ -f "$REDIS_DIR/run/redis.pid" ]; then
  REDIS_PID="$(cat "$REDIS_DIR/run/redis.pid" 2>/dev/null || echo "")"
  if [ -n "$REDIS_PID" ] && kill -0 "$REDIS_PID" >/dev/null 2>&1; then
    kill "$REDIS_PID" || true
  else
    echo "Redis PID $REDIS_PID not running, skipping kill"
  fi
fi

echo "Done"
