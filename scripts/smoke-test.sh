#!/usr/bin/env bash
set -euo pipefail

# Smoke test: start elasticsearch and redis (background), run a couple of checks, then stop them.
# Usage: ./scripts/smoke-test.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Starting Elasticsearch..."
bash "$ROOT_DIR/scripts/start-elasticsearch.sh" || true
echo "Starting Redis..."
bash "$ROOT_DIR/scripts/start-redis.sh --prebuilt" || true

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
  python3 - <<PY
import socket
try:
    s=socket.create_connection(('127.0.0.1',6379),2)
    s.sendall(b'*1\r\n$4\r\nPING\r\n')
    resp=s.recv(1024)
    if b'PONG' not in resp:
        raise SystemExit(1)
except Exception as e:
    raise SystemExit(1)
PY
  echo "Redis responded (via python TCP check)"
fi

echo "Smoke test passed. Cleaning up..."

# Stop services by PID files if present
if [ -f "$ES_DIR/run/elasticsearch.pid" ]; then
  kill "$(cat $ES_DIR/run/elasticsearch.pid)" || true
fi
if [ -f "$REDIS_DIR/run/redis.pid" ]; then
  kill "$(cat $REDIS_DIR/run/redis.pid)" || true
fi

echo "Done"
