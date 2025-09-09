#!/usr/bin/env bash
set -euo pipefail

# Lightweight helper to download, start, stop, and inspect local non-root Elasticsearch
# and Redis distributions inside the repository.
# Usage: scripts/run-local.sh [start|stop|status|help] [-v]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ES_VERSION=${ES_VERSION:-8.9.0}
REDIS_VERSION=${REDIS_VERSION:-7.2.0}

ES_DIR="$ROOT_DIR/elasticsearch-$ES_VERSION"
REDIS_DIR="$ROOT_DIR/redis-$REDIS_VERSION"

PIDFILE_ES="$ES_DIR/run/elasticsearch.pid"
PIDFILE_REDIS="$REDIS_DIR/run/redis.pid"

verb=0
action="${1:-help}"
if [ "${2:-}" = "-v" ] || [ "${2:-}" = "--verbose" ]; then verb=1; fi

log(){ if [ "$verb" -eq 1 ]; then echo "$@"; fi }

download_es(){
  if [ -d "$ES_DIR" ]; then log "ES dir exists: $ES_DIR"; return 0; fi
  TAR_NAME="elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
  URL="https://artifacts.elastic.co/downloads/elasticsearch/${TAR_NAME}"
  echo "Downloading Elasticsearch $ES_VERSION..."
  curl -fSL "$URL" -o "/tmp/$TAR_NAME"
  tar -xzf "/tmp/$TAR_NAME" -C "$ROOT_DIR"
  rm -f "/tmp/$TAR_NAME"
  echo "Downloaded to $ES_DIR"
}

download_redis(){
  if [ -d "$REDIS_DIR" ]; then log "Redis dir exists: $REDIS_DIR"; return 0; fi
  TAR_NAME="redis-$REDIS_VERSION.tar.gz"
  URL="http://download.redis.io/releases/$TAR_NAME"
  echo "Downloading Redis $REDIS_VERSION..."
  curl -fSL "$URL" -o "/tmp/$TAR_NAME"
  tar -xzf "/tmp/$TAR_NAME" -C "$ROOT_DIR"
  rm -f "/tmp/$TAR_NAME"
  echo "Downloaded to $REDIS_DIR"
}

ensure_dirs(){
  mkdir -p "$ES_DIR/data" "$ES_DIR/logs" "$ES_DIR/run"
  mkdir -p "$REDIS_DIR/data" "$REDIS_DIR/logs" "$REDIS_DIR/run"
}

start_es(){
  if [ -f "$PIDFILE_ES" ] && kill -0 "$(cat "$PIDFILE_ES")" >/dev/null 2>&1; then
    echo "Elasticsearch already running (pid $(cat $PIDFILE_ES))"; return 0
  fi
  if [ ! -d "$ES_DIR" ]; then download_es; fi
  ensure_dirs
  if ! grep -q '^discovery.type' "$ES_DIR/config/elasticsearch.yml" 2>/dev/null; then
    cat >> "$ES_DIR/config/elasticsearch.yml" <<'EOF'
network.host: 127.0.0.1
http.port: 9200
discovery.type: single-node
# Disable security for local dev
xpack.security.enabled: false
EOF
  fi
  echo "Starting Elasticsearch (logs -> $ES_DIR/logs)"
  nohup "$ES_DIR/bin/elasticsearch" > "$ES_DIR/logs/es.stdout.log" 2> "$ES_DIR/logs/es.stderr.log" &
  echo $! > "$PIDFILE_ES"
  sleep 1
}

start_redis(){
  if [ -f "$PIDFILE_REDIS" ] && kill -0 "$(cat "$PIDFILE_REDIS")" >/dev/null 2>&1; then
    echo "Redis already running (pid $(cat $PIDFILE_REDIS))"; return 0
  fi
  if [ ! -d "$REDIS_DIR" ]; then download_redis; fi
  # build if needed
  if [ ! -f "$REDIS_DIR/src/redis-server" ]; then
    echo "Building Redis (requires make & gcc)"
    (cd "$REDIS_DIR" && make -j$(nproc))
  fi
  ensure_dirs
  echo "Starting Redis (logs -> $REDIS_DIR/logs)"
  nohup "$REDIS_DIR/src/redis-server" --dir "$REDIS_DIR/data" > "$REDIS_DIR/logs/redis.stdout.log" 2> "$REDIS_DIR/logs/redis.stderr.log" &
  echo $! > "$PIDFILE_REDIS"
  sleep 1
}

stop_es(){
  if [ -f "$PIDFILE_ES" ]; then
    pid=$(cat "$PIDFILE_ES")
    echo "Stopping Elasticsearch pid $pid"
    kill "$pid" || true
    rm -f "$PIDFILE_ES"
  else
    echo "No ES pidfile; attempting pkill for $ES_DIR"
    pkill -f "$ES_DIR" || true
  fi
}

stop_redis(){
  if [ -f "$PIDFILE_REDIS" ]; then
    pid=$(cat "$PIDFILE_REDIS")
    echo "Stopping Redis pid $pid"
    kill "$pid" || true
    rm -f "$PIDFILE_REDIS"
  else
    echo "No Redis pidfile; attempting pkill for $REDIS_DIR"
    pkill -f "$REDIS_DIR" || true
  fi
}

status(){
  echo "Elasticsearch:"
  if [ -f "$PIDFILE_ES" ] && kill -0 "$(cat "$PIDFILE_ES")" >/dev/null 2>&1; then
    echo "  running pid $(cat $PIDFILE_ES)"
  else
    # Try to detect Elasticsearch by process name or listening port as a fallback
    es_pid=$(pgrep -f "$ES_DIR" | head -n1 || true)
    if [ -n "$es_pid" ]; then
      echo "  running pid $es_pid (no pidfile)"
    else
      if command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | grep -q ':9200'; then
        # attempt to extract pid from ss output
        listener_pid=$(ss -ltnp 2>/dev/null | grep ':9200' | sed -n '1p' | awk -F',' '{print $2}' | sed 's/.*pid=\([0-9]*\).*/\1/' | sed 's/[^0-9]*//g')
        if [ -n "$listener_pid" ]; then
          echo "  running pid $listener_pid (listening on 9200)"
        else
          echo "  running (listening on 9200)"
        fi
      else
        echo "  not running (no pid or process dead)"
      fi
    fi
  fi
  echo "Redis:"
  if [ -f "$PIDFILE_REDIS" ] && kill -0 "$(cat "$PIDFILE_REDIS")" >/dev/null 2>&1; then
    echo "  running pid $(cat $PIDFILE_REDIS)"
  else
    # Try to detect redis by process name or listening port as a fallback
    redis_pid=$(pgrep -x redis-server | head -n1 || true)
    if [ -n "$redis_pid" ]; then
      echo "  running pid $redis_pid (process exists)"
    else
      if command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | grep -q ':6379'; then
        listener_pid=$(ss -ltnp 2>/dev/null | grep ':6379' | sed -n '1p' | awk -F',' '{print $2}' | sed 's/.*pid=\([0-9]*\).*/\1/' | sed 's/[^0-9]*//g')
        if [ -n "$listener_pid" ]; then
          echo "  running pid $listener_pid (listening on 6379)"
        else
          echo "  running (listening on 6379)"
        fi
      else
        echo "  not running (no pid or process dead)"
      fi
    fi
  fi
}

case "$action" in
  start)
    start_es
    start_redis
    echo "Started. Use 'scripts/run-local.sh status' to check."
    ;;
  stop)
    stop_redis
    stop_es
    ;;
  status)
    status
    ;;
  help|--help|-h)
    cat <<'EOF'
Usage: scripts/run-local.sh [start|stop|status|help]

Commands:
  start   Download (if needed) and start local ES and Redis under the repo (no sudo).
  stop    Stop local ES and Redis started by this script.
  status  Show pidfiles and running state.
  help    Show this message.

Notes:
- You may need to run 'sudo sysctl -w vm.max_map_count=262144' once on the host for Elasticsearch to run.
- ES data will be under elasticsearch-<version>/data; Redis data under redis-<version>/data.
EOF
    ;;
  *)
    echo "Unknown action: $action"; exit 2
    ;;
esac
