#!/usr/bin/env bash
set -euo pipefail

# Lightweight helper to download, start, stop, and inspect local
# Elasticsearch, Kibana, and Redis distributions inside the repository.
# Usage: scripts/execution/run-local.sh [start|stop|status|help] [-v]

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ES_VERSION=${ES_VERSION:-8.9.0}
REDIS_VERSION=${REDIS_VERSION:-7.2.0}

ES_DIR="$ROOT_DIR/elasticsearch-$ES_VERSION"
REDIS_DIR="$ROOT_DIR/redis-$REDIS_VERSION"
KIBANA_VERSION=${KIBANA_VERSION:-8.9.0}
KIBANA_DIR="$ROOT_DIR/kibana-$KIBANA_VERSION"
PIDFILE_KIBANA="$KIBANA_DIR/run/kibana.pid"

PIDFILE_ES="$ES_DIR/run/elasticsearch.pid"
PIDFILE_REDIS="$REDIS_DIR/run/redis.pid"

# Add binaries to PATH for convenience
export PATH="$ES_DIR/bin:$KIBANA_DIR/bin:$PATH"

verb=0
action="${1:-help}"
if [ "${2:-}" = "-v" ] || [ "${2:-}" = "--verbose" ]; then verb=1; fi

log(){ if [ "$verb" -eq 1 ]; then echo "$@"; fi }

download_es(){
  EXISTING_ES_DIR=$(ls -d "$ROOT_DIR"/elasticsearch-* 2>/dev/null | head -1 || true)
  if [ -n "$EXISTING_ES_DIR" ]; then
    ES_DIR="$EXISTING_ES_DIR"
    log "Using existing ES dir: $ES_DIR"
    return 0
  fi
  if [ -d "$ES_DIR" ]; then log "ES dir exists: $ES_DIR"; return 0; fi
  TAR_NAME="elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
  URL="https://artifacts.elastic.co/downloads/elasticsearch/${TAR_NAME}"
  echo "Downloading Elasticsearch $ES_VERSION..."
  if [ -f "/tmp/$TAR_NAME" ]; then
    echo "Resuming partial download..."
    curl -fSL -C - "$URL" -o "/tmp/$TAR_NAME"
  else
    curl -fSL "$URL" -o "/tmp/$TAR_NAME"
  fi
  tar -xzf "/tmp/$TAR_NAME" -C "$ROOT_DIR"
  rm -f "/tmp/$TAR_NAME"
  echo "Downloaded to $ES_DIR"
}

download_redis(){
  EXISTING_REDIS_DIR=$(ls -d "$ROOT_DIR"/redis-* 2>/dev/null | head -1 || true)
  if [ -n "$EXISTING_REDIS_DIR" ]; then
    REDIS_DIR="$EXISTING_REDIS_DIR"
    log "Using existing Redis dir: $REDIS_DIR"
    return 0
  fi
  if [ -d "$REDIS_DIR" ]; then log "Redis dir exists: $REDIS_DIR"; return 0; fi
  TAR_NAME="redis-$REDIS_VERSION.tar.gz"
  URL="http://download.redis.io/releases/$TAR_NAME"
  echo "Downloading Redis $REDIS_VERSION..."
  if [ -f "/tmp/$TAR_NAME" ]; then
    echo "Resuming partial download..."
    curl -fSL -C - "$URL" -o "/tmp/$TAR_NAME"
  else
    curl -fSL "$URL" -o "/tmp/$TAR_NAME"
  fi
  tar -xzf "/tmp/$TAR_NAME" -C "$ROOT_DIR"
  rm -f "/tmp/$TAR_NAME"
  echo "Downloaded to $REDIS_DIR"
}

ensure_dirs(){
  mkdir -p "$ES_DIR/data" "$ES_DIR/logs" "$ES_DIR/run"
  mkdir -p "$REDIS_DIR/data" "$REDIS_DIR/logs" "$REDIS_DIR/run"
}

start_es(){
  # Check if Elasticsearch is already running (Docker container or accessible)
  if command -v docker >/dev/null 2>&1 && docker ps --filter "name=elasticsearch" --filter "status=running" | grep -q elasticsearch; then
    echo "Elasticsearch already running (Docker container)"
    return 0
  elif curl -s --max-time 5 http://elasticsearch:9200/_cluster/health >/dev/null 2>&1; then
    echo "Elasticsearch already running (accessible on elasticsearch:9200)"
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or not available"
    echo "Please install Docker or start Elasticsearch manually"
    return 1
  fi

  echo "Starting Elasticsearch via Docker Compose..."
  docker-compose up -d elasticsearch
}

start_redis(){
  # Check if Redis is already running (Docker container or accessible)
  if command -v docker >/dev/null 2>&1 && docker ps --filter "name=redis" --filter "status=running" | grep -q redis; then
    echo "Redis already running (Docker container)"
    return 0
  elif redis-cli -h redis -p 6379 ping 2>/dev/null | grep -q PONG; then
    echo "Redis already running (accessible on redis:6379)"
    return 0
  elif uv run python -c "import redis; redis.Redis(host='redis', port=6379).ping()" 2>/dev/null; then
    echo "Redis already running (accessible on redis:6379 via Python)"
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or not available"
    echo "Please install Docker or start Redis manually"
    return 1
  fi

  echo "Starting Redis via Docker Compose..."
  docker-compose up -d redis
}

start_kibana(){
  # Check if Kibana is already running (Docker container or accessible)
  if command -v docker >/dev/null 2>&1 && docker ps --filter "name=kibana" --filter "status=running" | grep -q kibana; then
    echo "Kibana already running (Docker container)"
    return 0
  elif curl -s --max-time 5 http://kibana:5601/api/status >/dev/null 2>&1; then
    echo "Kibana already running (accessible on kibana:5601)"
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or not available"
    echo "Please install Docker or start Kibana manually"
    return 1
  fi

  echo "Starting Kibana via Docker Compose..."
  docker-compose up -d kibana
}

stop_es(){
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or not available"
    return 1
  fi
  if docker ps --filter "name=elasticsearch" | grep -q elasticsearch; then
    echo "Stopping Elasticsearch Docker container..."
    docker-compose stop elasticsearch
  else
    echo "No Elasticsearch Docker container found"
  fi
}

stop_redis(){
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or not available"
    return 1
  fi
  if docker ps --filter "name=redis" | grep -q redis; then
    echo "Stopping Redis Docker container..."
    docker-compose stop redis
  else
    echo "No Redis Docker container found"
  fi
}

stop_kibana(){
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or not available"
    return 1
  fi
  if docker ps --filter "name=kibana" | grep -q kibana; then
    echo "Stopping Kibana Docker container..."
    docker-compose stop kibana
  else
    echo "No Kibana Docker container found"
  fi
}

status(){
  echo "Elasticsearch:"
  if command -v docker >/dev/null 2>&1 && docker ps --filter "name=elasticsearch" --filter "status=running" | grep -q elasticsearch; then
    echo "  running (Docker container)"
  elif curl -s --max-time 5 http://elasticsearch:9200/_cluster/health >/dev/null 2>&1; then
    echo "  running (accessible on elasticsearch:9200)"
  else
    echo "  not running"
  fi

  echo "Redis:"
  if command -v docker >/dev/null 2>&1 && docker ps --filter "name=redis" --filter "status=running" | grep -q redis; then
    echo "  running (Docker container)"
  elif redis-cli -h redis -p 6379 ping 2>/dev/null | grep -q PONG; then
    echo "  running (accessible on redis:6379)"
  elif uv run python -c "import redis; redis.Redis(host='redis', port=6379).ping()" 2>/dev/null; then
    echo "  running (accessible on redis:6379 via Python)"
  else
    echo "  not running"
  fi

  echo "Kibana:"
  if command -v docker >/dev/null 2>&1 && docker ps --filter "name=kibana" --filter "status=running" | grep -q kibana; then
    echo "  running (Docker container)"
  elif curl -s --max-time 5 http://kibana:5601/api/status >/dev/null 2>&1; then
    echo "  running (accessible on kibana:5601)"
  else
    echo "  not running"
  fi
}

case "$action" in
  start)
    start_es
    es_result=$?
    start_redis
    redis_result=$?
    start_kibana
    kibana_result=$?

    if [ $es_result -eq 0 ] && [ $redis_result -eq 0 ] && [ $kibana_result -eq 0 ]; then
      echo "All services are running. Use 'scripts/execution/run-local.sh status' to check."
      exit 0
    else
      echo "Some services failed to start. Check the output above."
      exit 1
    fi
    ;;
  stop)
    stop_redis
    stop_es
    stop_kibana
    ;;
  status)
    status
    ;;
  help|--help|-h)
    cat <<'EOF'
Usage: scripts/execution/run-local.sh [start|stop|status|help]

Commands:
  start   Start Elasticsearch, Redis, and Kibana via Docker Compose
  stop    Stop the Docker containers
  status  Show status of Docker containers and service accessibility
  help    Show this message.

Requirements:
- Docker and Docker Compose must be installed
- Services will be accessible at:
  * Elasticsearch: http://localhost:9200
  * Kibana: http://localhost:5601
  * Redis: localhost:6379

Notes:
- If Docker is not available, status will check service accessibility via ports
- Data is persisted in Docker named volumes
EOF
    ;;
  *)
    echo "Unknown action: $action"; exit 2
    ;;
esac
