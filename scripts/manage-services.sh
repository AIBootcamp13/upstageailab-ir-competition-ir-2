#!/usr/bin/env bash
set -euo pipefail

# Manage systemd user services for the local Elasticsearch and Redis started via scripts/
# Usage: scripts/manage-services.sh install|uninstall|status

CMD="$1"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

# Derive absolute project path (assume script lives in the repo's scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ES_UNIT="$UNIT_DIR/information-retrieval-elasticsearch.service"
REDIS_UNIT="$UNIT_DIR/information-retrieval-redis.service"

case "$CMD" in
  install)
  printf '%s' "[Unit]\nDescription=Information Retrieval - Elasticsearch (local)\nAfter=network.target\n\n[Service]\nType=simple\nWorkingDirectory=$PROJECT_ROOT\nExecStart=$PROJECT_ROOT/scripts/start-elasticsearch.sh --foreground 8.9.0\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n" > "$ES_UNIT"

  printf '%s' "[Unit]\nDescription=Information Retrieval - Redis (local)\nAfter=network.target\n\n[Service]\nType=simple\nWorkingDirectory=$PROJECT_ROOT\nExecStart=$PROJECT_ROOT/scripts/start-redis.sh --foreground 7.2.0\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n" > "$REDIS_UNIT"

    systemctl --user daemon-reload
    systemctl --user enable --now information-retrieval-elasticsearch.service || true
    systemctl --user enable --now information-retrieval-redis.service || true
    echo "Installed and started user services (if systemd user is available)."
    ;;
  uninstall)
    systemctl --user disable --now information-retrieval-elasticsearch.service || true
    systemctl --user disable --now information-retrieval-redis.service || true
    rm -f "$ES_UNIT" "$REDIS_UNIT"
    systemctl --user daemon-reload || true
    echo "Uninstalled services"
    ;;
  status)
    systemctl --user status information-retrieval-elasticsearch.service || true
    systemctl --user status information-retrieval-redis.service || true
    ;;
  *)
    echo "Usage: $0 install|uninstall|status"
    exit 2
    ;;
esac
