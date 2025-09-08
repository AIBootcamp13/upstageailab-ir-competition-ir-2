#!/usr/bin/env bash
set -euo pipefail

# Manage systemd user services for the local Elasticsearch and Redis started via scripts/
# Usage: scripts/manage-services.sh install|uninstall|status

CMD="$1"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

ES_UNIT="$UNIT_DIR/information-retrieval-elasticsearch.service"
REDIS_UNIT="$UNIT_DIR/information-retrieval-redis.service"

case "$CMD" in
  install)
    cat > "$ES_UNIT" <<'EOF'
[Unit]
Description=Information Retrieval - Elasticsearch (local)
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/workspace/information-retrieval-prj
ExecStart=%h/workspace/information-retrieval-prj/scripts/start-elasticsearch.sh --foreground 8.9.0
Restart=on-failure

[Install]
WantedBy=default.target
EOF

    cat > "$REDIS_UNIT" <<'EOF'
[Unit]
Description=Information Retrieval - Redis (local)
After=network.target

[Service]
Type=simple
WorkingDirectory=%h/workspace/information-retrieval-prj
ExecStart=%h/workspace/information-retrieval-prj/scripts/start-redis.sh --foreground 7.2.0
Restart=on-failure

[Install]
WantedBy=default.target
EOF

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
