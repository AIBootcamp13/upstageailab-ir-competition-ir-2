#!/usr/bin/env bash
set -euo pipefail

# Cleanup downloaded/extracted distributions under the project root
# Usage: scripts/cleanup-distros.sh [--keep-readme]

KEEP_README=0
for arg in "$@"; do
  case "$arg" in
    --keep-readme) KEEP_README=1 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

for d in elasticsearch-* redis-*; do
  if [ -d "$d" ]; then
    if [ "$KEEP_README" -eq 1 ] && [ -f "$d/README.asciidoc" ]; then
      echo "Preserving $d/README.asciidoc and removing rest"
      tmpdir="/tmp/keep_${d}"
      mkdir -p "$tmpdir"
      mv "$d/README.asciidoc" "$tmpdir/"
      rm -rf "$d"
      mkdir -p "$d"
      mv "$tmpdir/README.asciidoc" "$d/"
      rm -rf "$tmpdir"
    else
      echo "Removing $d"
      rm -rf "$d"
    fi
  fi
done

echo "Cleanup complete"
