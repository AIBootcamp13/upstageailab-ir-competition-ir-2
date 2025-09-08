# Docker-less workarounds for Elasticsearch and Redis

This document explains small scripts in `scripts/` to run Elasticsearch and Redis on a Linux host without Docker. These are intended for local development only.

## Elasticsearch

Script: `scripts/start-elasticsearch.sh [version]` (defaults to 8.9.0)

Behavior:
- Downloads the official Linux tarball into the project root if `elasticsearch-<version>` is missing.
- Extracts it and appends a minimal `config/elasticsearch.yml` for single-node dev mode and disables security (only for local dev).
- Starts Elasticsearch in the background with logs under the distribution `logs/` folder.

Notes:
- The script disables `xpack.security` for convenience. Don't use this in production.
- If you prefer a different version, pass it as the first argument.

Example:

```bash
./scripts/start-elasticsearch.sh 8.9.0
curl http://127.0.0.1:9200
```

## Redis

Script: `scripts/start-redis.sh [version]` (defaults to 7.2.0)

Behavior:
- Downloads the official Redis source tarball and builds it locally (requires `make` and a C toolchain).
- Starts `redis-server` in the background using a project-local `data/` directory and writes logs under `logs/`.

Notes:
- Building Redis requires `build-essential` or equivalent packages installed on Linux.

Example:

```bash
./scripts/start-redis.sh 7.2.0
redis-cli -p 6379 ping
```

## Security and cleanup
- These scripts are for development convenience only. They do not configure production-grade security or resilience.
- To stop the servers, find the PID printed by the script and `kill <pid>`.
- The downloaded distributions are placed under the repo root (and `.gitignore` ignores `elasticsearch-*` patterns). If you want to avoid storing large tarballs in the repo tree, delete the extracted folders after use or run the scripts from a temporary location.

## Alternatives
- If you later get Docker access, using the official Docker images is still recommended for parity with production.
- Use a remote managed service (Elastic Cloud, managed Redis) if local install is not feasible.
