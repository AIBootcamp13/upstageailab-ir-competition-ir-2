Kibana local run (no-sudo)
===========================

This document explains how to run Kibana locally from the prebuilt tarball and the repository helper script.

Quick start (downloads and starts Elasticsearch, Redis, and Kibana):

```bash
./scripts/execution/run-local.sh start
```

Kibana local run (no-sudo)
===========================

This document explains how to run Kibana locally from the prebuilt tarball and the repository helper script.

Quick start (downloads and starts Elasticsearch, Redis, and Kibana):

```bash
./scripts/execution/run-local.sh start
```

Check status:

```bash
./scripts/execution/run-local.sh status
```

Stop services:

```bash
./scripts/execution/run-local.sh stop
```

Notes:
- The script downloads Kibana into `.local_kibana/` and creates a minimal `kibana.yml` that points to `http://127.0.0.1:9200`.
- By default Kibana is started with `NODE_OPTIONS=--max-old-space-size=2048`. You can change this in the script or set `KIBANA_HEAP` before calling the script.
- The helper disables xpack security in the local `kibana.yml` for convenience. Do not use this config in production.

If you prefer to run Kibana manually:

1. Download `kibana-8.9.0-linux-x86_64.tar.gz` from the Elastic artifacts site.
2. Extract to a folder and edit `config/kibana.yml` to set `elasticsearch.hosts: ["http://127.0.0.1:9200"]`.
3. Start with:

```bash
export NODE_OPTIONS=--max-old-space-size=2048
./bin/kibana
```

See also: `docs/README_REINDEX_ORCHESTRATOR.md` for the index orchestrator and recompute instructions.
