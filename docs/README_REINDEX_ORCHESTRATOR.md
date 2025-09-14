Reindex Orchestrator — README
=============================

What this does
--------------

`scripts/maintenance/index_orchestrator.py` is a conservative CLI utility to help you safely reindex data and atomically swap an alias in Elasticsearch.

Key features
------------
- Create a timestamped target index (copies settings & mappings from the source).
- Run `_reindex` asynchronously and poll the Tasks API until completion.
- Optional verification (`--verify`) that compares document counts before alias swap.
- Optional rollback (`--rollback-on-failure`) to delete the target index when verification fails.
- Optional recompute mode (`--recompute-embeddings`) that calls `scripts/maintenance/recompute.stream_and_index` to recompute embeddings from a JSONL file and bulk ingest into the new index.

Quick examples
--------------

- Dry-run (show planned actions only):

```bash
poetry run python scripts/maintenance/index_orchestrator.py --alias documents --dry-run
```

- Full reindex + verify + alias swap:

```bash
poetry run python scripts/maintenance/index_orchestrator.py --alias documents --verify --force
```

- Recompute embeddings from `data/documents.jsonl` and index into the new target (dry-run):

```bash
poetry run python scripts/maintenance/index_orchestrator.py --alias documents --recompute-embeddings --documents-path data/documents.jsonl --dry-run
```

Notes and best practices
------------------------
- Always start with `--dry-run` to confirm planned actions.
- Use `--verify` to run a doc-count check before swapping the alias; this is a lightweight sanity check. For more robust validation, run the project's retrieval tests against the new index (see plan document).
- `--recompute-embeddings` requires `data/documents.jsonl` to exist and a repository embedding loader available. The recompute module is a scaffold: it will try to find `ir_core.embeddings.compute_embeddings` or similar helper functions. If none are found, it falls back to zero-vectors (useful for testing only).
- The orchestrator writes to stdout and returns a non-zero exit code on failures suitable for CI scripts.

Troubleshooting
---------------
- If `--recompute-embeddings` fails because an embedding function cannot be found, implement a small wrapper that exposes a `compute_embeddings(batch)` function and importable under `ir_core.embeddings`.
- For large datasets, tune `--batch-size` when recomputing to match your hardware.

Security
--------
- The local sample configs used for Kibana and payloads disable some protections for local development convenience. Do not use those settings in production.
