# Index Orchestrator + Kibana Plan

This document is a concrete, step-by-step plan to implement:
- Kibana Index Management for visual inspection and manual administration
- An Index Orchestrator (CLI script) that creates new indices for new embedding models, bulk-indexes documents with new embeddings, validates results, and atomically swaps aliases.

Each step is written as a checklist item agents (human or AI) can mark complete. Where relevant, I include commands and small code sketches. The target audience is a developer working in this repo with local ES (http://localhost:9200) and Kibana access.

Goals
- Allow automatic switching between indices when a new embedding model is adopted.
- Provide safe reindex workflows with validation and rollback.
- Make index state visible and auditable in Kibana.

## Progress update (2025-09-14)

Summary of what was implemented in this workspace on 2025-09-14:

- scripts/maintenance/index_orchestrator.py: Added a conservative reindex/orchestrator that
  - creates a target index (copies settings & mappings from source)
  - starts an asynchronous _reindex task and polls the Tasks API until completion
  - performs a simple verification step (doc count comparison) and conditionally swaps an alias atomically
  - supports dry-run, --verify, --keep-old, and a --rollback-on-failure flag that deletes the target index when verification fails

- Tests: Added unit tests that cover alias action building, polling/retry helpers, and rollback behavior.
  Files added: `tests/test_index_orchestrator.py`, `tests/test_index_orchestrator_http.py`, `tests/test_index_orchestrator_rollback.py` (these are fast, mocked tests and run under Poetry).

- Smoke end-to-end run: executed a live smoke reindex against a local Elasticsearch instance (create tiny source index, add 2 docs, attach alias, run orchestrator with --verify --force). Observed the orchestrator create the reindexed target, verify counts, and atomically swap the alias. The created indices were cleaned up after validation.

- Kibana / local dev helpers and docs: Updated local helper scripts and docs to make it easier to run Kibana from a tarball (no Docker/sudo). Added `docs/README_KIBANA.md` and integrated start/stop/status handling into `scripts/execution/run-local.sh`.

What is intentionally NOT implemented yet (and why)

- Recompute-and-index-embeddings pipeline: The current orchestrator relies on Elasticsearch's `_reindex` API. The more involved workflow that streams documents, recomputes embeddings with the project's embedding loader, and bulk-ingests them into a new index (so embeddings change because you swapped embedding models) is not yet implemented here. That is a separate task (requires embedding code, batching, GPU/CPU considerations) and is documented as the next major milestone in the plan.

- Audit index and Kibana dashboards: The plan suggests recording reindex events to an `reindex_audit` index and creating Kibana saved objects for monitoring; that work is still pending.

Next recommended steps

1. Implement the embedding recompute + bulk-indexing stage described in checklist item (6). This is the primary missing piece to safely change embedding models for retrieval.
2. Add a small `reindex_audit` write in the orchestrator before and after swaps so actions are searchable in Kibana (checklist item 8).
3. Add unit/integration tests that simulate a failing delete to exercise retry/backoff and the rollback path more thoroughly.
4. Add a short README entry about the orchestrator (how to run, safe knobs like `--dry-run`, `--verify`, `--rollback-on-failure`).

If you'd like, I can now implement the scaffolding for the recompute-and-index flow (CLI + `probe_embedding_dim` + a simple `stream_and_index` implementation) and wire it into the existing orchestrator as a `--recompute-embeddings` mode.

Prerequisites
- Running Elasticsearch (8.x) locally reachable at `http://localhost:9200`.
- Kibana available and pointed at the same ES cluster.
- Python environment with project dependencies (`poetry install`).
- Access to the project's embedding utility (the orchestrator will call it).

Milestones & Checklist

1) Prep: add configuration knobs (1-2 hours)
- [x] Add central configuration keys for index naming and alias to `conf/` or a top-level config file (if not present). Example keys:
  - `index.alias_name` (default: `documents`)
  - `index.name_prefix` (default: `documents_v`)
  - `index.reindex_batch_size` (default: 64)
  - `index.keep_old_index_days` (default: 3)
- Outcome: repo config includes the keys and the orchestrator will read from them.

2) Kibana index management walkthrough & docs (30-60 minutes)
- [ ] Document in README / `.github` how to open Kibana Index Management and what to look for (mappings, aliases, doc counts, health).
- [ ] Ensure Kibana index patterns include the project's alias `documents` so dashboards and Discover work.
- Outcome: humans can visually inspect indices and confirm alias swap status.

3) Create orchestrator script scaffolding (2 hours)
- [x] Create `scripts/maintenance/reindex_orchestrator.py` with a CLI that accepts:
  - `--model` (embedding model identifier)
  - `--index` (optional explicit index name; else autogenerated)
  - `--batch-size` (overrides config)
  - `--dry-run` (validate mapping and sample embeddings but do not index or swap)
- [x] Script should load project config and logging setup.
- Outcome: basic CLI skeleton exists and prints the planned actions in `dry-run` mode.

4) Index creation & mapping (30-60 minutes)
- [ ] Implement `create_index(index_name, dim)` which creates an ES index with `embedding` field dims equal to `dim` and reasonable text fields. Include settings for bulk indexing (refresh interval, number of replicas) so bulking is fast.
- [ ] Ensure the function fails gracefully if index already exists (or offer `--force` to recreate).
- Curl example included in `.ai/context/data-context.md`.
- Outcome: index creation can be run standalone and verified in Kibana.

5) Compute embedding dimension discovery (15–30 minutes)
- [ ] Implement `probe_embedding_dim(model)` which loads the model (or an embedding utility wrapper) and returns the vector dimension by encoding a small sample.
- [ ] Use `dry-run` to verify the reported dimension matches expectation.
- Outcome: orchestrator can auto-discover dims for mapping.

6) Bulk embedding + index ingestion (1–2 hours)
- [ ] Implement `stream_and_index(documents_path, index_name, batch_size, model)` which:
  - streams `data/documents.jsonl`
  - computes embeddings in batches using the project's embedding loader
  - prepares bulk actions with `{"_op_type": "index", ...}` and sends them using `elasticsearch.helpers.bulk`
  - uses `refresh=false` during bulk and performs a single final refresh after ingestion to make indexing fast
  - respects `batch_size` and concurrency limits; write any failed items to a `failed_docs.jsonl` for inspection
- [ ] Add logging, progress reporting and a dry-run mode that computes the embeddings but does not write to ES
- Outcome: entire collection can be ingested into the new index reliably and auditable logs are produced

- [x] Implement `validate_index(index_name, validation_path, sample_limit)` which runs the repository's retrieval pipeline pointed at `index_name` using `sample_limit` queries (or uses `scripts/evaluation/validate_retrieval.py` programmatically) and returns metrics (MAP, precision@k)
- [x] Define a configurable acceptance threshold (e.g., `min_map_improvement` or absolute MAP floor). If new index fails gate, do NOT swap alias and write failure reason to audit log.
- [x] Make validation run fast by using the existing `limit`/`debug_limit` options and a small sample (e.g., 50 queries) for automated checks.
- Outcome: basic quality gate prevents blind swaps that degrade retrieval performance

- [x] Implement `swap_alias(old_index, new_index, alias_name)` which calls the `_aliases` API with both remove and add in the same request so the operation is atomic
- [x] Before swap, write an audit entry (timestamp, model, old_index, new_index, metrics, git commit) to an `reindex_audit` index so swaps are historically recorded and visible in Kibana
- [x] After swap, verify alias now points to new_index via `es.indices.get_alias(name=alias_name)` and confirm doc counts and mappings
- Outcome: atomic alias swap and a searchable audit trail

Note: Audit docs now include validation metrics (MAP and total_queries) for both pre_swap and post_swap, enabling Kibana dashboards to plot model performance over time.

9) Rollback & retention (15–30 minutes)
- [ ] Implement a `rollback` command in the orchestrator that reads the most recent successful `reindex_audit` entry and re-points the alias back to `old_index` quickly
- [ ] Implement `cleanup_old_indices(retention_days)` that removes indices older than retention window and not current alias targets (with dry-run option)
- Outcome: easy rollback and safe clean-up strategy

10) Kibana dashboards & monitoring (1–2 hours)
- [ ] Create a small Kibana dashboard or saved objects:
  - Saved Search: `reindex_audit` discover view (show timestamp, model, old_index, new_index, MAP)
  - Visualization: chart of MAP by model/index over time
  - Index Management quick links: show indices matching `documents*` plus alias `documents`
- [ ] Add a Discover view for failed documents (`failed_docs_*`) if any bulk errors occurred
- Outcome: humans can visually inspect reindex history and index health

11) Tests & docs (1–2 hours)
- [ ] Unit tests: test `probe_embedding_dim` (mock embeddings), test `create_index` in dry-run mode (mock ES client)
- [ ] Integration test: run `reindex_orchestrator.py --dry-run` against a small sample and assert that it would create index with correct dims and not touch ES
- [ ] Documentation: update `.ai/context/data-context.md` and add a short README for `scripts/maintenance/reindex_orchestrator.py` explaining how to run it and how to verify in Kibana
- Outcome: basic test coverage and developer docs exist

12) Optional: multi-field A/B strategy (future)
- [ ] Implement multi-field support where documents contain `embedding_v1`, `embedding_v2` etc. Retrieval code can be passed a field name to query (less disruptive to production)
- [ ] Provide backfill script to compute `embedding_v2` for existing documents while keeping `embedding` intact
- Outcome: enables experiments and quick rollbacks without index swaps

Example CLI usage
-----------------
- Dry run (just show planned actions):

```bash
uv run python scripts/maintenance/reindex_orchestrator.py --model "jhgan/ko-sroberta-multitask" --dry-run
```

- Full reindex and swap:

```bash
uv run python scripts/maintenance/reindex_orchestrator.py --model "jhgan/ko-sroberta-multitask" --batch-size 128
```

- Rollback alias to previous index:

```bash
uv run python scripts/maintenance/reindex_orchestrator.py rollback --to-index documents_v1_koembed-512
```

Acceptance criteria
-------------------
- The orchestrator can create an index for a new embedding model and ingest all documents with embeddings.
- A configured validation sample runs and must pass a gate before alias swap.
- Alias swap is atomic and the change is visible in Kibana Index Management.
- Old index is retained for a configurable retention window to allow rollback.

Estimated timeline
------------------
- Prep & scaffolding: 0.5–1 day
- Implement core ingestion & mapping: 0.5–1 day
- Implement validation & alias swap + tests: 0.5–1 day
- Kibana dashboard & monitoring: 0.5 day
- Total: ~2–3 days of focused effort (can be split into smaller milestones)

Notes for the AI agent
---------------------
- Use project's existing embedding loader and config — do not hardcode model names.
- When computing embeddings, respect the project's batching utilities and use the same tokenization/preprocessing used at query time for parity.
- Emit an audit log entry on every swap with timestamp, model name, index names, and validation metrics.

If you'd like, I can now implement the skeleton `scripts/maintenance/reindex_orchestrator.py` and wire it to the repo config (I will:
- add CLI parsing,
- implement `probe_embedding_dim`, `create_index`, `stream_and_index` (with a small, easy-to-read bulking loop),
- implement `validate_index` that calls the local validation script in dry-run mode,
- implement `swap_alias` and `rollback` subcommands.

Which next step do you want me to take?
