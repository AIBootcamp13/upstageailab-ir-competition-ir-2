# Data context and index guidance for Information Retrieval RAG

This document is written for human and AI contributors who work on the Information Retrieval RAG project. It describes the dataset shape, indexing conventions, common workflows (reindex, alias swap), and quick checks you can run locally.

Purpose
-------
This repository contains a corpus of scientific documents used for retrieval-augmented generation experiments. The corpus is too large to inspect manually during routine tasks, and switching embedding encoders requires reindexing. This document centralizes the facts and commands agents need to work efficiently and avoid redundant work.

Quick facts
-----------
- Document store: Elasticsearch (localhost:9200 by default)
- Documents: ~4,700 documents located at `data/documents.jsonl`.
- Evaluation / Validation sets:
  - `data/eval.jsonl` (~220 examples)
  - `data/validation.jsonl` (~50 examples)
- Typical retrieval pipeline: query rewriting -> retrieval (sparse and/or dense) -> re-ranking -> generation
- Embedding model setting: configured via project settings (e.g. `model.embedding_model` in hydra config)

Data shape
----------
Each document in `data/documents.jsonl` is a JSON object with at least the following fields:
- `docid` (string UUID) — primary identifier for the document
- `src` (string) — source dataset name or partition (e.g. `ko_mmlu__nutrition__test`)
- `content` (string) — the textual content used for indexing and embedding

Why reindexing matters
----------------------
- Embeddings are numeric vectors produced by an embedding model. Different models produce different vectors (and sometimes different dimensions). If you change the embedding model you must update stored document embeddings used by dense retrieval.
- If you change model dimensionality you cannot update the existing `dense_vector` field on an index — you must create a new index with the correct mapping.

Index naming & alias strategy (recommended)
-------------------------------------------
Use a versioned index name plus an alias for the active data. Example:
- Index name: `documents_v2_onnx-768` or `documents_v3_koembed-1024`
- Alias name: `documents` (used by the application and pipeline)

Flow to create a new index and switch atomically:
1. Create a new index with mapping for the new dense vector dimension (if using dense retrieval).
2. Compute embeddings for all documents using the chosen embedding model and bulk-index them into the new index.
3. Point the application alias to the new index:
   - `POST /_aliases` with actions to add alias to new index and remove alias from old index in one request.
4. Optionally, keep the old index for rollback for some time, then delete it once you're confident.

Mapping example (dense vector field)
-----------------------------------
A minimal mapping for dense vector field (Elasticsearch 8.x):
```
PUT /documents_v2
{
  "mappings": {
    "properties": {
      "docid": {"type": "keyword"},
      "src": {"type": "keyword"},
      "content": {"type": "text"},
      "embedding": {"type": "dense_vector", "dims": 768}
    }
  }
}
```
Replace `dims` with the embedding output dimension of your chosen model.

Quick reindex recipe (high level)
---------------------------------
- Input: `data/documents.jsonl` and a chosen embedding model
- Output: new ES index with embeddings
Steps:
1. Create new index with mapping (see above).
2. Stream documents, compute embeddings in batches, bulk-index into ES with `_bulk`.
3. Update alias atomically.

Practical tips for ~4.7k docs
----------------------------
- 4.7k documents is small enough to re-embed on a single GPU (if using GPU) or CPU in acceptable time. You can run the whole reindex in minutes to an hour depending on model and hardware.
- Use batching (e.g. 64–512) to compute embeddings and bulk-index.
- Use `refresh=false` during bulk-ing and refresh only at the end to speed indexing.
- Throttle concurrency to avoid overwhelming ES; a single moderate client with batches is fine for this corpus.

Commands and scripts
--------------------
- Create index: use Elasticsearch client or `curl` for the mapping shown above.
- Bulk-index: use `helpers.bulk` from `elasticsearch-py` or the project's bulk utilities if present.
- Alias swap example (atomic):
```
POST /_aliases
{
  "actions": [
    {"remove": {"index": "documents_v1", "alias": "documents"}},
    {"add":    {"index": "documents_v2_ko-768", "alias": "documents"}}
  ]
}
```

How to visually check and manage indices
----------------------------------------
- Use Kibana (if available) — index management UI shows indices, aliases, settings and mappings.
- Use curl/HTTP API:
```
# list indices
curl -sS "http://localhost:9200/_cat/indices?v" | less

# list aliases
curl -sS "http://localhost:9200/_cat/aliases?v"

# get mapping for index
curl -sS "http://localhost:9200/documents_v2/_mapping" | jq .
```
- Use Python elasticsearch client for programmatic checks (see scripts in `scripts/maintenance/`).

Automatic switching when embedding models change
------------------------------------------------
- You can make switching automatic by implementing a reindex orchestrator that:
  1. Accepts a new embedding model name
  2. Creates the new index mapping (with proper dims)
  3. Computes embeddings and bulk-indexs
  4. Swaps alias atomically
- The orchestration can be a small script (`scripts/maintenance/reindex_embeddings.py`) that reads `data/documents.jsonl` and the configured model in `conf/` or from CLI args.
- Add a check that verifies the new index has the expected vector dimension before swapping aliases.

Safety and rollback
-------------------
- Always keep the old index for a short time and do not delete it immediately.
- Run evaluation (e.g. `scripts/evaluation/validate_retrieval.py` with a small sample) against the new index before alias swap. Keep a measurable metric (MAP) to compare.

What agents need to know about the data (short)
----------------------------------------------
- Don’t attempt to print or exhaustively inspect `data/documents.jsonl` — it’s large. Instead, compute and record simple statistics:
  - Document count
  - Distribution of `src` values (which datasets are present and their counts)
  - Average and median content length (characters / tokens)
  - Any documents missing mandatory fields (`docid`, `content`)
- Use these stats to detect accidental changes in the corpus size or schema.

Suggested .github checklist for reindex tasks
--------------------------------------------
- [ ] Confirm embedding model name and expected dim
- [ ] Create new index with mapping
- [ ] Re-embed and bulk-index into new index (batches)
- [ ] Run sample validation set and compare MAP/metrics
- [ ] Swap alias
- [ ] Monitor system, then delete old index after N days

Contact and context
-------------------
- Project overview: `docs/notes/project-overview.md`
- Data files: `data/documents.jsonl`, `data/eval.jsonl`, `data/validation.jsonl`, `data/sample_submission.jsonl`

Appendix: small utilities
------------------------
- If you want, I can add `scripts/maintenance/reindex_embeddings.py` which:
  - reads documents
  - computes embeddings via the project's embedding utility
  - bulk indexes to ES
  - performs the alias swap

If you want me to create that helper script now, say "Create reindex script" and I will add it and wire a minimal CLI to the repo.
