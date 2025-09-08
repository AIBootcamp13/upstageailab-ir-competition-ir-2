# Smoke test: embeddings + hybrid retrieval

This tiny smoke test exercises the new transformers-based embedding wrapper
and the hybrid retrieval reranker without requiring a running Elasticsearch
cluster. It uses a synthetic BM25 candidate list and calls `hybrid_retrieve`.

Files
- `scripts/smoke_test.py` — self-contained test script.

Prerequisites
- Python 3.10
- Install minimal runtime dependencies (recommended inside the `refactor` folder):

```pwsh
poetry install
# Then install a compatible torch wheel manually if you need GPU support
# See README.md for the exact torch wheel command for your platform.
```

Quick run

From the `refactor` directory run:

```pwsh
# run the smoke test
python scripts/smoke_test.py
```

What it does
- Encodes a small set of Korean sentences using the model configured by
  `refactor/src/ir_core/config.py` (the `EMBEDDING_MODEL` setting).
- Builds a synthetic set of BM25 hits and monkeypatches `sparse_retrieve` to
  return those hits.
- Calls `hybrid_retrieve` to rerank candidates by cosine similarity and
  prints the top results.

If the test fails to import `ir_core`, make sure you run it from the
`refactor` directory so that `refactor/src` is on the Python path, or run:

```pwsh
python -c "import sys; sys.path.insert(0,'refactor/src'); import scripts.smoke_test;"
```

Notes
- The first run will download the selected HF model weights and may take
  several minutes depending on your connection.
- This script is intentionally minimal; after confirming the smoke test you
  should run a full integration test against a local Elasticsearch instance
  (docker-compose in the repository can help with that).
