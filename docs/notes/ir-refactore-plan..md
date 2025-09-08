# IR Refactor Plan — Modern, modular RAG project

This document defines a practical, modern project layout and phased plan so the current notebook-based prototype can be refactored into a reusable, testable, and production-ready codebase. It includes an assessment of the proposed architecture, modifications to support easy experimentation and submission generation, and notes about a later LibreChat integration.

---

## Quick summary / recommendation

- The proposed multi-service design (FastAPI backend, Streamlit UI, abstracted vector stores, Hydra configs, WandB logging) is appropriate and portable to real projects.
- Start small: extract core logic into a single Python package (`src/ir_core`) and add thin adapters for FastAPI / Streamlit later. This minimizes early complexity while preserving the ability to scale.
- The plan below adds experiment/config management (Hydra), reproducible runs (Docker + docker-compose), unit/integration tests, and a clear path to generate evaluation submissions.

---

## Checklist (deliverables in this plan)

- Project layout (modern Python package layout)
- Minimal MVP tasks to get a working, testable RAG pipeline
- Experiment & config guidance (Hydra + WandB + seeded runs)
- Testing, CI, and local reproducible infra (docker-compose for ES/Chroma/Redis)
- Submission generation flow and evaluation hooks
- LibreChat integration notes (deferred to final stage)

---

## Recommended project layout (explicit)

```
information_retrieval_rag/
├─ pyproject.toml
├─ README.md
├─ docker-compose.yml    # elasticsearch, chromadb, redis (optional)
├─ .env.example
├─ src/
│  ├─ ir_core/
│  │  ├─ __init__.py
│  │  ├─ config.py            # dataclass/consts and config loader (hydra-friendly)
│  │  ├─ es_client.py         # ES client helpers, mappings, health checks
│  │  ├─ embeddings.py        # model loader, encode_batch, normalization
│  │  ├─ indexing.py          # chunking, bulk index, reindex missing embeddings
│  │  ├─ retrieval.py         # sparse/dense/hybrid retrieval & rerankers
│  │  ├─ rag.py               # assemble context and call LLM client (stateless)
│  │  ├─ eval.py              # metrics (P@k, MRR), evaluation runner, CSV/JSONL writer
│  │  ├─ stores/              # adapters: elasticsearch_store.py, chroma_store.py
│  │  ├─ utils.py             # IO, jsonl helpers, text normalization, logging setup
│  │  └─ tests/               # pytest tests for metrics, rerank, small indexing
│  ├─ api/
│  │  └─ fastapi_app.py       # simple FastAPI app wrapping rag.rag_infer
│  ├─ app/
│  │  └─ streamlit_app.py     # optional Streamlit UI for experimentation
│  └─ scripts/
│     ├─ reindex.py           # CLI script to compute embeddings + index
│     └─ evaluate.py          # run evaluation and save submission
└─ data/
	 ├─ documents.jsonl
	 └─ eval.jsonl
```

---

## Phased plan (practical, minimal overhead)

Phase 0 — Prep (0.5 day)

- Create `pyproject.toml`, minimal `src/` layout, basic README, and `.env.example`.
- Add `docker-compose` for Elasticsearch (and Chromadb if desired). Start and verify ES mapping works.

Phase 1 — Core package & indexing (1–2 days)

- Implement `es_client` and `embeddings` (model loader with lazy init). Write `indexing.index_documents()` that: chunk texts, compute embeddings in batches, and bulk-index documents with an `embedding_model` metadata field.
- Provide `indexing.reindex_missing_embeddings()` which computes embeddings for docs missing the `embeddings` field.

Phase 2 — Retrieval & evaluation (1–2 days)

- Implement `retrieval.sparse_retrieve`, `retrieval.dense_retrieve`, `retrieval.hybrid_retrieve` (bm25_k, rerank_k, optional interpolation `alpha`). Add unit tests for reranker behavior.
- Implement `eval.run_eval()` that computes P@1/3/5 and MRR given a labeled subset. Add capability to produce submission files (CSV/JSONL).

Phase 3 — Orchestration & API (1–2 days)

- Add CLI `scripts/reindex.py` and `scripts/evaluate.py` (Hydra config support) and a small FastAPI endpoint for `/rag/answer` used by the Streamlit UI.

Phase 4 — Experimentation & CI (ongoing)

- Add Hydra configs (`conf/`) that parameterize model, vector store, `bm25_k`, `alpha`, and logging.
- Integrate WandB for experiment tracking (optional); otherwise store run metadata locally under `outputs/`.
- Add GitHub Actions: lint, pytest, and a small integration test using docker-compose (start ES, run a tiny indexing + retrieval smoke test).

Phase 5 — Productionization & LibreChat (deferred)

- After retrieval & evaluation metrics are stable, attach a frontend integration like LibreChat. LibreChat is primarily a UI and orchestration layer — integrate it after a stable, documented API exists.

---

## Assessment of your proposed design

- Strengths: The clear separation between UI / backend / storage and using Hydra/WandB for experiments is well aligned with production workflows.
- Risks & mitigations:
	- Risk: building API + UI + multiple stores at once increases scope. Mitigate by incremental extraction: implement `ir_core` first, then adapters for stores and UI.
	- Risk: embedding/model mismatch between indexing and query time. Mitigate by storing `embedding_model` in index metadata and refusing dense retrieval if mismatch.
	- Risk: Elasticsearch plugin installation (nori) may require node restart and appropriate permissions. Prefer Docker images prebuilt with plugins for CI/infra.

---

## Experimentation & submission workflow

1. Config-first: use Hydra to make runs reproducible and compare `bm25_k`, `alpha`, and `embedding_model` settings.
2. Small labeled validation set (20–50 queries): use it to tune `bm25_k` and interpolation weight. Keep it in `data/validation.jsonl` and reference in configs.
3. Submission runner: `scripts/evaluate.py --hydra.run.dir=outputs/run_x` should load `eval.jsonl`, run retrieval, and write final submission to `outputs/submission-{run_id}.csv`.
4. Logging: store retrieval candidates and cosines for each query to help error analysis; save `outputs/{run_id}_candidates.jsonl`.

---

## Tests & quality gates

- Unit tests: metrics, encoding normalization, reranker sorting.
- Integration smoke test: start ES (docker-compose), index 50 synthetic docs, run hybrid retrieve for a few queries and assert non-empty results.
- CI: run linter + pytest; optional integration job runs only on `main` or via a workflow_dispatch trigger.

---

## LibreChat integration notes (deferred)

- Defer LibreChat integration until the RAG pipeline is stable and metrics acceptable.
- Integration steps: expose `/rag/answer` endpoints; map UI conversation flows to your `rag.rag_infer()` function; add conversation/metadata storage for auditability.

---

## Minimal MVP to aim for in one week

1. `src/ir_core/embeddings.py` and `es_client.py` implemented and tested.
2. `indexing.py` that indexes documents with embeddings (or a `reindex_missing_embeddings()` helper).
3. `retrieval.py` with sparse/dense/hybrid retrieval and a small notebook that shows overlap stats and candidate cosines.
4. `scripts/evaluate.py` that can produce a submission file for the competition eval set.

---

## Immediate next steps I can do for you

- Generate the `src/ir_core` package skeleton with minimal module stubs and a `scripts/` folder (recommended), or
- Create `scripts/evaluate.py` that reads Hydra config, runs hybrid retrieval, and writes submission CSV, or
- Add unit tests for `precision_at_k` and `mrr` and wire them into pytest.

Which of those three would you like me to create now? I recommend the package skeleton first.

---

Notes

- Keep changes incremental. The priority is to make the core reproducible and testable before adding UI or third-party integrations.

