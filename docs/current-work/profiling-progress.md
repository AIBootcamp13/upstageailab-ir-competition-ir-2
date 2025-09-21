# Profiling Progress Tracker

Link to plan: `docs/notes/strategies/profiling-implementation-plan.md`

Use these checkboxes to record progress. Update directly in this file, or use the helper script `scripts/data/update_progress.py`.

## Phase 1 — Per‑src document stats
- [ ] Per‑src token/word/char stats
- [ ] Long‑doc fraction per src
- [ ] Vocabulary overlap matrix
- [ ] (Optional) src clusters by vocab

## Phase 2 — Quality and dedup metrics
- [ ] Non‑alnum ratio / repeated line ratio / short unique token ratio
- [ ] Thresholds summarized in summary.json
- [ ] (Optional) MinHash near‑duplicate clusters

## Phase 3 — Chunking guidance
- [ ] Compressibility ratio per doc and per‑src
- [ ] Sentence/paragraph stats per‑src
- [ ] Updated chunking_recommendations.json

## Phase 4 — Retrieval diagnostics
- [ ] IDF extremes (global and per‑src)
- [ ] BM25 length stats per‑src
- [ ] (Optional) src answer priors

## Phase 5 — Embedding‑level signals
- [ ] Embedding norm distribution (per‑src)
- [ ] Cosine similarity distribution (sampled)
- [ ] Simple clustering with inertia/silhouette

## Phase 6 — Field/schema profiling
- [ ] Field example values per field per src
- [ ] src → category mapping saved

## Phase 7 — Query‑focused cacheables
- [x] TF‑IDF top keywords per src
- [x] Scientific term extraction via local LLM (Llama)
- [ ] Source glossary per src

## Validation & docs
- [ ] scripts/list_scripts.py updated for new flags
- [ ] profiling‑retrieval‑guide.md updated with new metrics
- [ ] summary.json includes concise overview of new metrics