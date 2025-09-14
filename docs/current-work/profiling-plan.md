# Profiling Implementation Plan (end-to-end, checkable)

This plan turns the “useful extras to collect next” list into concrete, checkable tasks you can execute incrementally. It maps each signal to: actions, code touch‑points, artifacts, and acceptance criteria. Keep using the same profiler entry point and write all artifacts under `outputs/reports/data_profile/<timestamp>/` with a `latest` symlink.

Scope notes
- Repo: `information_retrieval_rag` (branch: 05_feature/kibana)
- Runner: Poetry only. Prefer `PYTHONPATH=src poetry run python …` for scripts that import `ir_core`.
- Primary script to extend: `scripts/data/profile_documents.py` (add flags and writers as needed). Reuse utilities in `src/ir_core` when helpful.
- Artifacts directory: `outputs/reports/data_profile/` with `latest -> <timestamp>` symlink.

## Phase 0 — Baseline sanity (already available)

- [x] Unique `src` list and counts per `src` → `unique_src.json`, `src_counts.json`
- [x] Field presence/length stats → `field_presence.json`, `field_length_stats.json`, `per_src_length_stats.json`
- [x] TF‑IDF keywords per src → `keywords_per_src.json`
- [x] Global/per‑src stopwords → `stopwords_global.json`, `per_src_stopwords.json`
- [x] Exact duplicates (normalized hash) → `duplicates.json`
- [x] Near‑duplicates (SimHash + LSH) → `near_duplicates.json`
- [x] Summary → `summary.json`
- [x] Latest symlink maintenance

Definition of Done: profiler runs end‑to‑end on `data/documents.jsonl` and writes the above; retrieval imports succeed.

Optional run
```bash
# Baseline run
poetry run python scripts/data/profile_documents.py \
  --file_path data/documents.jsonl \
  --out_dir outputs/reports/data_profile \
  --save 1
```

## Phase 1 — Per‑src document stats (depth)

Goals: richer per‑src stats to guide chunking and cross‑domain boosts.

Tasks
- [ ] Compute per‑src token/word/char stats, fraction of long docs (e.g., top 10% by words)
- [ ] Compute vocabulary overlap between src groups/clusters

Implementation
- Extend `profile_documents.py`:
  - Add `--tokenizer` option (simple whitespace default; optionally reuse any available tokenizer if configured)
  - Aggregate per‑src: mean/median/P90/P95 for tokens/words/chars; long‑doc fraction threshold configurable (e.g., `--long_doc_pct 0.9`)
  - Build per‑src vocabulary sets using TF‑IDF/CountVectorizer; compute pairwise Jaccard or overlap coefficients; optionally cluster src by overlap (agglomerative/k‑means on Jaccard distance)

Artifacts
- `per_src_length_stats.json` (extend with tokens/long_doc_fraction)
- `vocab_overlap_matrix.json` (N×N map; sparse list or top‑K neighbors per src)
- `src_clusters_by_vocab.json` (optional)

Definition of Done
- [ ] New artifacts exist and values look reasonable on a small printed preview in `summary.json`
- [ ] No >1.5× slowdown vs baseline for default settings (can add `--max_features` to cap vocab)

## Phase 2 — Quality and dedup metrics (precision+noise control)

Goals: flag boilerplate/low‑content docs; strengthen near‑dup fidelity.

Tasks
- [ ] Boilerplate/low‑content ratios: non‑alnum ratio, repeated line ratio, short unique token ratio
- [ ] MinHash Jaccard estimation for near‑dup validation (optional, complement SimHash)

Implementation
- Add text cleanliness metrics in `profile_documents.py`:
  - `non_alnum_ratio = non_alnum_chars / total_chars`
  - `repeated_line_ratio = repeated_lines / total_lines` (normalize lines; ignore empties)
  - `short_unique_token_ratio = unique_short_tokens / total_tokens` with a cutoff (e.g., len<=3)
- Add `--boilerplate_thresholds` to surface recommended flags in `summary.json` (don’t hard‑drop by default)
- Optionally implement MinHash (e.g., datasketch) guarded by `--minhash 1`; write clusters and overlaps

Artifacts
- `quality_metrics.jsonl` (per‑doc compact metrics)
- `boilerplate_flags.json` (docids exceeding thresholds)
- `minhash_near_duplicates.json` (if enabled)

Definition of Done
- [ ] Metrics distributions summarized in `summary.json`
- [ ] Share of flagged docs per src reported, with sensible ranges

## Phase 3 — Chunking guidance (indexing knobs)

Goals: produce signals to set chunk size/overlap per domain and for semantic vs fixed chunking.

Tasks
- [ ] Compressibility ratio: gzip_size / raw_size per doc and per‑src stats
- [ ] Sentence/paragraph counts per doc to support semantic chunking

Implementation
- In `profile_documents.py` add:
  - `--compressibility 1` to compute gzip ratio (use `gzip` lib in‑mem); aggregate per‑src stats
  - Lightweight sentence/paragraph heuristics (split on punctuation and blank lines) with language‑agnostic rules
- Extend `src/ir_core/retrieval/chunking.py` to consume new stats for recommendations

Artifacts
- `compressibility_stats.json` (per‑src)
- `sentence_paragraph_stats.json` (per‑src)
- `chunking_recommendations.json` (updated)

Definition of Done
- [ ] Recommendations include chunk_size and overlap per src, with rationale fields
- [ ] Document guidance captured in `docs/profiling-retrieval-guide.md`

## Phase 4 — Retrieval diagnostics (BM25/hybrid tuning)

Goals: surface per‑src priors and IDF extremes to guide stopwords, boosts, and hybrid tuning.

Tasks
- [ ] Per‑src BM25‑friendly features: avg doc length, variance (already partially available; consolidate)
- [ ] IDF extremes/top rare terms per src
- [ ] Per‑src priors for answering specific query types (store as priors; used downstream)

Implementation
- Extend the vectorizer run to dump IDF extremes: top‑N highest/lowest IDF tokens per src and global
- Compute per‑src avg_len and variance explicitly for BM25 `b` intuition
- Add a simple `priors.json` that can be manually curated or bootstrapped from evaluation labels when available

Artifacts
- `idf_extremes_global.json`, `idf_extremes_per_src.json`
- `bm25_len_stats.json`
- `src_answer_priors.json` (schema: {src: {topic|tag: prior}})

Definition of Done
- [ ] Extremes and stats present and summarized in `summary.json`
- [ ] Retrieval layer can optionally consume these via toggles (no hard dependency)

## Phase 5 — Embedding‑level signals (vector health)

Goals: find embedding outliers and heterogeneous sources.

Tasks
- [ ] Embedding norm distribution, cosine similarity distribution (sampled) per src
- [ ] Simple clustering per src (k‑means; record inertia/silhouette)

Implementation
- Reuse embeddings loader used in retrieval (respect batching and GPU if available)
- Add flags: `--embedding_health 1`, `--embedding_model <name>`, `--embed_sample_size <N>`
- For each src: compute norms, sample cosine similarities; compute k‑means for small K (e.g., 2–4) and record inertia; silhouette (sampled) if feasible

Artifacts
- `embedding_health.json` (per‑src: norm stats, similarity quantiles)
- `embedding_clustering.json` (per‑src: K, inertia, silhouette if computed)

Definition of Done
- [ ] Outlier thresholds proposed in `summary.json` (e.g., extremely low norm or near‑zero variance)
- [ ] Optional `boilerplate_flags.json` cross‑checked with embedding anomalies

## Phase 6 — Field/schema profiling (standardization + categories)

Goals: normalize fields and map `src` to categories for retrieval boosts and eval slices.

Tasks
- [ ] Field presence distribution with example values per field
- [ ] Map `src` → category (ARC vs MMLU, subject); maintain mapping file

Implementation
- Extend `profile_documents.py` to sample example values per field per src (safe truncation; sanitize)
- Create `src/ir_core/analysis/source_mapping.py` or update constants to persist `SRC_TO_CATEGORY`

Artifacts
- `field_examples.json` (small samples per field per src)
- `src_to_category.json`

Definition of Done
- [ ] Category mapping referenced by retrieval (optional boosts) and by evaluation slicing scripts

## Phase 7 — Query‑focused cacheables (LLM‑friendly)

Goals: reduce LLM tokens by precomputing compact descriptors.

Tasks
- [ ] Per‑src “top keywords” (TF‑IDF) — already exists
- [ ] Build a small “source glossary” per src
- [ ] Extract scientific terms using local LLM (Ollama) and merge/dedupe

Implementation
- Keep `keywords_per_src.json` as base
- Add a generator to create `source_glossary.json` per src (2–3 lines of description from sample docs; can be LLM‑assisted or heuristic)
- Use or extend `scripts/data/extract_scientific_terms.py` to process `keywords_per_src.json` with `--model qwen2:7b | llama3.1:8b` and write `scientific_terms_extracted.json`; add `--merge_threshold` for dedupe

Artifacts
- `source_glossary.json`
- `scientific_terms_extracted.json`

Definition of Done
- [ ] Glossary length bounded; scientific terms deduped and sorted per src
- [ ] Retrieval can optionally boost using these terms (toggle), or query rewriter can use them

---

## Cross‑cutting: CLI, config, and docs

- [ ] Update `scripts/list_scripts.py` with descriptions for any new flags/artifacts
- [ ] Add/extend toggles in `conf/settings.yaml` and `src/ir_core/config/__init__.py` only when features are used at query time (e.g., duplicate filtering, near‑dup penalty, boosting)
- [ ] Extend `docs/profiling-retrieval-guide.md` with how to enable/disable features and interpret artifacts

## Validation & guardrails

Minimal checks per run
- [ ] Profiler completes within expected time budget (record wall‑clock) and memory is stable
- [ ] `summary.json` includes a compact overview of newly added metrics
- [ ] JSON schemas remain consistent across runs; new keys are appended without breaking consumers

Quality gates (optional)
- [ ] `poetry run pytest -q` passes
- [ ] `PYTHONPATH=src poetry run python scripts/evaluation/smoke_test.py` passes

## How to run (patterns)

Examples (toggle features as needed)
```bash
# With keywords, stopwords, near-dups, and vocab overlap
poetry run python scripts/data/profile_documents.py \
  --file_path data/documents.jsonl \
  --out_dir outputs/reports/data_profile \
  --save 1 --keywords_top_k 20 --min_df 2 --max_features 20000 \
  --stopwords_top_n 200 --per_src_stopwords_top_n 50 \
  --near_dup 1 --near_dup_hamming 3 \
  --vocab_overlap 1

# Embedding health (sampled)
PYTHONPATH=src poetry run python scripts/data/profile_documents.py \
  --file_path data/documents.jsonl --save 1 \
  --embedding_health 1 --embedding_model <name> --embed_sample_size 2000

# Scientific term extraction (local LLM)
poetry run python scripts/data/extract_scientific_terms.py \
  --input_file outputs/reports/data_profile/latest/keywords_per_src.json \
  --model qwen2:7b
```

## Tracking template (per item)

Copy and paste for each new metric you implement:

- Title: <metric name>
- Status: [ ] not‑started | [ ] in‑progress | [ ] completed
- Owner: <name>
- Code touch‑points: <files/modules>
- Flags: <CLI args>
- Artifacts: <paths>
- Definition of Done:
  - [ ] <criterion 1>
  - [ ] <criterion 2>
- Run note: <command or invocation pattern>
- Validation: <quick checks or unit test>

## Resume checklist (today)

- [ ] Phase 1: per‑src tokens + long‑doc fraction
- [ ] Phase 1: vocab overlap matrix
- [ ] Phase 2: boilerplate metrics (+ thresholds into summary)
- [ ] Phase 3: compressibility ratio + sentence/paragraph stats → update chunking recommendations
- [ ] Phase 4: IDF extremes + BM25 len stats export
- [ ] Phase 5: embedding health (sampled) and clustering
- [ ] Phase 6: field examples + src→category mapping
- [ ] Phase 7: source glossary + confirm scientific term extraction output present

Notes
- Keep runs incremental; each added metric should be guarded by a flag and degrade gracefully if disabled.
- Prefer sparse or top‑K outputs for matrices to bound artifact size.
- Use the existing `latest` symlink to avoid hard‑coding timestamps.
