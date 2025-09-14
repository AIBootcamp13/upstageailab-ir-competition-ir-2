# Profiling the documents.jsonl corpus

- Added profile_documents.py to profile a JSONL corpus: unique src values, counts per src, field presence, and basic length stats; writes JSON reports to `outputs/reports/data_profile/<timestamp>/`.
- Captured 63 unique src values from the combined ARC and MMLU Korean datasets.

Artifacts:
- Report dir: 20250914_091111
  - unique_src.json, src_counts.json, `field_presence.json`, `field_length_stats.json`, summary.json

## unique 'src' datasets (63 total)

- ko_ai2_arc__ARC_Challenge__test
- ko_ai2_arc__ARC_Challenge__train
- ko_ai2_arc__ARC_Challenge__validation
- ko_mmlu__anatomy__test
- ko_mmlu__anatomy__train
- ko_mmlu__anatomy__validation
- ko_mmlu__astronomy__test
- ko_mmlu__astronomy__train
- ko_mmlu__astronomy__validation
- ko_mmlu__college_biology__test
- ko_mmlu__college_biology__train
- ko_mmlu__college_biology__validation
- ko_mmlu__college_chemistry__test
- ko_mmlu__college_chemistry__train
- ko_mmlu__college_chemistry__validation
- ko_mmlu__college_computer_science__test
- ko_mmlu__college_computer_science__train
- ko_mmlu__college_computer_science__validation
- ko_mmlu__college_medicine__test
- ko_mmlu__college_medicine__train
- ko_mmlu__college_medicine__validation
- ko_mmlu__college_physics__test
- ko_mmlu__college_physics__train
- ko_mmlu__college_physics__validation
- ko_mmlu__computer_security__test
- ko_mmlu__computer_security__train
- ko_mmlu__computer_security__validation
- ko_mmlu__conceptual_physics__test
- ko_mmlu__conceptual_physics__train
- ko_mmlu__conceptual_physics__validation
- ko_mmlu__electrical_engineering__test
- ko_mmlu__electrical_engineering__train
- ko_mmlu__electrical_engineering__validation
- ko_mmlu__global_facts__test
- ko_mmlu__global_facts__train
- ko_mmlu__global_facts__validation
- ko_mmlu__high_school_biology__test
- ko_mmlu__high_school_biology__train
- ko_mmlu__high_school_biology__validation
- ko_mmlu__high_school_chemistry__test
- ko_mmlu__high_school_chemistry__train
- ko_mmlu__high_school_chemistry__validation
- ko_mmlu__high_school_computer_science__test
- ko_mmlu__high_school_computer_science__train
- ko_mmlu__high_school_computer_science__validation
- ko_mmlu__high_school_physics__test
- ko_mmlu__high_school_physics__train
- ko_mmlu__high_school_physics__validation
- ko_mmlu__human_aging__test
- ko_mmlu__human_aging__train
- ko_mmlu__human_aging__validation
- ko_mmlu__human_sexuality__test
- ko_mmlu__human_sexuality__train
- ko_mmlu__human_sexuality__validation
- ko_mmlu__medical_genetics__test
- ko_mmlu__medical_genetics__train
- ko_mmlu__medical_genetics__validation
- ko_mmlu__nutrition__test
- ko_mmlu__nutrition__train
- ko_mmlu__nutrition__validation
- ko_mmlu__virology__test
- ko_mmlu__virology__train
- ko_mmlu__virology__validation

Also saved to unique_src.json.

## useful extras to collect next

To improve MAP and reduce LLM compute, I suggest capturing these reusable signals (incrementally; all can be appended to the profiler):

- Per-src document stats
  - Counts per src (already saved), plus per-src length stats (tokens/words/chars) and fraction of long docs; use to tune chunk size/overlap per domain.
  - Vocabulary overlap between src groups; helps decide cross-domain retrieval boosts.

- Quality and dedup metrics
  - Exact/near-duplicate detection (e.g., MinHash/SimHash Jaccard > 0.9); drop or downweight near-dups to reduce index bloat and reranking noise.
  - Boilerplate/low-content ratio (non-alnum ratio, repeated lines, short unique token ratio); flag and downweight.

- Chunking guidance
  - Compressibility ratio (gzip size / text size) as a proxy for redundancy; informs chunk overlap.
  - Sentence/paragraph count to support semantic chunking.

- Retrieval diagnostics (feeds MAP)
  - Per-src BM25-friendly features: avg doc length, variance; helps tune b, k1 per domain.
  - IDF extremes/top rare terms per src to design stopword lists and query rewriting rules.
  - Hybrid tuning hooks: store per-src priors so you can boost/penalize sources likely to answer specific query types.

- Embedding-level signals
  - Embedding norm and cosine similarity distributions; flag outliers and bad texts.
  - Simple clustering per src (k-means inertia/silhouette) to spot heterogeneous sources that might need sub-clusters.

- Field/schema profiling
  - Field presence distribution (already saved) and example values; helps standardize fields (title/abstract/content).
  - Map src to a category (e.g., ARC vs MMLU, subject), to enable category-based retrieval boosts and evaluation slices.

- Query-focused cacheables
  - Precompute per-src “top keywords” (TF-IDF) and store as short descriptors for LLM context selection without sending full docs.
  - Build a small “source glossary” that the LLM can use to understand datasets without reading many samples.

## how to rerun (optional)

- From repo root:
```bash
poetry run python scripts/data/profile_documents.py --file_path data/documents.jsonl
```

This generates a timestamped report under data_profile and prints a concise summary including the unique src list.

## completion summary

- Unique src list extracted (63 total) and saved.
- Reusable profiling script added and registered.
- Reports written with counts, field presence, and length stats.
- Provided concrete next metrics to collect to guide chunking, hybrid tuning, and quality filtering for better MAP and lower LLM cost.

Made changes.