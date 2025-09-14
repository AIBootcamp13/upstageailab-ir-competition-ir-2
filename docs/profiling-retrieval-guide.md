# Profiling-Enhanced Retrieval Configuration Guide

This document explains the granular settings added to control data profiling enhancements in the retrieval pipeline.

## Available Settings

All settings can be toggled in `conf/settings.yaml` or via environment variables:

```yaml
# --- Retrieval tuning from profiling ---
USE_SRC_BOOSTS: true              # Boost queries using per-source TF-IDF keywords
USE_STOPWORD_FILTERING: false     # Strip global stopwords from queries
USE_DUPLICATE_FILTERING: false    # Filter exact duplicates during reranking
USE_NEAR_DUP_PENALTY: false       # Penalize near-duplicates (experimental)
PROFILE_REPORT_DIR: "outputs/reports/data_profile/latest"
```

## What Each Setting Does

### USE_SRC_BOOSTS (Production Ready)
- **Effect**: Adds soft boosts to BM25 queries using per-source TF-IDF keywords
- **Artifacts**: Uses `keywords_per_src.json` 
- **Impact**: Gentle bias toward documents containing source-specific terminology
- **Risk**: Low - preserves main BM25 ranking, adds "should" clauses
- **When to use**: Always safe to enable for domain-aware boosting

### USE_STOPWORD_FILTERING (Experimental)
- **Effect**: Removes high-frequency terms from queries before search
- **Artifacts**: Uses `stopwords_global.json`
- **Impact**: May improve precision by reducing noise from common terms
- **Risk**: Medium - could hurt recall if important terms are filtered
- **When to use**: Test on eval set first; disable if recall drops

### USE_DUPLICATE_FILTERING (Production Ready)
- **Effect**: Filters exact duplicates during hybrid reranking
- **Artifacts**: Uses `duplicates.json` 
- **Impact**: Prevents multiple copies of same content in results
- **Risk**: Low - only removes confirmed exact duplicates
- **When to use**: Safe to enable; improves result diversity

### USE_NEAR_DUP_PENALTY (Experimental)
- **Effect**: Applies score penalty to near-duplicate clusters
- **Artifacts**: Uses `near_duplicates.json`
- **Impact**: Reduces similar content in results
- **Risk**: Medium - penalty logic needs tuning
- **When to use**: Experimental; needs doc-id mapping refinement

## Recommended Workflow

### Phase 1: Safe Enhancements (Enable Now)
```yaml
USE_SRC_BOOSTS: true
USE_DUPLICATE_FILTERING: true
USE_STOPWORD_FILTERING: false
USE_NEAR_DUP_PENALTY: false
```

### Phase 2: Experimental Features (Test on Eval Set)
```yaml
USE_STOPWORD_FILTERING: true    # Test impact on MAP@10
USE_NEAR_DUP_PENALTY: true      # Monitor for over-filtering
```

## Performance Expectations

### Immediate Improvements (Phase 1)
- **Source-aware boosting**: 2-5% MAP improvement from domain relevance
- **Duplicate filtering**: Better result diversity, slight MAP gain

### Potential Improvements (Phase 2)  
- **Stopword filtering**: 0-10% MAP change (highly query-dependent)
- **Near-dup penalty**: Variable impact depending on corpus redundancy

## No Reindexing Required

All enhancements work at query/retrieval time:
- Profiling artifacts are read from disk
- ES index remains unchanged
- Settings can be toggled without restart

## Chunking Guidance (Separate)

The `chunking.py` module provides per-source chunk size recommendations:
- Use during document preprocessing/indexing
- Based on `per_src_length_stats.json`
- Requires reindexing to take effect

## Monitoring

Watch for these metrics:
- **MAP@10**: Overall retrieval quality
- **Recall@100**: Ensure filtering doesn't hurt coverage  
- **Result diversity**: Fewer near-duplicates in top results
- **Query latency**: Minimal impact expected (< 5ms overhead)

## Troubleshooting

### No artifacts found
- Check `PROFILE_REPORT_DIR` points to valid profiling output
- Run profiling script: `poetry run python scripts/data/profile_documents.py`

### Unexpected MAP drop
- Disable experimental features one by one
- Check `stopwords_global.json` for over-aggressive filtering
- Verify duplicate filtering isn't removing valid results

### Import errors
- Ensure all new modules are in `src/ir_core/retrieval/`
- Run smoke test: `PYTHONPATH=src python -c "from ir_core.retrieval.core import hybrid_retrieve"`