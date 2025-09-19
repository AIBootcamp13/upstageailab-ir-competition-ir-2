# Phase 1 Implementation Plan: Per-Source Document Stats (Depth)
## Performance Improvement Roadmap for Scientific Documents IR System

**Date:** September 15, 2025
**Current Status:** Infrastructure ready, starting Phase 1 implementation
**Language:** Korean (native) for input/output, English for technical implementation

---

## Executive Summary

Based on current performance analysis:
- **Total Queries:** 106
- **High Performance:** 46 (43%)
- **Failed Queries:** 50 (47%)
- **Main Issues:** Infrastructure errors (29), ambiguous queries (7), false negatives (7)

Phase 1 focuses on **per-source document statistics** to understand data distribution patterns and inform chunking/indexing strategies for better retrieval performance.

---

## Current Performance Baseline

### Error Analysis
```
successful: 56 (53%)
failed: 50 (47%)
├── infrastructure_error: 29 (27%)
├── ambiguous_query: 7 (7%)
├── false_negative: 7 (7%)
├── out_of_domain: 3 (3%)
└── complex_multi_concept: 4 (4%)
```

### Performance Segmentation
```
high_performance: 46 (43%)
medium_performance: 7 (7%)
low_performance: 3 (3%)
failed: 50 (47%)
```

---

## Phase 1 Objectives

### Primary Goals
1. **Token Statistics:** Compute per-source token/word/char stats using project tokenizer
2. **Long Document Analysis:** Identify fraction of long documents (top 10% by word count)
3. **Vocabulary Overlap:** Build overlap matrix between source groups for domain understanding
4. **Source Clustering:** Optional clustering by vocabulary similarity

### Expected Outcomes
- `per_src_length_stats.json` extended with token stats and long-doc fractions
- `vocab_overlap_matrix.json` for understanding cross-domain relationships
- `src_clusters_by_vocab.json` for domain grouping insights
- Updated `summary.json` with actionable recommendations

---

## Implementation Details

### 1. Token Statistics Extension
```python
# Add to profile_documents.py
--tokenizer option (default: whitespace, optional: project tokenizer)
- Compute token counts per document using ir_core.embeddings.core.load_model()
- Aggregate per-source: mean/median/P90/P95 for tokens
- Extend per_src_length_stats.json with token metrics
```

### 2. Long Document Fraction Analysis
```python
# Configuration
--long_doc_pct 0.9  # Threshold for "long documents" (default: 90th percentile)
--long_doc_field "content_words"  # Field to analyze (words/chars/tokens)

# Output: per_src_length_stats.json
{
  "src_name": {
    "content_words": {...},
    "long_doc_fraction": 0.12,  # 12% of docs are in top 10%
    "long_doc_threshold": 1500   # Word count threshold
  }
}
```

### 3. Vocabulary Overlap Matrix
```python
# Build per-source vocabulary sets from TF-IDF terms
--vocab_overlap 1  # Enable vocabulary overlap computation
--vocab_sample_size 10000  # Limit vocab size for performance

# Compute Jaccard similarity between src pairs
# Output: vocab_overlap_matrix.json
{
  "src_A": {
    "src_B": 0.15,  # 15% vocabulary overlap
    "src_C": 0.08
  }
}
```

### 4. Source Clustering (Optional)
```python
# Agglomerative clustering on Jaccard distance matrix
--vocab_clustering 1  # Enable clustering
--max_clusters 5  # Maximum number of clusters

# Output: src_clusters_by_vocab.json
{
  "clusters": [
    {"id": 0, "sources": ["biology_src1", "chemistry_src2"], "size": 2},
    {"id": 1, "sources": ["physics_src1"], "size": 1}
  ]
}
```

---

## Integration with Query Enhancement

### Query Rewriting Opportunities
Based on current error analysis, integrate profiling insights with query enhancement:

1. **Ambiguous Query Handling (7 cases)**
   - Use vocabulary overlap data to identify multi-domain queries
   - Rewrite queries to target specific source clusters
   - Example: "새로운 땅이 생겨나는 메커니즘" → "판 구조론과 화산 활동으로 새로운 지각 형성"

2. **False Negative Reduction (7 cases)**
   - Leverage long document analysis for chunking optimization
   - Adjust retrieval parameters based on per-source token distributions
   - Use vocabulary clustering for cross-domain search expansion

3. **Complex Multi-Concept Queries (4 cases)**
   - Implement query decomposition based on source clustering
   - Use vocabulary overlap to identify related concepts across domains

### Performance Optimization Strategies

#### Infrastructure Error Mitigation (29 cases)
- **Memory Management:** Use profiling insights to optimize chunk sizes
- **Timeout Handling:** Set conservative timeouts based on token statistics
- **Resource Allocation:** Allocate resources based on document length distributions

#### Retrieval Enhancement Pipeline
```
Query Input → Classification → Rewriting → Multi-Source Retrieval → Re-ranking
                     ↓
            Profiling Insights Integration
                     ↓
        Per-Source Optimization Parameters
```

---

## Technical Implementation Plan

### File Modifications

#### `scripts/data/profile_documents.py`
```python
# New CLI options
--tokenizer "project" | "whitespace"
--long_doc_pct 0.9
--vocab_overlap 1
--vocab_clustering 1
--max_clusters 5

# New data structures
per_src_tokens: defaultdict[str, List[int]]
vocab_sets: Dict[str, Set[str]]
overlap_matrix: Dict[str, Dict[str, float]]
```

#### `src/ir_core/retrieval/chunking.py`
```python
# Consume profiling outputs for dynamic chunking
def get_chunking_params(src: str) -> Dict[str, Any]:
    """Get optimized chunking parameters based on profiling data"""
    # Load per_src_length_stats.json
    # Return chunk_size, overlap based on token distributions
```

### Output Artifacts

#### Extended `per_src_length_stats.json`
```json
{
  "biology_src": {
    "content_chars": {"mean": 2450, "p95": 5200},
    "content_words": {"mean": 380, "p95": 820},
    "content_tokens": {"mean": 320, "p95": 680},
    "long_doc_fraction": 0.08,
    "long_doc_threshold": 1200
  }
}
```

#### New `vocab_overlap_matrix.json`
```json
{
  "biology_src": {
    "chemistry_src": 0.12,
    "physics_src": 0.05
  },
  "chemistry_src": {
    "physics_src": 0.08
  }
}
```

#### New `src_clusters_by_vocab.json`
```json
{
  "clusters": [
    {
      "id": 0,
      "sources": ["bio1", "chem1", "phys1"],
      "centroid_terms": ["세포", "분자", "에너지"],
      "size": 3
    }
  ]
}
```

---

## Testing and Validation Strategy

### Unit Testing
```bash
# Test token statistics computation
PYTHONPATH=src poetry run python -m pytest tests/test_profiling_token_stats.py

# Test vocabulary overlap calculation
PYTHONPATH=src poetry run python -m pytest tests/test_vocab_overlap.py
```

### Integration Testing
```bash
# Full profiling run with all Phase 1 features
poetry run python scripts/data/profile_documents.py \
  --file_path data/documents.jsonl \
  --out_dir outputs/reports/data_profile \
  --save 1 \
  --tokenizer project \
  --long_doc_pct 0.9 \
  --vocab_overlap 1 \
  --vocab_clustering 1

# Validate outputs
PYTHONPATH=src poetry run python scripts/evaluation/validate_profiling_outputs.py
```

### Performance Benchmarks
- **Target:** <1.5x baseline runtime with new features enabled
- **Memory:** Stable memory usage during processing
- **Accuracy:** Validate token counts against known tokenizer outputs

---

## Success Criteria

### Functional Requirements
- [ ] All new CLI options work without errors
- [ ] Output JSON files contain expected data structures
- [ ] Token statistics match project tokenizer behavior
- [ ] Vocabulary overlap calculations are reproducible

### Performance Requirements
- [ ] Processing time < 150% of baseline
- [ ] Memory usage remains stable
- [ ] No crashes on edge cases (empty docs, missing fields)

### Quality Requirements
- [ ] Comprehensive error handling and logging
- [ ] Clear documentation in docstrings
- [ ] Integration with existing profiling infrastructure

---

## Risk Mitigation

### Technical Risks
1. **Tokenizer Performance:** Lazy loading with fallback to whitespace
2. **Memory Issues:** Streaming processing for large vocabularies
3. **Numerical Stability:** Safe division and NaN handling in overlap calculations

### Data Quality Risks
1. **Missing Content:** Graceful handling of documents without content field
2. **Encoding Issues:** UTF-8 handling for Korean text
3. **Sparse Data:** Minimum document thresholds for reliable statistics

---

## Next Steps After Phase 1

### Phase 2: Quality and Deduplication
- Boilerplate detection metrics
- Enhanced near-duplicate detection with MinHash
- Quality scoring per source

### Phase 3: Chunking Optimization
- Compressibility analysis
- Sentence/paragraph boundary detection
- Dynamic chunk size recommendations

### Integration with Query Enhancement
- Use profiling insights for query routing
- Implement source-aware retrieval boosting
- Add multi-domain query decomposition

---

## Timeline and Milestones

**Week 1:** Core token statistics implementation
**Week 2:** Long document analysis and vocabulary overlap
**Week 3:** Source clustering and integration testing
**Week 4:** Performance optimization and documentation

---

## Dependencies and Prerequisites

### Required Packages
- numpy, scipy, scikit-learn (already available)
- ir_core.embeddings.core.load_model() (project tokenizer)
- Existing profiling infrastructure

### Data Requirements
- `data/documents.jsonl` with `content`, `src` fields
- Sufficient documents per source for reliable statistics (>10 docs/src)

### Environment Setup
```bash
# Ensure Poetry environment is active
poetry install
poetry run python --version  # Should be 3.10+

# Test infrastructure
./scripts/execution/run-local.sh status
```

---

*This plan integrates profiling insights with query enhancement strategies to systematically improve retrieval performance while maintaining Korean language support for user interactions.*