# Debugging Plan: Improving Retrieval Relevance in RAG System

## Overview
The system has successfully fixed retrieval failures, but now faces low-quality retrieval relevance. Retrieved documents are semantically related but not the best matches for queries, leading to poor answer quality.

## Current Issues
- Retrieval scores are very low (0.02-0.03) indicating poor relevance
- Documents retrieved are topically distant from query intent
- Generation fails even when relevant documents are retrieved
- Hybrid retrieval (BM25 + dense) not optimally balanced

## Debugging Plan

### ✅ 0. Script Organization
- [x] Reorganize loose scripts into submodules (fine_tuning/, visualization/, maintenance/)
- [x] Update all references and CLI commands

### 1. Analyze Current Retrieval Performance
- [ ] Examine retrieval scores distribution in sample_submission_refactor_12.csv
- [ ] Analyze failure patterns: education spending → dementia/charts; school bus → genetics
- [ ] Check query preprocessing and embedding quality
- [ ] Validate BM25 vs dense retrieval contribution

### 2. Fine-tune Embedding Model
- [ ] Run `scripts/fine_tuning/fine_tune_embedding.py` on documents_ko_with_metadata.jsonl
- [ ] Use hypothetical questions for contrastive learning
- [ ] Monitor training metrics and memory usage
- [ ] Validate fine-tuned model performance on held-out data

### 3. Update Configuration for Fine-tuned Model
- [ ] Update settings.yaml to use fine-tuned model path
- [ ] Change EMBEDDING_PROVIDER to sentence_transformers
- [ ] Test basic retrieval functionality with new model

### 4. Tune BM25 Sparse Retrieval Parameters
- [ ] Analyze current boost values for different fields (keywords, summary, content)
- [ ] Experiment with boost combinations (e.g., keywords:2.0, summary:1.5, content:1.0)
- [ ] Test impact on retrieval relevance for different query types
- [ ] Find optimal balance between sparse and dense retrieval (ALPHA parameter)

### 5. Implement Re-ranking Component
- [ ] Add Cross-Encoder model (e.g., bge-reranker-large) for re-ranking top-50 candidates
- [ ] Integrate re-ranking into retrieval pipeline
- [ ] Tune re-ranking threshold and number of final candidates
- [ ] Evaluate improvement in top-1 and top-3 relevance

### 6. Improve Generation Context Utilization
- [ ] Analyze generation failures where relevant docs exist (e.g., contraception query)
- [ ] Refine prompts to explicitly extract from provided context
- [ ] Add context validation and answer extraction instructions
- [ ] Test prompt improvements on problematic queries

### 7. Test and Evaluate Improvements
- [ ] Run evaluation script on sample queries after each major change
- [ ] Track retrieval score improvements (target: 0.3+ average scores)
- [ ] Monitor answer quality improvements
- [ ] Compare before/after performance metrics

### 8. Re-index with Fine-tuned Embeddings
- [ ] Create new Elasticsearch index with fine-tuned model
- [ ] Re-index all documents with new embeddings
- [ ] Update INDEX_NAME in settings.yaml
- [ ] Validate indexing quality and search performance

## Success Criteria
- Retrieval scores improve from 0.02-0.03 to 0.3-0.6 range
- Fewer "insufficient information" answers
- Better semantic matching for complex queries
- Improved answer accuracy on evaluation set

## Timeline
- Phase 1 (Analysis): 1-2 days
- Phase 2 (Fine-tuning): 2-3 days
- Phase 3 (Parameter Tuning): 1-2 days
- Phase 4 (Re-ranking): 2-3 days
- Phase 5 (Generation): 1 day
- Phase 6 (Testing): Ongoing

## Notes
- Use poetry run python for all script executions
- Monitor memory usage during fine-tuning
- Backup current working configuration before changes
- Document all parameter changes and their impacts