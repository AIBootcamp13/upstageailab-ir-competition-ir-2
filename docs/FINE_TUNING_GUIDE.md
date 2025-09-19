# Fine-tuning Retrieval Models

This guide explains how to fine-tune retrieval models using your enhanced validation data.

## Overview

Your RAG system uses two main retrieval components:
1. **Embedding Model** (Polyglot-Ko): Converts queries and documents to vectors
2. **Reranker Model** (Optional): Re-ranks initial retrieval results

Fine-tuning these models can significantly improve retrieval quality.

## Prerequisites

1. **Enhanced Validation Data**: Run the generation script first:
   ```bash
   PYTHONPATH=src poetry run python scripts/data/generate_enhanced_validation.py --config-path ../../conf --config-name settings
   ```

2. **Required Libraries**: Already installed via Poetry
   - sentence-transformers
   - transformers
   - torch

## Fine-tuning Embedding Model

### Configuration

Edit `conf/settings.yaml`:
```yaml
fine_tune:
  model_type: embedding
  base_model: EleutherAI/polyglot-ko-1.3b
  output_dir: models/fine_tuned_embedding
  epochs: 3
  batch_size: 16
  val_split: 0.1
```

### Run Fine-tuning

```bash
PYTHONPATH=src poetry run python scripts/fine_tune_retrieval.py --config-path ../../conf --config-name settings
```

### What happens:
- Uses contrastive learning on query-positive/negative document pairs
- Positive pairs: query + ideal context
- Negative pairs: query + hard negative context
- Trains for improved semantic similarity matching

## Fine-tuning Reranker Model

### Configuration

Edit `conf/settings.yaml`:
```yaml
fine_tune:
  model_type: reranker
  base_model: bert-base-multilingual-cased  # or xlm-roberta-base
  output_dir: models/fine_tuned_reranker
  epochs: 3
  batch_size: 8  # Smaller batch size for reranker
  val_split: 0.1
```

### Run Fine-tuning

```bash
PYTHONPATH=src poetry run python scripts/fine_tune_retrieval.py --config-path ../../conf --config-name settings
```

### What happens:
- Uses cross-encoder approach (query + document as single input)
- Binary classification: relevant vs. not relevant
- Better at understanding query-document relationships

## Testing Fine-tuned Models

### Configuration

Edit `conf/settings.yaml`:
```yaml
test_fine_tuned:
  embedding_model_path: models/fine_tuned_embedding
  reranker_model_path: models/fine_tuned_reranker
```

### Run Evaluation

```bash
PYTHONPATH=src poetry run python scripts/test_fine_tuned_models.py --config-path ../../conf --config-name settings
```

### Metrics Reported:
- **Recall@K**: Fraction of ideal contexts found in top-K results
- **Precision@K**: Fraction of top-K results that are ideal contexts

## Integration into Production

After fine-tuning, you can:

1. **Update Configuration**: Modify `conf/settings.yaml` to use fine-tuned models
2. **Update Code**: Modify retrieval code to load fine-tuned models
3. **Re-index**: If using fine-tuned embeddings, re-index your documents

### Example Configuration Changes:

```yaml
# For embedding model
EMBEDDING_MODEL: /path/to/fine_tuned_embedding

# For reranker (add to retrieval pipeline)
reranker_model_path: /path/to/fine_tuned_reranker
```

## Tips for Better Results

1. **Data Quality**: Ensure your enhanced validation data has good ideal contexts and hard negatives
2. **Model Selection**:
   - For Korean: Use Polyglot-Ko or KoBERT-based models
   - For multilingual: Use XLM-RoBERTa or mBERT
3. **Training Parameters**:
   - Start with 2-3 epochs
   - Use smaller batch sizes for rerankers (8-16)
   - Monitor validation loss to avoid overfitting
4. **Evaluation**: Always test on held-out data to measure improvement

## Expected Improvements

- **Embedding Fine-tuning**: 5-15% improvement in retrieval recall
- **Reranker Fine-tuning**: 10-25% improvement in top-K accuracy
- **Combined**: 15-30% overall improvement

## Troubleshooting

- **Out of Memory**: Reduce batch size or use gradient checkpointing
- **Poor Performance**: Check data quality and increase training epochs
- **Convergence Issues**: Adjust learning rate or use different optimizer</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/docs/FINE_TUNING_GUIDE.md