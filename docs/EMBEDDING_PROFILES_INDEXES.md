# Embedding Profiles and Indexes Documentation

## Overview

This document provides comprehensive information about available embedding profiles and Elasticsearch indexes in the RAG system. This documentation is designed for AI agents and maintenance purposes.

## Available Embedding Profiles

### 1. Korean Profile (`korean`)
- **Description**: Korean setup using KR-SBERT (768d)
- **Provider**: huggingface
- **Model**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- **Dimensions**: 768
- **Index Name**: `docs-ko-krsbert-d768-20250917`
- **Data Config**: ko
- **Translation**: Disabled
- **Use Case**: Korean-only documents and queries

### 2. English Profile (`english`)
- **Description**: English setup using all-MiniLM-L6-v2 (384d)
- **Provider**: huggingface
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Index Name**: `docs-en-minilm-d384-20250917`
- **Data Config**: en
- **Translation**: Enabled
- **Use Case**: English documents with translation support

### 3. Bilingual Profile (`bilingual`)
- **Description**: Bilingual setup using KR-SBERT (768d)
- **Provider**: huggingface
- **Model**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- **Dimensions**: 768
- **Index Name**: `docs-bilingual-krsbert-d768-20250917`
- **Data Config**: bilingual
- **Translation**: Disabled
- **Use Case**: Mixed Korean-English content

### 4. Solar Profile (`solar`)
- **Description**: Solar API setup (4096d)
- **Provider**: solar
- **Model**: `solar-embedding-1-large-passage`
- **Dimensions**: 4096
- **Index Name**: `docs-solar-d4096-20250917`
- **Data Config**: bilingual
- **Translation**: Disabled
- **Use Case**: High-dimensional embeddings via Solar API

### 5. Polyglot-1B Profile (`polyglot-1b`)
- **Description**: Polyglot-Ko-1.3B setup (2048d)
- **Provider**: polyglot
- **Model**: `EleutherAI/polyglot-ko-1.3b`
- **Dimensions**: 2048
- **Index Name**: `docs-ko-polyglot-1b-d2048-20250917`
- **Data Config**: ko
- **Translation**: Disabled
- **Quantization**: full
- **Batch Size**: 8
- **Max Threads**: 8
- **Use Case**: Korean documents with high-quality embeddings

### 6. Polyglot Profile (`polyglot`)
- **Description**: Polyglot-Ko-3.8B setup (3072d)
- **Provider**: polyglot
- **Model**: `EleutherAI/polyglot-ko-3.8b`
- **Dimensions**: 3072
- **Index Name**: `docs-ko-polyglot-3b-d3072-20250917`
- **Data Config**: ko
- **Translation**: Disabled
- **Quantization**: full
- **Batch Size**: 4
- **Max Threads**: 4
- **Use Case**: Korean documents with maximum quality embeddings

### 7. Polyglot-3B Profile (`polyglot-3b`)
- **Description**: Polyglot-Ko-3.8B setup (3072d) - Alias for `polyglot`
- **Provider**: polyglot
- **Model**: `EleutherAI/polyglot-ko-3.8b`
- **Dimensions**: 3072
- **Index Name**: `docs-ko-polyglot-3b-d3072-20250917`
- **Data Config**: ko
- **Translation**: Disabled
- **Quantization**: full
- **Batch Size**: 4
- **Max Threads**: 4
- **Use Case**: Korean documents with maximum quality embeddings

## Index Naming Convention

All indexes follow the professional naming convention:
```
docs-{language}-{model}-{dimensions}-{date}
```

### Components:
- **docs**: Fixed prefix indicating document index
- **language**: Language identifier (ko, en, bilingual)
- **model**: Model identifier (krsbert, minilm, polyglot-1b, polyglot-3b, solar)
- **dimensions**: Vector dimensions (d768, d384, d2048, d3072, d4096)
- **date**: Creation date in YYYYMMDD format (20250917)

### Examples:
- `docs-ko-krsbert-d768-20250917`
- `docs-en-minilm-d384-20250917`
- `docs-bilingual-krsbert-d768-20250917`
- `docs-solar-d4096-20250917`
- `docs-ko-polyglot-1b-d2048-20250917`
- `docs-ko-polyglot-3b-d3072-20250917`

## Configuration Switching

### Using Switch Config Script
```bash
# Switch to Korean configuration
PYTHONPATH=src poetry run python switch_config.py korean

# Switch to English configuration
PYTHONPATH=src poetry run python switch_config.py english

# Switch to Bilingual configuration
PYTHONPATH=src poetry run python switch_config.py bilingual

# Switch to Solar configuration
PYTHONPATH=src poetry run python switch_config.py solar

# Switch to Polyglot-1B configuration
PYTHONPATH=src poetry run python switch_config.py polyglot-1b

# Switch to Polyglot-3B configuration
PYTHONPATH=src poetry run python switch_config.py polyglot-3b
```

### Manual Configuration
Each profile can be activated by updating the main settings file:
```yaml
# In conf/settings.yaml
EMBEDDING_PROVIDER: "polyglot"  # or "huggingface" or "solar"
EMBEDDING_MODEL: "EleutherAI/polyglot-ko-1.3b"
EMBEDDING_DIMENSION: 2048
INDEX_NAME: "docs-ko-polyglot-1b-d2048-20250917"
```

## Data Files

### Available Data Files:
- `data/documents_ko.jsonl` - Korean scientific documents
- `data/documents_bilingual.jsonl` - Bilingual documents
- `data/documents_ko_with_metadata.jsonl` - Korean documents with metadata

### Index Creation:
```bash
# Create Korean index
PYTHONPATH=src poetry run python switch_config.py korean
PYTHONPATH=src poetry run python scripts/maintenance/reindex.py data/documents_ko.jsonl --index docs-ko-krsbert-d768-20250917

# Create English index
PYTHONPATH=src poetry run python switch_config.py english
PYTHONPATH=src poetry run python scripts/maintenance/reindex.py data/documents_bilingual.jsonl --index docs-en-minilm-d384-20250917

# Create Bilingual index
PYTHONPATH=src poetry run python switch_config.py bilingual
PYTHONPATH=src poetry run python scripts/maintenance/reindex.py data/documents_bilingual.jsonl --index docs-bilingual-krsbert-d768-20250917

# Create Polyglot-1B index
PYTHONPATH=src poetry run python switch_config.py polyglot-1b
PYTHONPATH=src poetry run python scripts/maintenance/reindex.py data/documents_ko.jsonl --index docs-ko-polyglot-1b-d2048-20250917
```

## Model Parameters

### Common Parameters:
- **ALPHA**: 0.4 (balance between BM25 and dense retrieval)
- **BM25_K**: 200 (documents to retrieve with BM25)
- **RERANK_K**: 10 (documents to rerank with dense retrieval)

### Provider-Specific Parameters:

#### HuggingFace Models:
- **EMBEDDING_BATCH_SIZE**: 32
- **EMBEDDING_MAX_LENGTH**: 512
- **EMBEDDING_DEVICE**: auto
- **EMBEDDING_USE_FAST_TOKENIZER**: true

#### Polyglot Models:
- **POLYGLOT_QUANTIZATION**: "full"
- **POLYGLOT_BATCH_SIZE**: 8 (1B) or 4 (3B)
- **POLYGLOT_MAX_THREADS**: 8 (1B) or 4 (3B)

#### Solar API:
- **UPSTAGE_API_KEY**: Set via environment variable
- **SOLAR_BASE_URL**: https://api.upstage.ai/v1/solar
- **SOLAR_MODEL**: solar-embedding-1-large

## Maintenance Operations

### Index Management:
```bash
# List all indexes
curl -X GET "localhost:9200/_cat/indices?v"

# Delete old indexes
curl -X DELETE "localhost:9200/old_index_name"

# Check index health
curl -X GET "localhost:9200/_cluster/health"
```

### Cache Management:
```bash
# Clear Redis cache
redis-cli FLUSHALL

# Check Redis memory usage
redis-cli INFO memory
```

### Configuration Validation:
```bash
# Check current configuration
PYTHONPATH=src poetry run python switch_config.py status

# Validate embedding dimensions match index
PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py debug=true debug_limit=2
```

## Troubleshooting

### Common Issues:

1. **Dimension Mismatch**: Ensure embedding dimensions match the index
2. **Index Not Found**: Verify index exists and is accessible
3. **Model Loading Failed**: Check model name and provider configuration
4. **Memory Issues**: Reduce batch size for large models
5. **Cache Issues**: Clear Redis cache if embeddings seem stale

### Performance Optimization:
- Use appropriate batch sizes for your hardware
- Enable GPU acceleration when available
- Monitor memory usage with large models
- Use quantization for memory-constrained environments

## API Reference

### Switch Config Script:
```python
# switch_config.py
def switch_profile(profile_name: str):
    """Switch to specified embedding profile"""
```

### Reindex Script:
```python
# scripts/maintenance/reindex.py
def reindex_data(data_file: str, index_name: str):
    """Reindex data into specified Elasticsearch index"""
```

### Validation Script:
```python
# scripts/evaluation/validate_retrieval.py
def run_validation(cfg: DictConfig):
    """Run retrieval validation with current configuration"""
```

This documentation should be updated whenever new embedding profiles or indexes are added to the system.</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/docs/EMBEDDING_PROFILES_INDEXES.md