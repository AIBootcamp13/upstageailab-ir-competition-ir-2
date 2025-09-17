# Indexing Scripts

This directory contains scripts for managing Elasticsearch indexes and configuration switching.

## Scripts

### `index_with_embeddings.py`
Script for indexing documents with embeddings into Elasticsearch.

**Usage:**
```bash
PYTHONPATH=src poetry run python scripts/indexing/index_with_embeddings.py <jsonl_file> <index_name>
```

**Parameters:**
- `jsonl_file`: Path to the JSONL file containing documents
- `index_name`: Name of the Elasticsearch index to create/update

### `switch_config.py`
Configuration switcher for Korean/English RAG setup. Helps switch between different embedding models and index configurations.

**Usage:**
```bash
# Switch to Korean configuration
PYTHONPATH=src poetry run python scripts/indexing/switch_config.py korean

# Switch to English configuration
PYTHONPATH=src poetry run python scripts/indexing/switch_config.py english

# Check current configuration
PYTHONPATH=src poetry run python scripts/indexing/switch_config.py status
```

## Features

- Automated index creation with embeddings
- Configuration management for different languages
- Batch processing for large document sets
- Index alias management
- Configuration validation