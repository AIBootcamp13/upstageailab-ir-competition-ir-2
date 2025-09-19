# Testing Scripts

This directory contains scripts for testing and validating RAG system components.

## Scripts

### `test_polyglot_optimized.py`
Test script for Polyglot-Ko embedding provider with memory optimization.

**Usage:**
```bash
PYTHONPATH=src poetry run python scripts/testing/test_polyglot_optimized.py
```

### `test_techniques.py`
Test script for various retrieval and generation techniques.

**Usage:**
```bash
PYTHONPATH=src poetry run python scripts/testing/test_techniques.py
```

### `test_metadata.jsonl`
Test metadata file containing test cases and expected results.

## Features

- Embedding provider testing
- Memory optimization validation
- Retrieval technique comparison
- Component integration testing
- Performance benchmarking