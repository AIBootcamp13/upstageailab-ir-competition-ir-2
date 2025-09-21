# Debugging Scripts

This directory contains scripts for debugging and analyzing RAG system performance.

## Scripts

### `debug_performance.py`
Comprehensive debugging script for RAG performance issues. Helps identify differences between working and failing runs by comparing configurations, query processing, and retrieval results.

**Usage:**
```bash
PYTHONPATH=src uv run python scripts/debugging/debug_performance.py
```

### `debug_submission_performance.md`
Documentation and analysis of submission performance debugging techniques.

## Features

- Query processing analysis
- Configuration comparison
- Retrieval result debugging
- Performance bottleneck identification
- Submission quality analysis