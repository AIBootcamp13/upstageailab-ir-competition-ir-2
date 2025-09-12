# Scripts Directory

This directory contains executable scripts for various tasks in the information retrieval RAG project. Scripts are organized into subfolders by category for better maintainability and discoverability.

## Subfolders

### `execution/`
Core runtime scripts for running the system.
- `run_rag.py`: Runs the full RAG pipeline with Hydra configuration.
- `run_query.py`: CLI tool for executing hybrid retrieval queries.
- `run-local.sh`: Manages local Elasticsearch and Redis instances (start/stop/status).

### `evaluation/`
Scripts for testing, validating, and benchmarking the system.
- `evaluate.py`: Runs evaluation on official datasets and logs to WandB.
- `validate_retrieval.py`: Validates retrieval performance with configurable parameters.
- `validate_domain_classification.py`: Checks domain classification accuracy.
- `smoke_test.py`: Python-based smoke tests for the system.
- `smoke-test.sh`: Shell wrapper for smoke testing with service management.

### `data/`
Scripts for data analysis, processing, and transformation.
- `analyze_data.py`: Analyzes document datasets for statistics (e.g., token counts).
- `check_duplicates.py`: Detects duplicate entries in datasets.
- `create_validation_set.py`: Generates validation datasets using LLM prompts.
- `transform_submission.py`: Formats submission files.
- `trim_submission.py`: Trims and cleans submission data.

### `infra/`
Infrastructure setup and management scripts.
- `start-elasticsearch.sh`: Downloads and starts local Elasticsearch (Linux tarball).
- `start-redis.sh`: Downloads, builds, and starts local Redis.
- `cleanup-distros.sh`: Cleans up downloaded distributions.

### `maintenance/`
Utilities for upkeep, demos, and miscellaneous tasks.
- `reindex.py`: CLI for bulk reindexing JSONL files to Elasticsearch.
- `swap_alias.py`: Atomically swaps Elasticsearch aliases between indices.
- `parallel_example.py`: Example script for parallel processing.
- `demo_ollama_integration.py`: Demo for Ollama model integration.

## Usage Notes
- Most Python scripts use Hydra for configuration (see `conf/` directory).
- Ensure `PYTHONPATH=src` or run via Poetry for proper imports.
- Shell scripts are for Linux environments; check dependencies (e.g., `make` for Redis).
- Run scripts from the project root for correct relative paths.

## Contributing
- Add new scripts to the appropriate subfolder.
- Update this README and script docstrings with usage examples.
- Test scripts before committing changes.
