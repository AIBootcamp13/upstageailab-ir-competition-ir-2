#!/usr/bin/env python3
"""
List all scripts in the scripts/ directory with descriptions.

Usage: python scripts/list_scripts.py
"""

import os
import sys
from pathlib import Path

# Descriptions for each script (manually maintained)
SCRIPT_DESCRIPTIONS = {
    "execution/run_rag.py": "Runs the full RAG pipeline with Hydra configuration.",
    "execution/run_query.py": "CLI tool for executing hybrid retrieval queries.",
    "execution/run-local.sh": "Manages local Elasticsearch and Redis instances.",
    "evaluation/evaluate.py": "Runs evaluation on official datasets and logs to WandB.",
    "evaluation/validate_retrieval.py": "Validates retrieval performance with configurable parameters.",
    "evaluation/validate_domain_classification.py": "Checks domain classification accuracy.",
    "evaluation/smoke_test.py": "Python-based smoke tests for the system.",
    "evaluation/smoke-test.sh": "Shell wrapper for smoke testing with service management.",
    "test_huggingface_integration.py": "Tests HuggingFace model integration for retrieval and generation.",
    "data/analyze_data.py": "Analyzes document datasets for statistics (e.g., token counts).",
    "data/check_duplicates.py": "Detects duplicate entries in datasets.",
    "data/create_validation_set.py": "Generates validation datasets using LLM prompts.",
    "data/transform_submission.py": "Formats submission files.",
    "data/trim_submission.py": "Trims and cleans submission data.",
    "infra/start-elasticsearch.sh": "Downloads and starts local Elasticsearch.",
    "infra/start-redis.sh": "Downloads, builds, and starts local Redis.",
    "infra/cleanup-distros.sh": "Cleans up downloaded distributions.",
    "maintenance/reindex.py": "CLI for bulk reindexing JSONL files to Elasticsearch.",
    "maintenance/swap_alias.py": "Atomically swaps Elasticsearch aliases between indices.",
    "maintenance/parallel_example.py": "Example script for parallel processing.",
    "maintenance/demo_ollama_integration.py": "Demo for Ollama model integration.",
}


def list_scripts():
    scripts_dir = Path(__file__).parent
    print("Available scripts in scripts/ directory:\n")
    for subfolder in sorted(scripts_dir.iterdir()):
        if subfolder.is_dir() and not subfolder.name.startswith("__"):
            print(f"## {subfolder.name}/")
            for script in sorted(subfolder.iterdir()):
                if script.is_file():
                    rel_path = script.relative_to(scripts_dir)
                    desc = SCRIPT_DESCRIPTIONS.get(
                        str(rel_path), "No description available."
                    )
                    print(f"  - {rel_path}: {desc}")
            print()


if __name__ == "__main__":
    list_scripts()
