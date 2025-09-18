#!/usr/bin/env python3
"""
List all scripts in the scripts/ directory with descriptions.

Usage: poetry run python scripts/maintenance/list_scripts.py
"""

import os
import sys
from pathlib import Path

# Descriptions for each script (manually maintained)
SCRIPT_DESCRIPTIONS = {
    # Execution scripts
    "execution/run_rag.py": "Runs the full RAG pipeline with Hydra configuration.",
    "execution/run_query.py": "CLI tool for executing hybrid retrieval queries.",
    "execution/run-local.sh": "Manages local Elasticsearch and Redis instances.",

    # Evaluation scripts
    "evaluation/evaluate.py": "Runs evaluation on official datasets and logs to WandB.",
    "evaluation/validate_retrieval.py": "Validates retrieval performance with configurable parameters.",
    "evaluation/validate_domain_classification.py": "Checks domain classification accuracy.",
    "evaluation/smoke_test.py": "Python-based smoke tests for the system.",
    "evaluation/smoke-test.sh": "Shell wrapper for smoke testing with service management.",
    "evaluation/benchmark_enhancement.py": "Benchmarks query enhancement techniques.",
    "evaluation/create_balanced_validation.py": "Creates balanced validation datasets.",
    "evaluation/diagnose_validation_drop.py": "Diagnoses validation performance drops.",

    # Integration test scripts
    "integration/test_huggingface_integration.py": "Tests HuggingFace model integration for retrieval and generation.",
    "integration/test_qwen_integration.py": "Tests Qwen2 model integration with the RAG pipeline.",
    "integration/test_report_generator.py": "Generates and validates test reports for system evaluation.",
    "integration/test_visualizer.py": "Creates visualizations for test results and system performance.",

    # Data processing scripts
    "data/analyze_data.py": "Analyzes document datasets for statistics (e.g., token counts).",
    "data/profile_documents.py": "Profiles JSONL docs: unique 'src', counts, field presence, and length stats.",
    "data/check_duplicates.py": "Detects duplicate entries in datasets.",
    "data/create_validation_set.py": "Generates validation datasets using LLM prompts.",
    "data/transform_submission.py": "Formats submission files.",
    "data/trim_submission.py": "Trims and cleans submission data.",
    "data/extract_scientific_terms.py": "Extracts scientific terms from documents.",
    "data/generate_enhanced_validation.py": "Generates enhanced validation datasets.",
    "data/generate_metadata.py": "Generates metadata for datasets.",
    "data/persist_scientific_terms.py": "Persists scientific terms to storage.",
    "data/clean_scientific_terms.py": "Cleans and processes scientific terms.",
    "data/update_progress.py": "Updates progress tracking for data processing.",

    # Fine-tuning scripts
    "fine_tuning/fine_tune_retrieval.py": "Fine-tunes embedding and reranker models using enhanced validation data.",
    "fine_tuning/test_fine_tuned_models.py": "Tests and evaluates fine-tuned retrieval models.",

    # Infrastructure scripts
    "infra/start-elasticsearch.sh": "Downloads and starts local Elasticsearch.",
    "infra/start-redis.sh": "Downloads, builds, and starts local Redis.",
    "infra/cleanup-distros.sh": "Cleans up downloaded distributions.",

    # Maintenance scripts
    "maintenance/reindex.py": "CLI for bulk reindexing JSONL files to Elasticsearch.",
    "maintenance/swap_alias.py": "Atomically swaps Elasticsearch aliases between indices.",
    "maintenance/parallel_example.py": "Example script for parallel processing.",
    "maintenance/demo_ollama_integration.py": "Demo for Ollama model integration.",
    "maintenance/reindex_with_embeddings.py": "Reindexes documents with embeddings.",
    "maintenance/recompute.py": "Recomputes cached data and embeddings.",
    "maintenance/index_orchestrator.py": "Orchestrates complex indexing operations.",

    # CLI tools
    "cli/cli_menu.py": "Interactive CLI menu for common RAG operations.",

    # Debugging scripts
    "debugging/debug_performance.py": "Debugs RAG performance issues and bottlenecks.",

    # Indexing scripts
    "indexing/index_with_embeddings.py": "Indexes documents with embeddings to Elasticsearch.",
    "indexing/switch_config.py": "Switches between Korean/English RAG configurations.",

    # Testing scripts
    "testing/test_polyglot_optimized.py": "Tests Polyglot-Ko embedding provider with optimizations.",
    "testing/test_techniques.py": "Tests various retrieval and generation techniques.",

    # Translation scripts
    "translation/translate_validation.py": "Translates validation datasets between languages.",

    # Validation scripts
    "visualization/visualize_submissions.py": "Visualizes and analyzes submission results.",
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
