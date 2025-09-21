#!/usr/-bin/env python3
"""A simple CLI to run a query against the retrieval system."""

import os
import sys
import json


def _add_src_to_path() -> None:
    """Add the src directory to the Python path."""
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def run(query: str, index_name: str = "test", rerank_k: int = 5):
    """
    Executes a hybrid retrieval query and prints the results.

    Args:
        query: The search query string.
        index_name: The Elasticsearch index to target. Defaults to "test".
        rerank_k: The final number of documents to return after re-ranking.
    """
    _add_src_to_path()
    # Lazy import after path is set
    from ir_core.retrieval import hybrid_retrieve

    print(f"Executing query: '{query}' on index '{index_name}'...")

    try:
        results = hybrid_retrieve(query, rerank_k=rerank_k)

        if not results:
            print("No results found.")
            return

        print(f"\nTop {len(results)} results:")
        print("-" * 20)
        for i, res in enumerate(results):
            # Results may be returned as raw ES hits or wrapped in {"hit": ..., "score": ...}
            if isinstance(res, dict) and "hit" in res:
                hit = res["hit"]
                score = res.get("score", hit.get("_score", 0.0))
            else:
                hit = res
                score = hit.get("score", hit.get("_score", 0.0))

            source = hit.get("_source", {}) if isinstance(hit, dict) else {}
            content = source.get("content", "N/A")
            # Truncate content for cleaner display
            display_content = (content[:150] + "...") if len(content) > 150 else content

            print(f"Rank {i+1}:")
            print(f"  ID: {hit.get('_id')}")
            try:
                print(f"  Score: {float(score):.4f}")
            except Exception:
                print(f"  Score: {score}")
            print(f"  Content: {display_content.strip()}")
            print("-" * 20)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print(
            "Please ensure Elasticsearch is running and the index '{index_name}' exists."
        )
        print("You can start services with: ./scripts/execution/run-local.sh start")
        print(
            "You can index data with: PYTHONPATH=src uv run python scripts/maintenance/reindex.py data/documents.jsonl"
        )


if __name__ == "__main__":
    # To enable a simple CLI interface
    try:
        import fire
    except ImportError:
        print("Fire library not found. Please install it with 'poetry install'.")
        sys.exit(1)
    fire.Fire(run)
