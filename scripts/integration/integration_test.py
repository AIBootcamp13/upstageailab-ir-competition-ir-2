#!/usr/bin/env python3
"""
Integration Test for Profiling Insights in Retrieval Pipeline

Tests the enhanced retrieval system with profiling insights integration:
- Query expansion based on vocabulary overlap
- Dynamic chunking based on long document analysis
- Memory optimization based on token statistics
- Domain routing based on source clustering
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ir_core.retrieval.core import hybrid_retrieve
from ir_core.retrieval.insights_manager import (
    insights_manager,
    get_insights_status,
    refresh_insights
)
from ir_core.retrieval.chunking import get_chunking_config
from ir_core.config import settings


def test_insights_loading():
    """Test that profiling insights are loaded correctly."""
    print("=== Testing Profiling Insights Loading ===")

    status = get_insights_status()
    print(f"Insights loaded: {status['insights_loaded']}")
    print(f"Available insights: {status['available_insights']}")
    print(f"Long doc sources: {status['long_doc_sources']}")
    print(f"Vocab sources: {status['vocab_sources']}")
    print(f"Clusters count: {status['clusters_count']}")
    print(f"Stats sources: {status['stats_sources']}")

    return status['insights_loaded']


def test_chunking_recommendations():
    """Test dynamic chunking recommendations."""
    print("\n=== Testing Dynamic Chunking ===")

    # Test with a few sample sources
    test_sources = ["arxiv", "pubmed", "wikipedia", "springer"]

    for src in test_sources:
        config = get_chunking_config(src)
        print(f"{src}: chunk_size={config['chunk_size']}, "
              f"overlap_ratio={config['overlap_ratio']:.2f}, "
              f"source={config.get('source', 'unknown')}")


def test_enhanced_retrieval():
    """Test enhanced retrieval with profiling insights."""
    print("\n=== Testing Enhanced Retrieval ===")

    test_queries = [
        "machine learning algorithms",
        "quantum physics principles",
        "neural network architecture",
        "statistical analysis methods"
    ]

    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")

        # Test with profiling insights enabled
        print("With profiling insights:")
        try:
            results_with_insights = hybrid_retrieve(
                query=query,
                rerank_k=3,
                use_profiling_insights=True
            )
            print(f"  Retrieved {len(results_with_insights)} results")
            if results_with_insights:
                top_score = results_with_insights[0].get('score', 0)
                print(".3f")
        except Exception as e:
            print(f"  Error with insights: {e}")

        # Test with profiling insights disabled
        print("Without profiling insights:")
        try:
            results_without_insights = hybrid_retrieve(
                query=query,
                rerank_k=3,
                use_profiling_insights=False
            )
            print(f"  Retrieved {len(results_without_insights)} results")
            if results_without_insights:
                top_score = results_without_insights[0].get('score', 0)
                print(".3f")
        except Exception as e:
            print(f"  Error without insights: {e}")


def test_memory_optimization():
    """Test memory optimization recommendations."""
    print("\n=== Testing Memory Optimization ===")

    from ir_core.retrieval.insights_manager import get_memory_recommendation

    test_sources = ["arxiv", "pubmed", "wikipedia"]

    for src in test_sources:
        rec = get_memory_recommendation(src)
        print(f"{src}: batch_size={rec['batch_size']}, "
              f"avg_tokens={rec.get('avg_tokens', 'N/A')}, "
              f"recommendation='{rec['recommendation']}'")


def test_query_expansion():
    """Test query expansion based on vocabulary overlap."""
    print("\n=== Testing Query Expansion ===")

    from ir_core.retrieval.insights_manager import get_query_expansion_terms

    test_sources = ["arxiv", "pubmed"]

    for src in test_sources:
        terms = get_query_expansion_terms(src, top_k=3)
        print(f"{src}: expansion_terms={terms}")


def main():
    """Run all integration tests."""
    print("Starting Profiling Insights Integration Test")
    print("=" * 50)

    # Check if profiling data exists
    if not settings.PROFILE_REPORT_DIR:
        print("ERROR: PROFILE_REPORT_DIR not set in settings")
        return 1

    report_dir = Path(settings.PROFILE_REPORT_DIR)
    if not report_dir.exists():
        print(f"ERROR: Report directory does not exist: {report_dir}")
        return 1

    print(f"Using profiling data from: {report_dir}")

    # Run tests
    try:
        insights_loaded = test_insights_loading()
        if not insights_loaded:
            print("WARNING: Profiling insights not loaded - some tests may fail")

        test_chunking_recommendations()
        test_memory_optimization()
        test_query_expansion()
        test_enhanced_retrieval()

        print("\n" + "=" * 50)
        print("Integration test completed successfully!")
        return 0

    except Exception as e:
        print(f"\nERROR: Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())