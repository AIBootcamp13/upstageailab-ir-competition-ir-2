#!/usr/bin/env python3
"""
Quick test to verify scientific_search returns preserved scores
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ir_core.tools.retrieval_tool import scientific_search

def test_scientific_search_scores():
    """Test that scientific_search preserves BM25 and dense scores"""

    # Test with a simple query
    query = "나무의 분류 방법"
    print(f"Testing scientific_search with query: {query}")

    results = scientific_search(query, top_k=3)

    print(f"\nRetrieved {len(results)} documents:")
    for i, result in enumerate(results, 1):
        print(f"\nDocument {i}:")
        print(f"  ID: {result.get('id', 'N/A')}")
        print(f"  Content preview: {result.get('content', '')[:100]}...")
        print(f"  Final Score: {result.get('score', 0):.4f}")
        print(f"  RRF Score: {result.get('rrf_score', 0):.4f}")
        print(f"  Sparse Score: {result.get('sparse_score', 0):.4f}")
        print(f"  Dense Score: {result.get('dense_score', 0):.4f}")
        print(f"  ES Score: {result.get('es_score', 0):.4f}")

        # Check if scores are non-zero
        has_scores = (
            result.get('sparse_score', 0) > 0 or
            result.get('dense_score', 0) > 0 or
            result.get('es_score', 0) > 0
        )
        print(f"  Has preserved scores: {has_scores}")

if __name__ == "__main__":
    test_scientific_search_scores()