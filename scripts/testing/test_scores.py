#!/usr/bin/env python3
"""
Quick test script to verify score preservation in RRF
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ir_core.retrieval.core import hybrid_retrieve

def test_score_preservation():
    """Test that BM25 and dense scores are preserved in RRF results"""

    # Test with a simple query
    query = "나무의 분류 방법"
    print(f"Testing query: {query}")

    results = hybrid_retrieve(query, rerank_k=3)

    print(f"\nRetrieved {len(results)} documents:")
    for i, result in enumerate(results, 1):
        print(f"\nDocument {i}:")
        print(f"  ID: {result.get('_source', {}).get('docid', 'N/A')}")
        print(f"  Final Score: {result.get('score', 0):.4f}")
        print(f"  RRF Score: {result.get('rrf_score', 0):.4f}")
        print(f"  Sparse Score: {result.get('sparse_score', 0):.4f}")
        print(f"  Dense Score: {result.get('dense_score', 0):.4f}")
        print(f"  ES Score: {result.get('_score', 0):.4f}")

        # Check if scores are non-zero
        has_scores = (
            result.get('sparse_score', 0) > 0 or
            result.get('dense_score', 0) > 0 or
            result.get('_score', 0) > 0
        )
        print(f"  Has preserved scores: {has_scores}")

if __name__ == "__main__":
    test_score_preservation()