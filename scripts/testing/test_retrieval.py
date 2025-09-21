#!/usr/bin/env python3
"""
Simple test script to verify the improved BM25 retrieval
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set environment variables
os.environ['RAG_EVALUATION_MODE'] = 'true'

from ir_core.retrieval.core import sparse_retrieve

def test_retrieval():
    query = "각 나라 공교육 지출 현황"
    print(f"Testing query: {query}")
    print("=" * 50)

    results = sparse_retrieve(query, size=5)

    for i, hit in enumerate(results, 1):
        score = hit['_score']
        content = hit['_source']['content'][:200] + "..."
        print(f"{i}. Score: {score:.2f}")
        print(f"   Content: {content}")
        print()

    # Check if education spending document is in top results
    has_education_doc = any("공공 교육 지출" in hit['_source']['content'] for hit in results)
    print(f"Education spending document found in top 5: {has_education_doc}")

if __name__ == "__main__":
    test_retrieval()