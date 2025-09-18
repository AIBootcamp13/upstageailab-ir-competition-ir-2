#!/usr/bin/env python3
"""
Simple Retrieval Debug Test Script

This script tests sparse and dense retrieval with detailed debug output.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ir_core.retrieval.core import sparse_retrieve, dense_retrieve, hybrid_retrieve
from ir_core.embeddings.core import encode_texts
import numpy as np

def test_sparse_retrieval(query: str, size: int = 5):
    """Test sparse retrieval with detailed output."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING SPARSE RETRIEVAL")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    print(f"Size: {size}")
    print()

    # Perform sparse retrieval
    results = sparse_retrieve(query, size=size)

    print(f"📊 Retrieved {len(results)} documents:")
    print()

    # Process and display results with detailed information
    for i, hit in enumerate(results):
        source = hit.get('_source', {})
        doc_id = hit.get('_id', 'N/A')
        bm25_score = hit.get('_score', 0.0)

        print(f"{i+1}. ID: {doc_id}")
        print(f"   BM25 Score: {bm25_score:.4f}")
        content = source.get('content', '')
        print(f"   Content preview: {content[:150]}...")
        print()

def test_dense_retrieval(query: str, size: int = 5):
    """Test dense retrieval with detailed output."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING DENSE RETRIEVAL")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    print(f"Size: {size}")
    print()

    # Get query embedding
    print("🔄 Encoding query...")
    q_emb = encode_texts([query])[0]
    print(f"Query embedding shape: {q_emb.shape}")
    print()

    # Perform dense retrieval
    results = dense_retrieve(q_emb, size=size)

    print(f"📊 Retrieved {len(results)} documents:")
    print()

    # Process and display results with detailed information
    for i, hit in enumerate(results):
        source = hit.get('_source', {})
        doc_id = hit.get('_id', 'N/A')
        cosine_score = hit.get('_score', 0.0)

        print(f"{i+1}. ID: {doc_id}")
        print(f"   Cosine Score: {cosine_score:.4f}")
        content = source.get('content', '')
        print(f"   Content preview: {content[:150]}...")
        print()

def test_hybrid_retrieval(query: str, bm25_k: int = 10, rerank_k: int = 5, alpha: float = 0.4):
    """Test hybrid retrieval with detailed output."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING HYBRID RETRIEVAL")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    print(f"BM25 K: {bm25_k}, Rerank K: {rerank_k}, Alpha: {alpha}")
    print()

    # Perform hybrid retrieval
    results = hybrid_retrieve(query, bm25_k=bm25_k, rerank_k=rerank_k, alpha=alpha)

    print(f"📊 Retrieved {len(results)} documents:")
    print()

    # Process and display results with detailed information
    for i, result in enumerate(results):
        hit = result.get('hit', {})
        source = hit.get('_source', {})
        doc_id = hit.get('_id', 'N/A')
        final_score = result.get('score', 0.0)
        bm25_score = hit.get('_score', 0.0)
        cosine_score = result.get('cosine', 0.0)

        print(f"{i+1}. ID: {doc_id}")
        print(f"   Final Score: {final_score:.4f}")
        print(f"   BM25 Score: {bm25_score:.4f}")
        print(f"   Cosine Score: {cosine_score:.4f}")
        content = source.get('content', '')
        print(f"   Content preview: {content[:150]}...")
        print()

def main():
    """Main function to run all retrieval tests."""
    # Test query
    query = '통학 버스의 가치'

    print("🚀 RETRIEVAL DEBUG TEST SUITE")
    print("=" * 80)
    print(f"Testing with query: '{query}'")
    print("Debug mode: ENABLED")
    print()

    try:
        # Test sparse retrieval
        test_sparse_retrieval(query, size=5)

        # Test dense retrieval
        test_dense_retrieval(query, size=5)

        # Test hybrid retrieval
        test_hybrid_retrieval(query, bm25_k=10, rerank_k=5, alpha=0.4)

        print(f"\n{'='*80}")
        print("✅ All retrieval tests completed successfully!")
        print(f"{'='*80}")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()