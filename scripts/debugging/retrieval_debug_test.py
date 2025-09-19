#!/usr/bin/env python3
"""
Retrieval Debug Test Script

This script tests sparse and dense retrieval with detailed debug output
using the confidence logger to show comprehensive retrieval information.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ir_core.retrieval.core import sparse_retrieve, dense_retrieve, hybrid_retrieve
from ir_core.embeddings.core import encode_texts
from ir_core.query_enhancement.confidence_logger import ConfidenceLogger
import numpy as np

def test_sparse_retrieval_with_debug(query: str, size: int = 5):
    """Test sparse retrieval with detailed debug output."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING SPARSE RETRIEVAL")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    print(f"Size: {size}")
    print()

    # Initialize confidence logger with debug mode
    logger = ConfidenceLogger(debug_mode=True)

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
        print(f"   Content preview: {source.get('content', '')[:150]}...")
        print()

        # Log detailed retrieval information using confidence logger
        retrieval_scores = {
            'bm25_score': bm25_score,
            'rank': i + 1,
            'total_results': len(results)
        }

        logger.log_confidence_score(
            technique='sparse_retrieval',
            confidence=min(bm25_score / 100.0, 1.0),  # Normalize for confidence display
            query=query,
            reasoning=f"BM25 retrieval result #{i+1}",
            retrieval_scores=retrieval_scores,
            context={
                'doc_id': doc_id,
                'has_content': bool(source.get('content')),
                'content_length': len(source.get('content', ''))
            }
        )

def test_dense_retrieval_with_debug(query: str, size: int = 5):
    """Test dense retrieval with detailed debug output."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING DENSE RETRIEVAL")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    print(f"Size: {size}")
    print()

    # Initialize confidence logger with debug mode
    logger = ConfidenceLogger(debug_mode=True)

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
        print(f"   Content preview: {source.get('content', '')[:150]}...")
        print()

        # Log detailed retrieval information using confidence logger
        retrieval_scores = {
            'cosine_score': cosine_score,
            'rank': i + 1,
            'total_results': len(results),
            'embedding_norm': np.linalg.norm(q_emb)
        }

        logger.log_confidence_score(
            technique='dense_retrieval',
            confidence=min(abs(cosine_score), 1.0),  # Normalize for confidence display
            query=query,
            reasoning=f"Dense retrieval result #{i+1}",
            retrieval_scores=retrieval_scores,
            context={
                'doc_id': doc_id,
                'has_content': bool(source.get('content')),
                'content_length': len(source.get('content', ''))
            }
        )

def test_hybrid_retrieval_with_debug(query: str, bm25_k: int = 10, rerank_k: int = 5, alpha: float = 0.4):
    """Test hybrid retrieval with detailed debug output."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING HYBRID RETRIEVAL")
    print(f"{'='*80}")
    print(f"Query: '{query}'")
    print(f"BM25 K: {bm25_k}, Rerank K: {rerank_k}, Alpha: {alpha}")
    print()

    # Initialize confidence logger with debug mode
    logger = ConfidenceLogger(debug_mode=True)

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
        print(f"   Content preview: {source.get('content', '')[:150]}...")
        print()

        # Log detailed retrieval information using confidence logger
        retrieval_scores = {
            'final_score': final_score,
            'bm25_score': bm25_score,
            'cosine_score': cosine_score,
            'rank': i + 1,
            'total_results': len(results),
            'alpha': alpha
        }

        logger.log_confidence_score(
            technique='hybrid_retrieval',
            confidence=min(final_score, 1.0),  # Normalize for confidence display
            query=query,
            reasoning=f"Hybrid retrieval result #{i+1} (BM25 + Dense)",
            retrieval_scores=retrieval_scores,
            context={
                'doc_id': doc_id,
                'has_content': bool(source.get('content')),
                'content_length': len(source.get('content', ''))
            }
        )

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
        test_sparse_retrieval_with_debug(query, size=5)

        # Test dense retrieval
        test_dense_retrieval_with_debug(query, size=5)

        # Test hybrid retrieval
        test_hybrid_retrieval_with_debug(query, bm25_k=10, rerank_k=5, alpha=0.4)

        print(f"\n{'='*80}")
        print("✅ All retrieval tests completed successfully!")
        print(f"{'='*80}")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()