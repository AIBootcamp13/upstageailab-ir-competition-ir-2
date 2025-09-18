#!/usr/bin/env python3
"""
Test script to verify the debug functionality works correctly.
"""

import sys
import os
sys.path.append('/home/wb2x/workspace/information_retrieval_rag/src')

def test_imports():
    """Test that all required imports work."""
    try:
        from ir_core.retrieval.core import sparse_retrieve, dense_retrieve, hybrid_retrieve
        from ir_core.embeddings.core import encode_texts
        from ir_core.query_enhancement.confidence_logger import ConfidenceLogger
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_sparse():
    """Test sparse retrieval."""
    try:
        from ir_core.retrieval.core import sparse_retrieve
        results = sparse_retrieve("통학 버스의 가치", size=3)
        print(f"✅ Sparse retrieval: {len(results)} results")
        return len(results) > 0
    except Exception as e:
        print(f"❌ Sparse retrieval error: {e}")
        return False

def test_dense():
    """Test dense retrieval."""
    try:
        from ir_core.retrieval.core import dense_retrieve
        from ir_core.embeddings.core import encode_texts
        import numpy as np

        q_emb = encode_texts(["통학 버스의 가치"])[0]
        results = dense_retrieve(q_emb, size=3)
        print(f"✅ Dense retrieval: {len(results)} results")
        return len(results) > 0
    except Exception as e:
        print(f"❌ Dense retrieval error: {e}")
        return False

def test_confidence_logger():
    """Test confidence logger."""
    try:
        from ir_core.query_enhancement.confidence_logger import ConfidenceLogger
        logger = ConfidenceLogger(debug_mode=True)
        logger.log_confidence_score(
            technique='test',
            confidence=0.8,
            query='test query',
            retrieval_scores={'test_score': 0.8}
        )
        print("✅ Confidence logger working")
        return True
    except Exception as e:
        print(f"❌ Confidence logger error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing debug functionality...")
    print()

    success = True
    success &= test_imports()
    success &= test_sparse()
    success &= test_dense()
    success &= test_confidence_logger()

    print()
    if success:
        print("🎉 All tests passed! Debug functionality is ready.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")