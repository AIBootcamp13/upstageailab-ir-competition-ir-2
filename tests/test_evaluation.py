import pytest
from src.ir_core.evaluation.core import precision_at_k, mrr

# --- Tests for precision_at_k ---

def test_precision_at_k_perfect_score():
    """Tests if precision is 1.0 when all retrieved items are relevant."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc2", "doc3", "doc4"]
    assert precision_at_k(retrieved, relevant, k=3) == 1.0

def test_precision_at_k_partial_score():
    """Tests if precision is calculated correctly for a partial match."""
    retrieved = ["doc1", "doc5", "doc3"]
    relevant = ["doc1", "doc3", "doc4"]
    # 2 of the top 3 are relevant -> 2/3
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2/3)

def test_precision_at_k_zero_score():
    """Tests if precision is 0.0 when no retrieved items are relevant."""
    retrieved = ["doc5", "doc6", "doc7"]
    relevant = ["doc1", "doc2", "doc3"]
    assert precision_at_k(retrieved, relevant, k=3) == 0.0

def test_precision_at_k_k_greater_than_retrieved():
    """Tests behavior when k is larger than the number of retrieved items."""
    retrieved = ["doc1", "doc5"]
    relevant = ["doc1", "doc3"]
    # Only 1 item is relevant within the top 2 retrieved.
    # The score is divided by k=5, so 1/5 = 0.2
    assert precision_at_k(retrieved, relevant, k=5) == 0.2

def test_precision_at_k_empty_retrieved():
    """Tests that precision is 0.0 for empty retrieval list."""
    retrieved = []
    relevant = ["doc1", "doc2"]
    assert precision_at_k(retrieved, relevant, k=3) == 0.0

# --- Tests for mrr ---

def test_mrr_first_relevant():
    """Tests MRR when the first retrieved item is relevant."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc4"]
    # Relevant item is at rank 1, so MRR is 1/1 = 1.0
    assert mrr(retrieved, relevant) == 1.0

def test_mrr_third_relevant():
    """Tests MRR when the first relevant item is at rank 3."""
    retrieved = ["doc5", "doc6", "doc2", "doc1"]
    relevant = ["doc2", "doc4"]
    # First relevant item ('doc2') is at rank 3, so MRR is 1/3
    assert mrr(retrieved, relevant) == pytest.approx(1/3)

def test_mrr_no_relevant():
    """Tests that MRR is 0.0 if no relevant items are retrieved."""
    retrieved = ["doc5", "doc6", "doc7"]
    relevant = ["doc1", "doc2"]
    assert mrr(retrieved, relevant) == 0.0

def test_mrr_empty_retrieved():
    """Tests that MRR is 0.0 for an empty retrieval list."""
    retrieved = []
    relevant = ["doc1", "doc2"]
    assert mrr(retrieved, relevant) == 0.0
