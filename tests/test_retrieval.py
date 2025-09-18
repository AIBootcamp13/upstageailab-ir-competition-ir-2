import pytest
import numpy as np
from ir_core.retrieval.reranker import RRFReRanker, AlphaBlendReRanker

# Define a set of mock documents that our fake functions will return
MOCK_BM25_HITS = [
    {"_id": "doc1", "_source": {"content": "text 1"}, "_score": 10.0}, # high bm25
    {"_id": "doc2", "_source": {"content": "text 2"}, "_score": 5.0},
    {"_id": "doc3", "_source": {"content": "text 3"}, "_score": 1.0}, # low bm25
]

# Define mock dense results that will result in predictable cosine similarities
# Query embedding is [1, 0]
# Doc embeddings:
# doc1 -> [0, 1] (cosine=0.0, low similarity)
# doc2 -> [0.707, 0.707] (cosine=0.707, medium similarity)
# doc3 -> [1, 0] (cosine=1.0, high similarity)
MOCK_DENSE_RESULTS = [
    {"_id": "doc1", "score": 0.0},   # low similarity
    {"_id": "doc2", "score": 0.707}, # medium similarity
    {"_id": "doc3", "score": 1.0},   # high similarity
]

def test_rrf_reranker_logic():
    """
    Tests the RRF reranking logic directly with mock data.
    """
    reranker = RRFReRanker(k=60)

    # Test RRF ranking
    results = reranker.rank(MOCK_BM25_HITS, MOCK_DENSE_RESULTS)

    # With RRF, all documents should be returned with rrf_score
    assert len(results) == 3
    result_ids = [r["_id"] for r in results]

    # RRF should rank based on combined reciprocal ranks
    # With current mock data, all docs have equal RRF scores (0.0323)
    # So order is stable insertion order: doc1, doc2, doc3
    assert result_ids == ["doc1", "doc2", "doc3"]

    # Check that RRF scores are present
    for r in results:
        assert "rrf_score" in r
        assert "score" in r
        assert r["score"] == r["rrf_score"]


def test_alpha_blend_reranker_logic():
    """
    Tests the alpha-based reranking logic directly with mock data.
    """
    # Case A: Alpha = 0.0 (purely semantic search)
    # Reranking should be based only on cosine similarity.
    # Expected order: doc3 (cos=1.0), doc2 (cos=0.707), doc1 (cos=0.0)
    reranker_alpha_0 = AlphaBlendReRanker(alpha=0.0)
    results_alpha_0 = reranker_alpha_0.rank(MOCK_BM25_HITS, MOCK_DENSE_RESULTS)
    result_ids_alpha_0 = [r["_id"] for r in results_alpha_0]
    assert result_ids_alpha_0 == ["doc3", "doc2", "doc1"]
    assert results_alpha_0[0]["score"] == pytest.approx(1.0) # score should equal cosine

    # Case B: Alpha = 1.0 (purely keyword-based ranking)
    # Reranking should be based only on BM25 scores.
    # Expected order: doc1 (bm25=10.0), doc2 (bm25=5.0), doc3 (bm25=1.0)
    reranker_alpha_1 = AlphaBlendReRanker(alpha=1.0)
    results_alpha_1 = reranker_alpha_1.rank(MOCK_BM25_HITS, MOCK_DENSE_RESULTS)
    result_ids_alpha_1 = [r["_id"] for r in results_alpha_1]
    assert result_ids_alpha_1 == ["doc1", "doc2", "doc3"]
    # score should equal normalized bm25: 10.0 / (10.0 + 1.0) = 0.909
    assert results_alpha_1[0]["score"] == pytest.approx(10.0 / 11.0)

    # Case C: Alpha = 0.5 (balanced hybrid search)
    # We need to calculate the expected scores:
    # doc1: 0.5 * (10/11) + 0.5 * 0.0   = 0.4545
    # doc2: 0.5 * (5/6)   + 0.5 * 0.707 = 0.4166 + 0.3535 = 0.7701
    # doc3: 0.5 * (1/2)   + 0.5 * 1.0   = 0.25 + 0.5 = 0.75
    # Expected order: doc2, doc3, doc1
    reranker_alpha_0_5 = AlphaBlendReRanker(alpha=0.5)
    results_alpha_0_5 = reranker_alpha_0_5.rank(MOCK_BM25_HITS, MOCK_DENSE_RESULTS)
    result_ids_alpha_0_5 = [r["_id"] for r in results_alpha_0_5]
    assert result_ids_alpha_0_5 == ["doc2", "doc3", "doc1"]

