import pytest
import numpy as np
from ir_core import retrieval
from ir_core import embeddings

# Define a set of mock documents that our fake functions will return
MOCK_BM25_HITS = [
    {"_id": "doc1", "_source": {"content": "text 1"}, "_score": 10.0}, # high bm25
    {"_id": "doc2", "_source": {"content": "text 2"}, "_score": 5.0},
    {"_id": "doc3", "_source": {"content": "text 3"}, "_score": 1.0}, # low bm25
]

# Define mock embeddings that will result in predictable cosine similarities
# Query embedding is [1, 0]
# Doc embeddings:
# doc1 -> [0, 1] (cosine=0.0, low similarity)
# doc2 -> [0.707, 0.707] (cosine=0.707, medium similarity)
# doc3 -> [1, 0] (cosine=1.0, high similarity)
MOCK_EMBEDDINGS = {
    "query": np.array([1.0, 0.0]),
    "docs": np.array([
        [0.0, 1.0],
        [0.707, 0.707],
        [1.0, 0.0],
    ], dtype=np.float32)
}

def test_hybrid_retrieve_reranking_logic(monkeypatch):
    """
    Tests the core scoring and sorting logic of hybrid_retrieve
    by mocking its dependencies (sparse_retrieve, encode_texts, and redis_client).
    """
    # 1. Mock the Redis client to ensure this is a true unit test
    # This prevents the test from depending on an external Redis server.
    monkeypatch.setattr(retrieval.core, "redis_client", None)

    # 2. Define the fake functions for Elasticsearch and embeddings
    def mock_sparse_retrieve(query, size, index=None):
        return MOCK_BM25_HITS

    def mock_encode_query(text, **kwargs):
        return MOCK_EMBEDDINGS["query"]

    def mock_encode_texts(texts, **kwargs):
        # This is called for batches of documents
        return MOCK_EMBEDDINGS["docs"]

    # 3. Apply the mocks using pytest's monkeypatch fixture
    monkeypatch.setattr(retrieval.core, "sparse_retrieve", mock_sparse_retrieve)
    monkeypatch.setattr(retrieval.core, "encode_query", mock_encode_query)
    monkeypatch.setattr(retrieval.core, "encode_texts", mock_encode_texts)

    # 4. Run tests with different alpha values

    # Case A: Alpha = 0.0 (purely semantic search)
    # Reranking should be based only on cosine similarity.
    # Expected order: doc3 (cos=1.0), doc2 (cos=0.707), doc1 (cos=0.0)
    results_alpha_0 = retrieval.core.hybrid_retrieve("any query", alpha=0.0, rerank_k=3)
    result_ids_alpha_0 = [r["hit"]["_id"] for r in results_alpha_0]
    assert result_ids_alpha_0 == ["doc3", "doc2", "doc1"]
    assert results_alpha_0[0]["score"] == pytest.approx(1.0) # score should equal cosine

    # Case B: Alpha = 1.0 (purely keyword-based ranking)
    # Reranking should be based only on BM25 scores.
    # Expected order: doc1 (bm25=10.0), doc2 (bm25=5.0), doc3 (bm25=1.0)
    results_alpha_1 = retrieval.core.hybrid_retrieve("any query", alpha=1.0, rerank_k=3)
    result_ids_alpha_1 = [r["hit"]["_id"] for r in results_alpha_1]
    assert result_ids_alpha_1 == ["doc1", "doc2", "doc3"]
    # score should equal normalized bm25: 10.0 / (10.0 + 1.0) = 0.909
    assert results_alpha_1[0]["score"] == pytest.approx(10.0 / 11.0)

    # Case C: Alpha = 0.5 (balanced hybrid search)
    # We need to calculate the expected scores:
    # doc1: 0.5 * (10/11) + 0.5 * 0.0   = 0.4545
    # doc2: 0.5 * (5/6)   + 0.5 * 0.707 = 0.4166 + 0.3535 = 0.7701
    # doc3: 0.5 * (1/2)   + 0.5 * 1.0   = 0.25 + 0.5 = 0.75
    # Expected order: doc2, doc3, doc1
    results_alpha_0_5 = retrieval.core.hybrid_retrieve("any query", alpha=0.5, rerank_k=3)
    result_ids_alpha_0_5 = [r["hit"]["_id"] for r in results_alpha_0_5]
    assert result_ids_alpha_0_5 == ["doc2", "doc3", "doc1"]

    # Case D: Test rerank_k parameter
    # Should only return the top 2 results
    results_k_2 = retrieval.core.hybrid_retrieve("any query", alpha=0.5, rerank_k=2)
    assert len(results_k_2) == 2
    assert results_k_2[0]["hit"]["_id"] == "doc2"

