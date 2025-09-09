"""Retrieval implementations.

Contains sparse, dense and hybrid retrieval helpers moved from the legacy
`ir_core.retrieval` module.
"""
from ..infra import get_es
from ..config import settings
import numpy as np
from ..embeddings.core import encode_texts


def sparse_retrieve(query: str, size: int = 10, index: str = None):
    es = get_es()
    idx = index or settings.INDEX_NAME
    q = {"query": {"match": {"content": {"query": query}}}, "size": size}
    res = es.search(index=idx, body=q)
    return res.get("hits", {}).get("hits", [])


def dense_retrieve(query_emb: np.ndarray, size: int = 10, index: str = None):
    es = get_es()
    idx = index or settings.INDEX_NAME
    q = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                    "params": {"query_vector": query_emb.tolist()},
                },
            }
        },
    }
    res = es.search(index=idx, body=q)
    return res.get("hits", {}).get("hits", [])


def hybrid_retrieve(query: str, bm25_k: int = None, rerank_k: int = None, alpha: float = None):
    bm25_k = bm25_k or settings.BM25_K
    rerank_k = rerank_k or settings.RERANK_K
    alpha = settings.ALPHA if alpha is None else alpha

    bm25_hits = sparse_retrieve(query, size=bm25_k)
    bm25_ids = [h["_id"] for h in bm25_hits]
    texts = [h["_source"].get("content", "") for h in bm25_hits]
    if not texts:
        return []
    q_emb = encode_texts([query])[0]
    doc_embs = encode_texts(texts)
    cosines = (doc_embs @ q_emb).tolist()
    results = []
    for hit, cos in zip(bm25_hits, cosines):
        bm25_score = hit.get("_score", 0.0) or 0.0
        if alpha is None:
            score = cos
        else:
            score = alpha * (bm25_score / (bm25_score + 1.0)) + (1 - alpha) * cos
        results.append({"hit": hit, "cosine": float(cos), "score": float(score)})
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:rerank_k]
    return results
