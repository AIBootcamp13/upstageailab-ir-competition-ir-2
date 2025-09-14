# src/ir_core/retrieval/core.py
import redis
import json
import numpy as np
from typing import Optional, List, Any, cast

from ..infra import get_es
from ..config import settings
from ..embeddings.core import encode_texts
from .boosting import load_keywords_per_src, build_boosted_query
from .preprocessing import filter_stopwords
from .deduplication import (
    load_duplicates, 
    build_duplicate_blacklist, 
    filter_duplicates, 
    apply_near_dup_penalty
)

# --- NEW: Initialize Redis Client for Caching ---
# Create a Redis client instance from the URL in the settings.
# decode_responses=False is important as we will be storing raw bytes.
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
    redis_client.ping()
    print("Redis client connected successfully for caching.")
except redis.ConnectionError as e:
    print(f"Warning: Could not connect to Redis for caching. Performance will be degraded. Error: {e}")
    redis_client = None


def sparse_retrieve(query: str, size: int = 10, index: Optional[str] = None):
    es = get_es()
    idx = index or settings.INDEX_NAME

    # Optional query preprocessing
    processed_query = query
    if settings.USE_STOPWORD_FILTERING:
        processed_query = filter_stopwords(query)

    q = {"query": {"match": {"content": {"query": processed_query}}}, "size": size}

    # Optional boosting using profiling artifacts
    if settings.USE_SRC_BOOSTS and settings.PROFILE_REPORT_DIR:
        kw = load_keywords_per_src(settings.PROFILE_REPORT_DIR)
        if kw:
            q = build_boosted_query(processed_query, size, kw)

    res = es.search(index=idx, body=q)
    return res.get("hits", {}).get("hits", [])


def dense_retrieve(query_emb: np.ndarray, size: int = 10, index: Optional[str] = None):
    # ... (existing code is unchanged)
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


def hybrid_retrieve(
    query: str,
    bm25_k: Optional[int] = None,
    rerank_k: Optional[int] = None,
    alpha: Optional[float] = None,
):
    bm25_k = bm25_k or settings.BM25_K
    rerank_k = rerank_k or settings.RERANK_K
    alpha = settings.ALPHA if alpha is None else alpha

    bm25_hits = sparse_retrieve(query, size=bm25_k)
    if not bm25_hits:
        return []

    # --- Caching Logic ---
    doc_embs = []
    texts_to_encode = []
    indices_to_encode = [] # Keep track of which documents need encoding

    if redis_client:
        # 1. Try to fetch embeddings from Redis cache first
        doc_ids = [h.get("_id") for h in bm25_hits]
        cached_embs_raw = cast(List[Any], redis_client.mget(doc_ids))

        for i, emb_raw in enumerate(cached_embs_raw):
            if emb_raw:
                # If found in cache, decode it
                doc_embs.append(np.frombuffer(emb_raw, dtype=np.float32))
            else:
                # If not found, mark it for encoding
                doc_embs.append(None) # Placeholder
                texts_to_encode.append(bm25_hits[i]["_source"].get("content", ""))
                indices_to_encode.append(i)
    else:
        # If Redis is not available, encode everything
        texts_to_encode = [h["_source"].get("content", "") for h in bm25_hits]
        indices_to_encode = list(range(len(bm25_hits)))
        doc_embs = [None] * len(bm25_hits)


    # 2. Encode only the texts that were not in the cache
    if texts_to_encode:
        newly_encoded_embs = encode_texts(texts_to_encode)

        # 3. Fill in the missing embeddings and update the cache
        if redis_client:
            pipe = redis_client.pipeline()
            for i, emb in zip(indices_to_encode, newly_encoded_embs):
                doc_id = bm25_hits[i]["_id"]
                doc_embs[i] = emb
                # Cache the newly encoded embedding for future use
                pipe.set(doc_id, emb.tobytes())
            pipe.execute()
        else:
             for i, emb in zip(indices_to_encode, newly_encoded_embs):
                doc_embs[i] = emb

    # Ensure all embeddings are now populated
    doc_embs_np = np.array(doc_embs, dtype=np.float32)

    # --- End of Caching Logic ---

    q_emb = encode_texts([query])[0]
    cosines = (doc_embs_np @ q_emb).tolist()

    results = []
    for hit, cos in zip(bm25_hits, cosines):
        bm25_score = hit.get("_score", 0.0) or 0.0
        if alpha is None:
            score = cos
        else:
            score = alpha * (bm25_score / (bm25_score + 1.0)) + (1 - alpha) * cos
        results.append({"hit": hit, "cosine": float(cos), "score": float(score)})

    # Sort by combined score (descending)
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    # Deduplicate by stable doc id (prefer _source.docid if present,
    # otherwise fall back to ES internal _id). Since results_sorted is
    # ordered by score descending, keeping the first occurrence ensures
    # we preserve the highest-scored hit for each document.
    seen_ids = set()
    deduped = []
    for r in results_sorted:
        hit = r.get("hit", {})
        src = hit.get("_source", {}) if isinstance(hit, dict) else {}
        docid = None
        try:
            docid = src.get("docid") if isinstance(src, dict) else None
        except Exception:
            docid = None
        if not docid:
            # fallback to ES internal id
            try:
                docid = hit.get("_id")
            except Exception:
                docid = None

        if docid is None:
            # If no id can be determined, include the item (unique by position)
            deduped.append(r)
            continue

        if docid in seen_ids:
            continue
        seen_ids.add(docid)
        deduped.append(r)

    # Return top-k after deduplication
    final_results = deduped[:rerank_k]
    
    # Optional duplicate filtering using profiling artifacts
    if settings.USE_DUPLICATE_FILTERING and settings.PROFILE_REPORT_DIR:
        duplicates = load_duplicates(settings.PROFILE_REPORT_DIR)
        if duplicates:
            blacklist = build_duplicate_blacklist(duplicates)
            final_results = filter_duplicates(final_results, blacklist)
    
    # Optional near-duplicate penalty
    if settings.USE_NEAR_DUP_PENALTY and settings.PROFILE_REPORT_DIR:
        final_results = apply_near_dup_penalty(final_results, penalty=0.1)
    
    return final_results
