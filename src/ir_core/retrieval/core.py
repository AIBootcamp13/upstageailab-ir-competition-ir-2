# src/ir_core/retrieval/core.py
import redis
import json
import numpy as np
from typing import Optional, List, Any, cast

from ..infra import get_es
from ..config import settings
from ..embeddings.core import encode_texts, encode_query
from .boosting import load_keywords_per_src, build_boosted_query
from .preprocessing import filter_stopwords
from .deduplication import (
    load_duplicates,
    build_duplicate_blacklist,
    filter_duplicates,
    apply_near_dup_penalty
)
from .insights_manager import (
    insights_manager,
    get_chunking_recommendation,
    get_domain_cluster,
    get_memory_recommendation,
    get_query_expansion_terms
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
    use_profiling_insights: bool = True,
):
    """
    Enhanced hybrid retrieval with profiling insights integration.

    Args:
        query: Search query string
        bm25_k: Number of BM25 results to retrieve
        rerank_k: Number of final results to return
        alpha: Weight for BM25 vs dense retrieval (0.0 = dense only, 1.0 = BM25 only)
        use_profiling_insights: Whether to apply profiling-based optimizations
    """
    bm25_k = bm25_k or settings.BM25_K
    rerank_k = rerank_k or settings.RERANK_K
    alpha = settings.ALPHA if alpha is None else alpha

    # --- NEW: Apply profiling insights for query enhancement ---
    enhanced_query = query
    query_expansion_terms = []
    domain_routing = None

    # Get profiling insights configuration
    insights_config = getattr(settings, 'profiling_insights', {})
    use_query_expansion = insights_config.get('use_query_expansion', True)
    use_domain_routing = insights_config.get('use_domain_routing', True)
    use_memory_optimization = insights_config.get('use_memory_optimization', True)
    query_expansion_terms_count = insights_config.get('query_expansion_terms', 3)

    if use_profiling_insights and settings.PROFILE_REPORT_DIR and use_query_expansion:
        try:
            # Get query expansion terms based on vocabulary overlap
            # Use a representative source or try multiple sources
            insights = insights_manager.get_insights()
            vocab_sources = list(insights.get('vocab_overlap', {}).keys())

            if vocab_sources:
                # Use the first source as representative for query expansion
                representative_src = vocab_sources[0]
                query_expansion_terms = get_query_expansion_terms(representative_src, top_k=query_expansion_terms_count)

                # Enhance query with expansion terms if they seem relevant
                if query_expansion_terms:
                    # Simple relevance check: if expansion terms appear in query, boost them
                    query_lower = query.lower()
                    relevant_terms = [term for term in query_expansion_terms
                                    if term.lower() in query_lower or
                                       any(word in query_lower for word in term.split())]

                    if relevant_terms:
                        enhanced_query = f"{query} {' '.join(relevant_terms)}"
                        print(f"Enhanced query with expansion terms: {enhanced_query}")

        except Exception as e:
            print(f"Warning: Failed to apply query expansion: {e}")

    # Use enhanced query for BM25 retrieval
    bm25_hits = sparse_retrieve(enhanced_query, size=bm25_k)

    # --- NEW: Apply domain-based filtering if routing was determined ---
    if use_profiling_insights and domain_routing and use_domain_routing:
        try:
            # Filter results to preferred domains based on clustering
            filtered_hits = []
            for hit in bm25_hits:
                src = hit.get("_source", {}).get("src", "")
                if src:
                    cluster_info = get_domain_cluster(src)
                    if cluster_info.get("cluster_id") == domain_routing.get("target_cluster"):
                        filtered_hits.append(hit)

            # If we have filtered results, use them; otherwise fall back to all
            if filtered_hits:
                bm25_hits = filtered_hits[:bm25_k]  # Maintain original size limit
                print(f"Applied domain routing: {len(filtered_hits)} results from target cluster")
        except Exception as e:
            print(f"Warning: Failed to apply domain filtering: {e}")

    if not bm25_hits:
        return []

    # --- Enhanced Caching Logic with Memory Optimization ---
    doc_embs = []
    texts_to_encode = []
    indices_to_encode = [] # Keep track of which documents need encoding

    # Determine optimal batch size based on profiling insights
    optimal_batch_size = insights_config.get('memory_batch_fallback', 16)  # default from config
    if use_profiling_insights and bm25_hits and use_memory_optimization:
        try:
            # Get memory recommendations from a sample source
            sample_src = bm25_hits[0].get("_source", {}).get("src", "")
            if sample_src:
                memory_rec = get_memory_recommendation(sample_src)
                optimal_batch_size = memory_rec.get("batch_size", optimal_batch_size)
                print(f"Using optimized batch size: {optimal_batch_size} (based on {memory_rec.get('avg_tokens', 'N/A')} avg tokens)")
        except Exception as e:
            print(f"Warning: Failed to get memory optimization: {e}")

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

    q_emb = encode_query(query)
    cosines = (doc_embs_np @ q_emb).tolist()

    # Handle NaN values in cosine similarities
    cosines = [float(c) if not np.isnan(c) else 0.0 for c in cosines]

    results = []
    for hit, cos in zip(bm25_hits, cosines):
        bm25_score = hit.get("_score", 0.0) or 0.0

        # Handle NaN values from Elasticsearch
        if np.isnan(bm25_score):
            bm25_score = 0.0
        if np.isnan(cos):
            cos = 0.0

        if alpha is None:
            score = cos
        else:
            # Normalize BM25 score to avoid division by zero or NaN
            if bm25_score > 0:
                normalized_bm25 = bm25_score / (bm25_score + 1.0)
            else:
                normalized_bm25 = 0.0
            score = alpha * normalized_bm25 + (1 - alpha) * cos

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
