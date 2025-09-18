# src/ir_core/retrieval/core.py
import redis
import json
import numpy as np
from typing import Optional, List, Any, cast, Dict

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
from .keywords_integration import get_curated_keywords_integrator, enhance_query_with_curated_keywords
from ..generation import get_generator
from omegaconf import OmegaConf
import os

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


def build_flexible_match_query(query: str, size: int) -> dict:
    """
    Build an enhanced multi-field boolean query that leverages generated metadata.
    This replaces hardcoded logic with dynamic keyword extraction and multi-field search.

    The query searches across:
    - content: Full document text (default boost)
    - summary: Document summary (medium boost)
    - keywords: Extracted keywords (high boost)
    - hypothetical_questions: Generated questions (high boost)
    """
    # Extract dynamic keywords from the query using LLM
    dynamic_keywords = _extract_keywords_from_query(query)

    # Enhance with curated scientific keywords (improved with similarity threshold)
    try:
        integrator = get_curated_keywords_integrator()
        enhanced_query, all_keywords = integrator.enhance_query_with_keywords(
            query, dynamic_keywords, use_semantic_matching=True, max_additional=3
        )

        # Use all keywords (LLM + curated) for the keywords field
        combined_keywords = all_keywords if all_keywords else dynamic_keywords

    except Exception as e:
        print(f"Warning: Curated keywords integration failed: {e}")
        enhanced_query = query
        combined_keywords = dynamic_keywords

    # Build multi-field query with appropriate boosts
    bool_query = {
        "should": [
            # Highest boost for keywords field - direct keyword matches are very strong signals
            {"match": {"keywords": {"query": ' '.join(combined_keywords), "boost": 4.0}}},
            # High boost for hypothetical questions - these are phrased like user queries
            {"match": {"hypothetical_questions": {"query": enhanced_query, "boost": 3.0}}},
            # Medium boost for summary - concise version of document content
            {"match": {"summary": {"query": enhanced_query, "boost": 2.0}}},
            # Default boost for full content - comprehensive but less specific
            {"match": {"content": {"query": enhanced_query, "boost": 1.0}}}
        ],
        "minimum_should_match": 1  # At least one field must match
    }

    # Add phrase matching for better precision on Korean queries
    if any('\uac00' <= char <= '\ud7a3' for char in enhanced_query):
        bool_query["should"].append({
            "match_phrase": {
                "content": {
                    "query": enhanced_query,
                    "slop": 2,  # Allow some reordering
                    "boost": 1.5
                }
            }
        })

    return {
        "query": {
            "bool": bool_query
        },
        "size": size
    }


def sparse_retrieve(query: str, size: int = 10, index: Optional[str] = None):
    es = get_es()
    idx = index or settings.INDEX_NAME

    # Optional query preprocessing
    processed_query = query
    if settings.USE_STOPWORD_FILTERING:
        processed_query = filter_stopwords(query)

    # Create a more flexible boolean query for better term matching
    q = build_flexible_match_query(processed_query, size)

    # Debug: Print the query being sent to Elasticsearch
    print(f"DEBUG: BM25 Query for '{query}': {q}")

    # Optional boosting using profiling artifacts
    if settings.USE_SRC_BOOSTS and settings.PROFILE_REPORT_DIR:
        kw = load_keywords_per_src(settings.PROFILE_REPORT_DIR)
        if kw:
            boosting_result = build_boosted_query(processed_query, size, kw)
            boosting_clauses = boosting_result.get("boosting_clauses", [])

            # Integrate boosting clauses into the existing flexible query
            if boosting_clauses and "query" in q and "bool" in q["query"]:
                if "should" not in q["query"]["bool"]:
                    q["query"]["bool"]["should"] = []
                q["query"]["bool"]["should"].extend(boosting_clauses)

    res = es.search(index=idx, body=q)
    return res.get("hits", {}).get("hits", [])


def dense_retrieve(query_emb: np.ndarray, size: int = 10, index: Optional[str] = None):
    # Validate query embedding for NaN/inf values
    if np.isnan(query_emb).any() or np.isinf(query_emb).any():
        print(f"⚠️  Invalid query embedding detected (contains NaN/inf), falling back to BM25")
        return sparse_retrieve("", size, index)  # Fallback to BM25 with empty query

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


def reciprocal_rank_fusion(
    sparse_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combine sparse and dense retrieval results using Reciprocal Rank Fusion (RRF).

    RRF is a state-of-the-art method for fusing results from different retrieval systems.
    It uses reciprocal ranks instead of raw scores, making it robust to different scoring scales.

    Args:
        sparse_results: Results from sparse (BM25) retrieval
        dense_results: Results from dense (vector) retrieval
        k: RRF constant (typically 60)

    Returns:
        Combined and re-ranked results with preserved original scores
    """
    # Create a mapping of document IDs to their results
    doc_map = {}

    # Process sparse results
    for rank, result in enumerate(sparse_results, 1):
        doc_id = result.get("_source", {}).get("docid") or result.get("_id")
        if doc_id:
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "sparse_rank": rank,
                    "dense_rank": None,
                    "sparse_score": result.get("_score", 0.0),
                    "dense_score": None
                }
            else:
                doc_map[doc_id]["sparse_rank"] = rank
                doc_map[doc_id]["sparse_score"] = result.get("_score", 0.0)

    # Process dense results
    for rank, result in enumerate(dense_results, 1):
        doc_id = result.get("_source", {}).get("docid") or result.get("_id")
        if doc_id:
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "sparse_rank": None,
                    "dense_rank": rank,
                    "sparse_score": None,
                    "dense_score": result.get("score", 0.0)
                }
            else:
                doc_map[doc_id]["dense_rank"] = rank
                doc_map[doc_id]["dense_score"] = result.get("score", 0.0)

    # Calculate RRF scores
    for doc_id, doc_info in doc_map.items():
        rrf_score = 0.0

        if doc_info["sparse_rank"] is not None:
            rrf_score += 1.0 / (k + doc_info["sparse_rank"])

        if doc_info["dense_rank"] is not None:
            rrf_score += 1.0 / (k + doc_info["dense_rank"])

        doc_info["rrf_score"] = rrf_score

        # Preserve original scores in the result
        result = doc_info["result"]
        # Add RRF score
        result["rrf_score"] = rrf_score
        # Preserve original scores
        if doc_info["sparse_score"] is not None:
            result["sparse_score"] = doc_info["sparse_score"]
        if doc_info["dense_score"] is not None:
            result["dense_score"] = doc_info["dense_score"]
        # Set the final score to RRF score for compatibility
        result["score"] = rrf_score

    # Sort by RRF score (descending) and return results
    sorted_docs = sorted(doc_map.values(), key=lambda x: x["rrf_score"], reverse=True)

    return [doc_info["result"] for doc_info in sorted_docs]


def hybrid_retrieve(
    query: str,
    bm25_k: Optional[int] = None,
    rerank_k: Optional[int] = None,
    alpha: Optional[float] = None,
    use_profiling_insights: bool = True,
    use_rrf: bool = True,
):
    """
    Enhanced hybrid retrieval with RRF (Reciprocal Rank Fusion) support.

    This function now uses the new modular RetrievalPipeline for better maintainability.
    The original monolithic implementation has been refactored into separate components.

    Args:
        query: Search query string
        bm25_k: Number of BM25 results to retrieve
        rerank_k: Number of final results to return
        alpha: Weight for BM25 vs dense retrieval (deprecated when use_rrf=True)
        use_profiling_insights: Whether to apply profiling-based optimizations
        use_rrf: Whether to use RRF instead of alpha-based weighting
    """
    from .retrieval_pipeline import RetrievalPipeline

    # Create pipeline with existing Redis client
    alpha = alpha if alpha is not None else settings.ALPHA
    pipeline = RetrievalPipeline(
        redis_client=redis_client,
        use_rrf=use_rrf,
        alpha=alpha
    )

    # Execute retrieval using the modular pipeline
    return pipeline.retrieve(query, bm25_k, rerank_k, use_profiling_insights)


def _extract_keywords_from_query(query: str) -> List[str]:
    """
    Uses an LLM to extract the most critical keywords from a user query.
    This replaces hardcoded synonym mappings with dynamic, context-aware keyword extraction.
    """
    try:
        # Try to get the generator from settings
        from ..config import settings

        # Create a minimal config for the generator
        cfg_dict = {
            'pipeline': {
                'generator_type': getattr(settings, 'GENERATOR_TYPE', 'openai'),
                'generator_model_name': getattr(settings, 'GENERATOR_MODEL_NAME', 'gpt-4o-mini'),
            },
            'prompts': {
                'generation_qa': getattr(settings, 'GENERATOR_SYSTEM_MESSAGE_FILE', ''),
                'persona': getattr(settings, 'GENERATOR_SYSTEM_MESSAGE_FILE', ''),
            }
        }
        cfg = OmegaConf.create(cfg_dict)

        generator = get_generator(cfg)

        prompt = f"""
        Extract the most important and specific keywords from the following user query.
        Focus on nouns, technical terms, and core concepts that would be most relevant for document retrieval.
        Return only the keywords as a comma-separated list, no explanations.

        Query: "{query}"

        Keywords:
        """

        # Use the generator's client directly for a simple completion
        if hasattr(generator, 'client') and hasattr(generator, 'model_name'):
            response = generator.client.chat.completions.create(  # type: ignore
                model=generator.model_name,  # type: ignore
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            keywords_str = response.choices[0].message.content or ""
            keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
            print(f"Extracted keywords: {keywords}")
            return keywords
        else:
            # Fallback to simple splitting if no client available
            print("Warning: No LLM client available, falling back to simple keyword extraction")
            return [word.strip() for word in query.split() if len(word.strip()) > 1]

    except Exception as e:
        print(f"Warning: Keyword extraction failed: {e}")
        # Fallback to simple splitting
        return [word.strip() for word in query.split() if len(word.strip()) > 1]


# --- NEW: Initialize curated keywords integration ---
try:
    from .keywords_integration import initialize_keywords_integration
    initialize_keywords_integration()
    print("Curated keywords integration initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize curated keywords integration: {e}")
