#!/usr/bin/env python3
"""
Retrieval Pipeline Module

Orchestrates the entire retrieval process using modular components.
Provides a clean interface for hybrid retrieval with RRF or alpha-based fusion.
"""

import redis
import numpy as np
from typing import List, Dict, Any, Optional
from ..config import settings
from ..embeddings.core import encode_query
from .query_processor import QueryProcessor
from .candidate_generator import BM25Retriever, DenseRetriever
from .embedding_manager import EmbeddingManager
from .reranker import RRFReRanker, AlphaBlendReRanker
from .post_processor import PostProcessor
from .insights_manager import get_domain_cluster


class RetrievalPipeline:
    """Main retrieval pipeline that orchestrates all components"""

    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 use_rrf: bool = True,
                 alpha: float = 0.5):
        self.redis_client = redis_client
        self.use_rrf = use_rrf
        self.alpha = alpha

        # Initialize components
        self.query_processor = QueryProcessor(redis_client)
        self.sparse_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever()
        self.embedding_manager = EmbeddingManager(redis_client)

        # Initialize re-ranker
        if use_rrf:
            self.reranker = RRFReRanker(k=60)
        else:
            self.reranker = AlphaBlendReRanker(alpha=alpha)

        self.post_processor = PostProcessor()

    def retrieve(self,
                 query: str,
                 bm25_k: Optional[int] = None,
                 rerank_k: Optional[int] = None,
                 use_profiling_insights: bool = True) -> List[Dict[str, Any]]:
        """
        Execute the complete retrieval pipeline

        Args:
            query: Search query string
            bm25_k: Number of BM25 results to retrieve
            rerank_k: Number of final results to return
            use_profiling_insights: Whether to apply profiling-based optimizations

        Returns:
            Ranked and processed retrieval results
        """
        bm25_k = bm25_k or settings.BM25_K
        rerank_k = rerank_k or settings.RERANK_K

        # 1. Process and enhance the query
        query_info = self.query_processor.process(query)
        enhanced_query = query_info["enhanced_query"]

        # 2. Generate sparse candidates
        sparse_candidates = self.sparse_retriever.retrieve(enhanced_query, bm25_k, query_info)

        # 3. Apply domain-based filtering if enabled
        if use_profiling_insights and settings.PROFILE_REPORT_DIR:
            sparse_candidates = self._apply_domain_filtering(sparse_candidates)

        if not sparse_candidates:
            return []

        # 4. Get embeddings for dense retrieval
        doc_embeddings = self.embedding_manager.get_embeddings(sparse_candidates)

        # 5. Encode query for dense retrieval
        query_embedding = encode_query(query)

        # 6. Validate embeddings
        if self._validate_embeddings(query_embedding, doc_embeddings):
            # Perform dense retrieval
            dense_candidates = self._perform_dense_retrieval(
                query_embedding, sparse_candidates, doc_embeddings
            )
        else:
            # Fallback to BM25 only
            print("⚠️ Invalid embeddings detected, falling back to BM25 only")
            dense_candidates = []

        # 7. Re-rank and fuse results
        if dense_candidates:
            fused_results = self.reranker.rank(
                sparse_results=sparse_candidates,
                dense_results=dense_candidates
            )
        else:
            # No dense results, return sparse results as-is
            fused_results = sparse_candidates

        # 8. Apply post-processing
        final_results = self.post_processor.process(fused_results, rerank_k)

        return final_results

    def _apply_domain_filtering(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply domain-based filtering to candidates"""
        insights_config = getattr(settings, 'profiling_insights', {})
        use_domain_routing = insights_config.get('use_domain_routing', True)

        if not use_domain_routing:
            return candidates

        try:
            # Get domain routing information from a sample document
            if candidates:
                sample_src = candidates[0].get("_source", {}).get("src", "")
                if sample_src:
                    cluster_info = get_domain_cluster(sample_src)
                    if cluster_info.get("cluster_id"):
                        # Filter candidates to preferred domain
                        filtered_candidates = []
                        for candidate in candidates:
                            src = candidate.get("_source", {}).get("src", "")
                            if src:
                                candidate_cluster = get_domain_cluster(src)
                                if candidate_cluster.get("cluster_id") == cluster_info.get("cluster_id"):
                                    filtered_candidates.append(candidate)

                        if filtered_candidates:
                            print(f"Applied domain routing: {len(filtered_candidates)} results from target cluster")
                            return filtered_candidates

        except Exception as e:
            print(f"Warning: Failed to apply domain filtering: {e}")

        return candidates

    def _validate_embeddings(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> bool:
        """Validate that embeddings are valid for computation"""
        if np.isnan(query_emb).any() or np.isinf(query_emb).any():
            return False

        if doc_embs.size == 0 or np.isnan(doc_embs).any() or np.isinf(doc_embs).any():
            return False

        # Check dimension compatibility
        if doc_embs.shape[1] != query_emb.shape[0]:
            print(f"❌ Dimension mismatch: Document embeddings have dimension {doc_embs.shape[1]}, "
                  f"but query embedding has dimension {query_emb.shape[0]}")
            return False

        return True

    def _perform_dense_retrieval(self,
                                query_emb: np.ndarray,
                                sparse_candidates: List[Dict[str, Any]],
                                doc_embs: np.ndarray) -> List[Dict[str, Any]]:
        """Perform dense retrieval and format results"""
        # Calculate cosine similarities
        cosines = (doc_embs @ query_emb).tolist()
        cosines = [float(c) if not np.isnan(c) else 0.0 for c in cosines]

        # Format dense results
        dense_results = []
        for candidate, cosine in zip(sparse_candidates, cosines):
            dense_candidate = candidate.copy()
            dense_candidate["score"] = cosine
            dense_results.append(dense_candidate)

        # Sort by cosine similarity (descending)
        dense_results.sort(key=lambda x: x["score"], reverse=True)

        return dense_results


# Convenience function for backward compatibility
def hybrid_retrieve(query: str,
                   bm25_k: Optional[int] = None,
                   rerank_k: Optional[int] = None,
                   alpha: Optional[float] = None,
                   use_profiling_insights: bool = True,
                   use_rrf: bool = True) -> List[Dict[str, Any]]:
    """
    Backward-compatible interface for hybrid retrieval

    This function maintains the same interface as the original monolithic function
    but uses the new modular pipeline internally.
    """
    # Initialize Redis client
    redis_client = None
    try:
        import redis
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        redis_client.ping()
    except Exception:
        pass

    # Create pipeline
    alpha = alpha if alpha is not None else settings.ALPHA
    pipeline = RetrievalPipeline(
        redis_client=redis_client,
        use_rrf=use_rrf,
        alpha=alpha
    )

    # Execute retrieval
    return pipeline.retrieve(query, bm25_k, rerank_k, use_profiling_insights)