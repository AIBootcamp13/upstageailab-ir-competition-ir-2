#!/usr/bin/env python3
"""
Embedding Manager Module

Handles fetching, caching, and encoding of document embeddings.
Provides a unified interface for embedding operations with Redis caching support.
"""

import redis
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..config import settings
from ..embeddings.core import encode_texts
from .insights_manager import get_memory_recommendation


class EmbeddingManager:
    """Manages document embeddings with caching support"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client

    def get_embeddings(self, hits: List[Dict[str, Any]]) -> np.ndarray:
        """
        Get embeddings for the given hits, using cache when possible

        Args:
            hits: List of Elasticsearch hits

        Returns:
            Numpy array of embeddings
        """
        if not hits:
            return np.array([])

        # Extract document IDs and texts
        doc_ids = []
        texts_to_encode = []
        indices_to_encode = []

        for i, hit in enumerate(hits):
            if isinstance(hit, dict):
                doc_ids.append(hit.get("_id", ""))
            elif isinstance(hit, str):
                doc_ids.append(hit)
            else:
                doc_ids.append(str(hit) if hit else "")

        # Determine optimal batch size
        optimal_batch_size = self._get_optimal_batch_size(hits)

        # Try to fetch from cache first
        cached_embeddings = []
        if self.redis_client:
            from typing import cast
            cached_embs_raw = cast(List[Any], self.redis_client.mget(doc_ids))

            for i, emb_raw in enumerate(cached_embs_raw):
                if emb_raw:
                    # Decode cached embedding
                    cached_embeddings.append(np.frombuffer(emb_raw, dtype=np.float32))
                else:
                    # Mark for encoding
                    cached_embeddings.append(None)
                    text = self._extract_text_from_hit(hits[i])
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
        else:
            # No cache available, encode everything
            for i, hit in enumerate(hits):
                cached_embeddings.append(None)
                text = self._extract_text_from_hit(hit)
                texts_to_encode.append(text)
                indices_to_encode.append(i)

        # Encode missing embeddings
        if texts_to_encode:
            newly_encoded_embs = encode_texts(texts_to_encode)

            # Update cache and fill in missing embeddings
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                for i, emb in zip(indices_to_encode, newly_encoded_embs):
                    doc_id = doc_ids[i]
                    cached_embeddings[i] = emb
                    pipe.set(doc_id, emb.tobytes())
                pipe.execute()
            else:
                for i, emb in zip(indices_to_encode, newly_encoded_embs):
                    cached_embeddings[i] = emb

        # Convert to numpy array
        embeddings_array = np.array(cached_embeddings, dtype=np.float32)

        return embeddings_array

    def _extract_text_from_hit(self, hit: Dict[str, Any]) -> str:
        """Extract text content from an Elasticsearch hit"""
        if isinstance(hit, dict):
            return hit.get("_source", {}).get("content", "")
        elif isinstance(hit, str):
            return hit
        else:
            return str(hit) if hit else ""

    def _get_optimal_batch_size(self, hits: List[Dict[str, Any]]) -> int:
        """Determine optimal batch size based on profiling insights"""
        insights_config = getattr(settings, 'profiling_insights', {})
        optimal_batch_size = insights_config.get('memory_batch_fallback', 16)

        use_memory_optimization = insights_config.get('use_memory_optimization', True)

        if hits and use_memory_optimization:
            try:
                # Get memory recommendations from a sample source
                sample_hit = hits[0]
                if isinstance(sample_hit, dict):
                    sample_src = sample_hit.get("_source", {}).get("src", "")
                    if sample_src:
                        memory_rec = get_memory_recommendation(sample_src)
                        optimal_batch_size = memory_rec.get("batch_size", optimal_batch_size)
                        print(f"Using optimized batch size: {optimal_batch_size} (based on {memory_rec.get('avg_tokens', 'N/A')} avg tokens)")
            except Exception as e:
                print(f"Warning: Failed to get memory optimization: {e}")

        return optimal_batch_size