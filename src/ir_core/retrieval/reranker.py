#!/usr/bin/env python3
"""
ReRanker Module

Handles re-ranking and fusion of retrieval results from different sources.
Provides implementations for RRF (Reciprocal Rank Fusion) and alpha-based weighting.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class ReRanker(ABC):
    """Abstract base class for re-rankers"""

    @abstractmethod
    def rank(self, sparse_results: List[Dict[str, Any]],
             dense_results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Re-rank and fuse results from different retrieval sources"""
        pass


class RRFReRanker(ReRanker):
    """Reciprocal Rank Fusion re-ranker"""

    def __init__(self, k: int = 60):
        self.k = k

    def rank(self, sparse_results: List[Dict[str, Any]],
             dense_results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion to combine sparse and dense results"""
        print("Using Reciprocal Rank Fusion (RRF) for hybrid retrieval")

        # Create mapping of document IDs to results
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
                rrf_score += 1.0 / (self.k + doc_info["sparse_rank"])

            if doc_info["dense_rank"] is not None:
                rrf_score += 1.0 / (self.k + doc_info["dense_rank"])

            doc_info["rrf_score"] = rrf_score

            # Preserve original scores in the result
            result = doc_info["result"]
            result["rrf_score"] = rrf_score
            result["score"] = rrf_score  # Set final score to RRF score

            if doc_info["sparse_score"] is not None:
                result["sparse_score"] = doc_info["sparse_score"]
            if doc_info["dense_score"] is not None:
                result["dense_score"] = doc_info["dense_score"]

        # Sort by RRF score (descending) and return results
        sorted_docs = sorted(doc_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        return [doc_info["result"] for doc_info in sorted_docs]


class AlphaBlendReRanker(ReRanker):
    """Alpha-based weighted re-ranker"""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def rank(self, sparse_results: List[Dict[str, Any]],
             dense_results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Apply alpha-based weighting to combine sparse and dense results"""
        print(f"Using alpha-based weighting (alpha={self.alpha}) for hybrid retrieval")

        # Create mapping of results by document ID
        results_map = {}

        # Add sparse results
        for hit in sparse_results:
            doc_id = hit.get("_source", {}).get("docid") or hit.get("_id")
            if doc_id:
                results_map[doc_id] = {
                    "hit": hit,
                    "sparse_score": hit.get("_score", 0.0),
                    "dense_score": 0.0
                }

        # Add/update with dense results
        for result in dense_results:
            doc_id = result.get("_source", {}).get("docid") or result.get("_id")
            if doc_id:
                if doc_id in results_map:
                    results_map[doc_id]["dense_score"] = result.get("score", 0.0)
                else:
                    results_map[doc_id] = {
                        "hit": result,
                        "sparse_score": 0.0,
                        "dense_score": result.get("score", 0.0)
                    }

        # Calculate combined scores
        combined_results = []
        for doc_id, result_info in results_map.items():
            sparse_score = result_info["sparse_score"]
            dense_score = result_info["dense_score"]

            # Handle NaN values
            if np.isnan(sparse_score):
                sparse_score = 0.0
            if np.isnan(dense_score):
                dense_score = 0.0

            # Apply alpha weighting
            if sparse_score > 0:
                normalized_sparse = sparse_score / (sparse_score + 1.0)
            else:
                normalized_sparse = 0.0

            combined_score = self.alpha * normalized_sparse + (1 - self.alpha) * dense_score

            # Add score to the hit
            hit = result_info["hit"]
            hit["score"] = combined_score
            hit["sparse_score"] = sparse_score
            hit["dense_score"] = dense_score

            combined_results.append({
                "hit": hit,
                "score": combined_score
            })

        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return [result["hit"] for result in combined_results]