#!/usr/bin/env python3
"""
Post Processor Module

Handles post-processing of retrieval results including deduplication,
duplicate filtering, and near-duplicate penalties.
"""

from typing import List, Dict, Any, Set
from ..config import settings
from .deduplication import (
    load_duplicates,
    build_duplicate_blacklist,
    filter_duplicates,
    apply_near_dup_penalty
)


class PostProcessor:
    """Handles post-processing of retrieval results"""

    def __init__(self):
        pass

    def process(self, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Apply post-processing steps to retrieval results

        Args:
            results: List of retrieval results
            top_k: Number of final results to return

        Returns:
            Processed results
        """
        processed_results = results

        # Apply deduplication
        processed_results = self._deduplicate(processed_results)

        # Apply duplicate filtering using profiling artifacts
        processed_results = self._filter_duplicates(processed_results)

        # Apply near-duplicate penalty
        processed_results = self._apply_near_dup_penalty(processed_results)

        # Return top-k results
        return processed_results[:top_k]

    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents based on document ID"""
        seen_ids: Set[str] = set()
        deduped = []

        for result in results:
            if isinstance(result, dict):
                hit = result.get("hit", result)
            else:
                hit = result

            # Try to get document ID
            docid = None
            src = hit.get("_source", {}) if isinstance(hit, dict) else {}

            try:
                docid = src.get("docid") if isinstance(src, dict) else None
            except Exception:
                docid = None

            if not docid:
                try:
                    docid = hit.get("_id") if isinstance(hit, dict) else None
                except Exception:
                    docid = None

            if docid is None:
                # If no id can be determined, include the item (unique by position)
                deduped.append(result)
                continue

            if docid in seen_ids:
                continue

            seen_ids.add(docid)
            deduped.append(result)

        return deduped

    def _filter_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out known duplicate documents"""
        if not settings.USE_DUPLICATE_FILTERING or not settings.PROFILE_REPORT_DIR:
            return results

        try:
            duplicates = load_duplicates(settings.PROFILE_REPORT_DIR)
            if duplicates:
                blacklist = build_duplicate_blacklist(duplicates)
                return filter_duplicates(results, blacklist)
        except Exception as e:
            print(f"Warning: Failed to apply duplicate filtering: {e}")

        return results

    def _apply_near_dup_penalty(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply penalties to near-duplicate documents"""
        if not settings.USE_NEAR_DUP_PENALTY or not settings.PROFILE_REPORT_DIR:
            return results

        try:
            return apply_near_dup_penalty(results, penalty=0.1)
        except Exception as e:
            print(f"Warning: Failed to apply near-duplicate penalty: {e}")

        return results