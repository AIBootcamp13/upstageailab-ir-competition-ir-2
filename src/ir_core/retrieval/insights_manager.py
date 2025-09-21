"""
Profiling Insights Manager for Retrieval Pipeline Integration

Loads and caches Phase 1 profiling insights for real-time use in:
- Query routing based on domain clustering
- Dynamic chunking based on long document analysis
- Memory optimization based on token statistics
- Query expansion based on vocabulary overlap
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from functools import lru_cache
import time

from ..config import settings


class ProfilingInsightsManager:
    """
    Manages loading and caching of profiling insights for retrieval optimization.
    """

    def __init__(self, report_dir: Optional[str] = None):
        self.report_dir = Path(report_dir or settings.PROFILE_REPORT_DIR or "outputs/reports/data_profile/latest")
        self._cache = {}
        self._last_load_time = 0
        # Use configurable TTL from settings
        self._cache_ttl = getattr(settings, 'profiling_insights', {}).get('cache_ttl_seconds', 300)

    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed based on TTL and file modifications."""
        if not self.report_dir.exists():
            return False

        current_time = time.time()
        if current_time - self._last_load_time > self._cache_ttl:
            return True

        # Check if any profiling files have been modified
        profiling_files = [
            "long_doc_analysis.json",
            "vocab_overlap_matrix.json",
            "src_clusters_by_vocab.json",
            "per_src_length_stats.json"
        ]

        for filename in profiling_files:
            file_path = self.report_dir / filename
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                if mtime > self._last_load_time:
                    return True

        return False

    def _load_insights(self) -> Dict[str, Any]:
        """Load all profiling insights from disk."""
        insights = {}

        # Load long document analysis
        long_doc_path = self.report_dir / "long_doc_analysis.json"
        if long_doc_path.exists():
            try:
                with open(long_doc_path, 'r', encoding='utf-8') as f:
                    insights['long_doc'] = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load long_doc_analysis.json: {e}")

        # Load vocabulary overlap matrix
        vocab_path = self.report_dir / "vocab_overlap_matrix.json"
        if vocab_path.exists():
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    insights['vocab_overlap'] = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load vocab_overlap_matrix.json: {e}")

        # Load source clusters
        clusters_path = self.report_dir / "src_clusters_by_vocab.json"
        if clusters_path.exists():
            try:
                with open(clusters_path, 'r', encoding='utf-8') as f:
                    insights['clusters'] = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load src_clusters_by_vocab.json: {e}")

        # Load per-source length stats
        stats_path = self.report_dir / "per_src_length_stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    insights['per_src_stats'] = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load per_src_length_stats.json: {e}")

        return insights

    def get_insights(self, refresh: bool = False) -> Dict[str, Any]:
        """Get profiling insights, loading from cache or disk as needed."""
        if refresh or self._should_refresh_cache() or not self._cache:
            self._cache = self._load_insights()
            self._last_load_time = time.time()

        return self._cache

    @lru_cache(maxsize=128)
    def get_chunking_recommendation(self, src: str, default_size: int = 512) -> Dict[str, Any]:
        """Get chunking recommendations based on long document analysis."""
        insights = self.get_insights()
        long_doc_info = insights.get('long_doc', {}).get('per_source', {})

        if src not in long_doc_info:
            return {
                "chunk_size": default_size,
                "overlap": int(default_size * 0.1),
                "recommendation": "Default chunking (no profiling data)"
            }

        long_fraction = long_doc_info[src]['long_doc_fraction']

        # Adjust chunk size based on long document fraction
        if long_fraction > 0.12:  # High long-doc fraction
            chunk_size = 768
            overlap = 100
        elif long_fraction > 0.08:  # Medium long-doc fraction
            chunk_size = 640
            overlap = 75
        else:  # Low long-doc fraction
            chunk_size = 512
            overlap = 50

        return {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "long_doc_fraction": long_fraction,
            "recommendation": f"Optimized for {long_fraction:.1%} long documents"
        }

    @lru_cache(maxsize=128)
    def get_domain_cluster(self, src: str) -> Dict[str, Any]:
        """Find which cluster a source belongs to."""
        insights = self.get_insights()
        clusters = insights.get('clusters', {}).get('clusters', [])

        for cluster in clusters:
            if src in cluster['sources']:
                return {
                    "cluster_id": cluster['id'],
                    "cluster_size": cluster['size'],
                    "common_terms": cluster['common_terms'][:5],
                    "related_sources": cluster['sources'][:10]
                }

        return {"cluster_id": None, "error": "Source not found in clusters"}

    @lru_cache(maxsize=128)
    def get_memory_recommendation(self, src: str) -> Dict[str, Any]:
        """Get memory optimization recommendations based on token statistics."""
        insights = self.get_insights()
        per_src_stats = insights.get('per_src_stats', {})

        if src not in per_src_stats or 'content_tokens' not in per_src_stats[src]:
            return {
                "batch_size": 16,
                "recommendation": "Default batch size (no token stats)"
            }

        token_stats = per_src_stats[src]['content_tokens']
        avg_tokens = token_stats.get('mean', 300)

        # Adjust batch size based on average token count
        if avg_tokens > 600:  # Very long documents
            batch_size = 4
        elif avg_tokens > 400:  # Long documents
            batch_size = 8
        elif avg_tokens < 200:  # Short documents
            batch_size = 32
        else:  # Medium documents
            batch_size = 16

        return {
            "batch_size": batch_size,
            "avg_tokens": avg_tokens,
            "recommendation": f"Optimized for {avg_tokens:.0f} avg tokens per document"
        }

    @lru_cache(maxsize=64)
    def get_query_expansion_terms(self, src: str, top_k: int = 5) -> List[str]:
        """Get query expansion terms based on vocabulary overlap."""
        insights = self.get_insights()
        vocab_overlap = insights.get('vocab_overlap', {})

        if src not in vocab_overlap:
            return []

        # Find sources with high overlap (>0.3)
        related_sources = []
        for other_src, overlap_score in vocab_overlap[src].items():
            if overlap_score > 0.3:
                related_sources.append((other_src, overlap_score))

        # Sort by overlap score and get top related sources
        related_sources.sort(key=lambda x: x[1], reverse=True)
        top_sources = related_sources[:3]  # Top 3 related sources

        # Get common terms from related sources
        expansion_terms = []
        clusters = insights.get('clusters', {}).get('clusters', [])

        for related_src, _ in top_sources:
            for cluster in clusters:
                if related_src in cluster['sources']:
                    expansion_terms.extend(cluster['common_terms'][:2])  # 2 terms per source

        return list(set(expansion_terms))[:top_k]  # Remove duplicates and limit

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of profiling insights system."""
        insights = self.get_insights()

        return {
            "insights_loaded": bool(insights),
            "cache_age_seconds": time.time() - self._last_load_time,
            "available_insights": list(insights.keys()),
            "long_doc_sources": len(insights.get('long_doc', {}).get('per_source', {})),
            "vocab_sources": len(insights.get('vocab_overlap', {})),
            "clusters_count": len(insights.get('clusters', {}).get('clusters', [])),
            "stats_sources": len(insights.get('per_src_stats', {}))
        }


# Global instance for easy access
insights_manager = ProfilingInsightsManager()


def get_chunking_recommendation(src: str, default_size: int = 512) -> Dict[str, Any]:
    """Convenience function for chunking recommendations."""
    return insights_manager.get_chunking_recommendation(src, default_size)


def get_domain_cluster(src: str) -> Dict[str, Any]:
    """Convenience function for domain cluster lookup."""
    return insights_manager.get_domain_cluster(src)


def get_memory_recommendation(src: str) -> Dict[str, Any]:
    """Convenience function for memory optimization."""
    return insights_manager.get_memory_recommendation(src)


def get_query_expansion_terms(src: str, top_k: int = 5) -> List[str]:
    """Convenience function for query expansion."""
    return insights_manager.get_query_expansion_terms(src, top_k)


def refresh_insights() -> Dict[str, Any]:
    """Force refresh of profiling insights cache."""
    return insights_manager.get_insights(refresh=True)


def get_insights_status() -> Dict[str, Any]:
    """Get status of profiling insights system."""
    return insights_manager.get_system_status()