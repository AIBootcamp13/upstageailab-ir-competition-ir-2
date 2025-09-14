"""Chunking guidance based on per-source length statistics.

Provides recommended chunk sizes and overlap ratios based on 
document length distributions from profiling data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import settings


def load_per_src_length_stats(report_dir: str | None = None) -> Dict[str, Dict[str, Any]]:
    """Load per-source content length statistics."""
    report_dir = report_dir or settings.PROFILE_REPORT_DIR
    if not report_dir:
        return {}
    p = Path(report_dir) / "per_src_length_stats.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def recommend_chunk_size(src: str, 
                        default_size: int = 512,
                        length_stats: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
    """Recommend chunk size for a given source based on content length stats."""
    if length_stats is None:
        length_stats = load_per_src_length_stats()
    
    if not length_stats or src not in length_stats:
        return default_size
    
    stats = length_stats[src]
    char_stats = stats.get("content_chars", {})
    word_stats = stats.get("content_words", {})
    
    # Get median word count
    median_words = word_stats.get("p50", 100)
    
    # Heuristic: for short docs (< 200 words), use smaller chunks
    # for long docs (> 1000 words), use larger chunks
    if median_words < 200:
        return min(default_size, 256)  # smaller chunks for short docs
    elif median_words > 1000:
        return max(default_size, 768)  # larger chunks for long docs
    else:
        return default_size


def recommend_overlap_ratio(src: str,
                           default_ratio: float = 0.1,
                           length_stats: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
    """Recommend overlap ratio based on document variance."""
    if length_stats is None:
        length_stats = load_per_src_length_stats()
    
    if not length_stats or src not in length_stats:
        return default_ratio
    
    stats = length_stats[src]
    word_stats = stats.get("content_words", {})
    
    mean_words = word_stats.get("mean", 100)
    p90_words = word_stats.get("p90", 200)
    
    # Higher overlap for more variable content
    if p90_words > 2 * mean_words:  # high variance
        return min(0.2, default_ratio * 2)
    else:
        return default_ratio


def get_chunking_config(src: str) -> Dict[str, Any]:
    """Get complete chunking configuration for a source."""
    length_stats = load_per_src_length_stats()
    
    return {
        "chunk_size": recommend_chunk_size(src, length_stats=length_stats),
        "overlap_ratio": recommend_overlap_ratio(src, length_stats=length_stats),
        "has_stats": src in length_stats if length_stats else False
    }


__all__ = [
    "load_per_src_length_stats",
    "recommend_chunk_size", 
    "recommend_overlap_ratio",
    "get_chunking_config"
]