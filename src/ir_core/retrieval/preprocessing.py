"""Query preprocessing utilities using profiling outputs.

Handles stopword filtering and query normalization based on 
artifacts from the data profiling script.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Set, Optional

from ..config import settings


def load_stopwords(report_dir: str | None = None) -> Set[str]:
    """Load global stopwords from profiling output."""
    report_dir = report_dir or settings.PROFILE_REPORT_DIR
    if not report_dir:
        return set()
    p = Path(report_dir) / "stopwords_global.json"
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(data) if isinstance(data, list) else set()
    except Exception:
        return set()


def filter_stopwords(query: str, stopwords: Optional[Set[str]] = None) -> str:
    """Remove stopwords from query while preserving structure."""
    if stopwords is None:
        stopwords = load_stopwords()
    if not stopwords:
        return query
    
    # Simple token-based filtering (preserves phrases)
    tokens = query.split()
    filtered = [t for t in tokens if t.lower() not in stopwords]
    return " ".join(filtered) if filtered else query


__all__ = ["load_stopwords", "filter_stopwords"]