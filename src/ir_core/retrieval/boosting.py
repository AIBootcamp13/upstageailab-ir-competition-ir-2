"""Boosting helpers for sparse retrieval using profiling outputs.

Loads per-source TF-IDF keywords and length stats to construct a
boosted ES query. Falls back gracefully if artifacts are missing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from ..config import settings


def load_keywords_per_src(report_dir: str | None = None) -> Dict[str, List[str]]:
    report_dir = report_dir or settings.PROFILE_REPORT_DIR
    if not report_dir:
        return {}
    p = Path(report_dir) / "keywords_per_src.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # file format: { src: [{"term": str, "weight": float}, ...] }
        return {s: [d.get("term", "") for d in lst if isinstance(d, dict) and d.get("term")] for s, lst in data.items()}
    except Exception:
        return {}


def build_boosted_query(query: str, size: int, keywords_by_src: Dict[str, List[str]]) -> Dict[str, Any]:
    """Construct a boosted ES query.

    Boost strategy:
    - Must match the main query against content (BM25 defaults)
    - Should also match any per-src keywords as a dis_max clause to give small boosts
    - Each src keyword is OR'd; weights are uniform (we keep it simple and robust)
    """
    should_clauses: List[Dict[str, Any]] = []
    for src, terms in keywords_by_src.items():
        if not terms:
            continue
        # Use a match on content with OR'd terms (simple_query_string) and a small boost
        should_clauses.append(
            {
                "simple_query_string": {
                    "query": " | ".join(sorted(set(terms))[:30]),
                    "fields": ["content^1.0"],
                    "default_operator": "or"
                }
            }
        )

    bool_query: Dict[str, Any] = {
        "must": [{"match": {"content": {"query": query}}}],
        "should": should_clauses,
    }

    return {
        "size": size,
        "query": {"bool": bool_query},
    }


__all__ = ["load_keywords_per_src", "build_boosted_query"]
