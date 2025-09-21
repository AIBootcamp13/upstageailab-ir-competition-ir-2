"""Duplicate detection and filtering using profiling outputs.

Handles exact and near-duplicate filtering during retrieval reranking
based on artifacts from the data profiling script.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set, List, Any

from ..config import settings


def load_duplicates(report_dir: str | None = None) -> Dict[str, List[str]]:
    """Load exact duplicate groups: hash -> [docids]."""
    report_dir = report_dir or settings.PROFILE_REPORT_DIR
    if not report_dir:
        return {}
    p = Path(report_dir) / "duplicates.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Convert from list format: [{"hash": str, "count": int, "docids": [str]}]
        result = {}
        for group in data:
            if isinstance(group, dict) and "docids" in group:
                key = group.get("hash", "")
                docids = group.get("docids", [])
                if key and docids:
                    result[key] = docids
        return result
    except Exception:
        return {}


def load_near_duplicates(report_dir: str | None = None) -> List[List[str]]:
    """Load near-duplicate clusters: [[doc_indices], ...]."""
    report_dir = report_dir or settings.PROFILE_REPORT_DIR
    if not report_dir:
        return []
    p = Path(report_dir) / "near_duplicates.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Convert from format: [{"size": int, "doc_indices_sample": [str]}]
        clusters = []
        for group in data:
            if isinstance(group, dict) and "doc_indices_sample" in group:
                indices = group.get("doc_indices_sample", [])
                if len(indices) > 1:
                    clusters.append(indices)
        return clusters
    except Exception:
        return []


def build_duplicate_blacklist(duplicates: Dict[str, List[str]]) -> Set[str]:
    """Build a blacklist of docids to exclude (keep only first per group)."""
    blacklist = set()
    for docids in duplicates.values():
        if len(docids) > 1:
            # Keep first, blacklist rest
            blacklist.update(docids[1:])
    return blacklist


def filter_duplicates(results: List[Dict[str, Any]], 
                     duplicate_blacklist: Set[str]) -> List[Dict[str, Any]]:
    """Filter out blacklisted docids from retrieval results."""
    if not duplicate_blacklist:
        return results
    
    filtered = []
    for r in results:
        hit = r.get("hit", {})
        src = hit.get("_source", {}) if isinstance(hit, dict) else {}
        docid = src.get("docid") if isinstance(src, dict) else None
        if docid not in duplicate_blacklist:
            filtered.append(r)
    return filtered


def apply_near_dup_penalty(results: List[Dict[str, Any]], 
                          penalty: float = 0.1) -> List[Dict[str, Any]]:
    """Apply score penalty to near-duplicates (placeholder - needs doc mapping)."""
    # Note: This is a simplified implementation. In practice, you'd need
    # to map doc indices from profiling back to actual docids, which 
    # requires maintaining that mapping during profiling.
    # For now, this is a no-op placeholder.
    return results


__all__ = [
    "load_duplicates", 
    "load_near_duplicates", 
    "build_duplicate_blacklist", 
    "filter_duplicates", 
    "apply_near_dup_penalty"
]