"""Retrieval helpers package.

Re-export core retrieval functions from `ir_core.retrieval.core`.
"""
from .core import sparse_retrieve, dense_retrieve, hybrid_retrieve

__all__ = ["sparse_retrieve", "dense_retrieve", "hybrid_retrieve"]
