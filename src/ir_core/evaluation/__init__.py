"""Evaluation metrics package.

This module re-exports the canonical implementations from
``ir_core.evaluation.core`` to avoid duplication.
"""

from .core import precision_at_k, mrr

__all__ = ["precision_at_k", "mrr"]
