"""Embedding utilities package.

Re-exports encoding helpers from the concrete implementation in
`ir_core.embeddings.core`.
"""
from .core import load_model, encode_texts, encode_query

__all__ = ["load_model", "encode_texts", "encode_query"]
