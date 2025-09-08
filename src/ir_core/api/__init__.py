"""High-level facade package for the project.

This module exposes a stable set of helper functions that scripts
and CLIs can import. Implementations live in subpackages.
"""
from ..config import settings
from ..infra import get_es
from ..embeddings.core import load_model, encode_texts, encode_query
from ..retrieval.core import sparse_retrieve, dense_retrieve, hybrid_retrieve
from ..evaluation.core import precision_at_k, mrr
from ..utils.core import read_jsonl, write_jsonl

__all__ = [
    "settings",
    "get_es",
    "load_model",
    "encode_texts",
    "encode_query",
    "sparse_retrieve",
    "dense_retrieve",
    "hybrid_retrieve",
    "precision_at_k",
    "mrr",
    "read_jsonl",
    "write_jsonl",
]


def index_documents_from_jsonl(jsonl_path, index_name=None):
    """Tiny convenience helper used by scripts.

    Reads JSONL objects and indexes them into Elasticsearch using the
    configured `get_es()` client.
    """
    es = get_es()
    idx = index_name or settings.INDEX_NAME
    for doc in read_jsonl(jsonl_path):
        doc_id = doc.get("id") or doc.get("_id")
        es.index(index=idx, id=doc_id, document=doc)
