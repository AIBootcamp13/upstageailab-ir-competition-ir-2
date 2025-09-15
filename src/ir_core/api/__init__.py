"""High-level facade package for the project.

This module exposes a stable set of helper functions that scripts
and CLIs can import. Implementations live in subpackages.
"""
from ..config import settings
# NOTE: avoid importing `get_es` at module import time. Importing the
# package (for example `import ir_core.infra`) can cause package-level
# imports to pull in this module before tests have had a chance to
# monkeypatch `infra.get_es`. Resolve `get_es` at runtime inside the
# function instead.
from ..embeddings.core import load_model, encode_texts, encode_query
from ..retrieval.core import sparse_retrieve, dense_retrieve, hybrid_retrieve
from ..evaluation.core import precision_at_k, mrr
from ..utils.core import read_jsonl, write_jsonl

__all__ = [
    "settings",
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


def index_documents_from_jsonl(jsonl_path, index_name=None, batch_size: int = 500, *, dry_run: bool = False, verbose: bool = False, dedupe: bool = False):
    """Index documents from a JSONL file using the bulk API with progress."""
    # Resolve ES client at runtime so tests can patch `ir_core.infra.get_es`
    # before this module is imported.
    from .. import infra
    es = infra.get_es()
    idx = index_name or settings.INDEX_NAME

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
    except Exception:
        total = None

    from typing import Optional, Callable, Iterable
    tqdm: Optional[Callable[..., Iterable]] = None
    use_tqdm = False
    try:
        from tqdm import tqdm as _tqdm
        tqdm = _tqdm
        use_tqdm = True
    except Exception:
        tqdm = None

    indexed = 0

    try:
        from elasticsearch.helpers import bulk
    except Exception:
        bulk = None

    batch = []
    iterator = read_jsonl(jsonl_path)

    # If dedupe is requested, track seen docids and skip duplicates
    seen_docids = set() if dedupe else None

    if use_tqdm:
        assert tqdm is not None
        iterator = tqdm(iterator, total=total, desc=f"Indexing -> {idx}")

    import time
    start_time = time.time()

    for doc in iterator:
        # --- FIX ---
        # The original code only looked for "id" or "_id".
        # This updated line also checks for "docid", which matches the format
        # in your documents.jsonl file. This ensures the correct ID is used.
        doc_id = doc.get("docid") or doc.get("id") or doc.get("_id") or doc.get("eval_id")

        # If dedupe requested and we've already seen this docid, skip it
        if dedupe and doc_id in seen_docids:
            if verbose:
                print(f"Skipping duplicate docid during ingest: {doc_id}")
            continue

        if dedupe and doc_id is not None and seen_docids is not None:
            seen_docids.add(doc_id)

        action = {"_index": idx, "_id": doc_id, "_source": doc}
        batch.append(action)

        if len(batch) >= batch_size:
            try:
                if not dry_run:
                    if bulk is None:
                        raise ImportError("elasticsearch.helpers.bulk could not be imported")
                    client_with_opts = es.options(request_timeout=30)
                    success, _ = bulk(client_with_opts, batch)
                    indexed += success
                    # clear the batch after successful bulk to avoid re-sending
                    # the same actions in subsequent iterations
                    batch = []
                else:
                    indexed += len(batch)
                    # clear the batch after accounting for dry_run to avoid
                    # double-counting in subsequent iterations
                    batch = []

            except Exception as exc:
                print(f"Bulk indexing failure (batch ending with id={doc_id}): {exc}", flush=True)
    if batch:
        try:
            if not dry_run:
                if bulk is None:
                    raise ImportError("elasticsearch.helpers.bulk could not be imported")
                client_with_opts = es.options(request_timeout=30)
                success, _ = bulk(client_with_opts, batch)
                indexed += success
                batch = []
            else:
                indexed += len(batch)
                batch = []
        except Exception as exc:
            print(f"Final bulk indexing failure: {exc}", flush=True)

    print(f"Indexed {indexed} documents into '{idx}'.", flush=True)
