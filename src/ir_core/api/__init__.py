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


def index_documents_from_jsonl(jsonl_path, index_name=None, batch_size: int = 500, *, dry_run: bool = False, verbose: bool = False):
    """Index documents from a JSONL file using the bulk API with progress.

    Uses `tqdm` when available for a nice progress bar. Falls back to a
    lightweight heartbeat print every `flush_every` documents.

    Params:
        jsonl_path: path to JSONL file where each line is a JSON object.
        index_name: optional index override.
        batch_size: number of docs to send per bulk request.
    """
    es = get_es()
    idx = index_name or settings.INDEX_NAME

    # Count total lines to provide an accurate tqdm total. This is an
    # extra pass over the file but keeps the UX pleasant for large files.
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
    except Exception:
        total = None

    # Keep a typed reference for linters: tqdm may be unavailable at import time
    from typing import Optional, Callable, Iterable
    tqdm: Optional[Callable[..., Iterable]] = None
    use_tqdm = False
    try:
        # import into a temporary name then assign so the type is clear
        from tqdm import tqdm as _tqdm
        tqdm = _tqdm
        use_tqdm = True
    except Exception:
        tqdm = None

    flush_every = 100
    indexed = 0

    # Helper to print lightweight heartbeat for non-tqdm mode
    def heartbeat(n):
        if n % flush_every == 0:
            print(f"Indexed {n} documents...", flush=True)

    # Import bulk lazily to avoid editor/analysis errors when the
    # elasticsearch package isn't available in the environment used by
    # static analysis tools. This is still an error at runtime if the
    # package is missing.
    try:
        from elasticsearch.helpers import bulk
    except Exception:
        bulk = None

    # Build and send batches using helpers.bulk
    batch = []
    iterator = read_jsonl(jsonl_path)

    if use_tqdm:
        # Help static type-checkers: ensure tqdm is not None here
        assert tqdm is not None
        iterator = tqdm(iterator, total=total, desc=f"Indexing -> {idx}")

    import time
    start_time = time.time()
    last_batch_time = start_time

    for doc in iterator:
        doc_id = doc.get("id") or doc.get("_id")
        action = {"_index": idx, "_id": doc_id, "_source": doc}
        batch.append(action)

        if len(batch) >= batch_size:
            try:
                batch_start = time.time()
                if dry_run:
                    # simulate successful indexing without touching ES
                    success = len(batch)
                    errors = []
                    indexed += success
                else:
                    if bulk is None:
                        # Fall back to per-doc indexing if helpers.bulk isn't available.
                        for a in batch:
                            try:
                                es.index(index=a["_index"], id=a.get("_id"), document=a.get("_source"))
                                indexed += 1
                            except Exception as exc:
                                print(f"Failed to index doc id={a.get('_id')}: {exc}", flush=True)
                    else:
                        # Use Elasticsearch.options() to pass transport options per the new API
                        client_with_opts = es.options(request_timeout=30)
                        success, _ = bulk(client_with_opts, batch)
                        indexed += success
                batch_end = time.time()
                last_batch_time = batch_end - batch_start
                # Verbose per-batch timing/ETA
                if verbose:
                    elapsed = batch_end - start_time
                    rate = indexed / elapsed if elapsed > 0 else 0
                    remaining = (total - indexed) if total is not None else None
                    eta = None
                    if remaining is not None and rate > 0:
                        eta = remaining / rate
                    eta_s = f" ETA ~ {eta:.1f}s" if eta is not None else ""
                    print(f"Batch indexed {len(batch)} docs in {last_batch_time:.2f}s; total indexed={indexed}.{eta_s}", flush=True)
            except Exception as exc:
                print(f"Bulk indexing failure (batch ending with id={doc_id}): {exc}", flush=True)
                # continue after logging; items in batch may or may not have been indexed
            batch = []
            if not use_tqdm:
                heartbeat(indexed)

    # Flush remaining batch
    if batch:
        try:
            batch_start = time.time()
            if dry_run:
                success = len(batch)
                indexed += success
            else:
                if bulk is None:
                    for a in batch:
                        try:
                            es.index(index=a["_index"], id=a.get("_id"), document=a.get("_source"))
                            indexed += 1
                        except Exception as exc:
                            print(f"Failed to index doc id={a.get('_id')}: {exc}", flush=True)
                else:
                    client_with_opts = es.options(request_timeout=30)
                    success, _ = bulk(client_with_opts, batch)
                    indexed += success
            batch_end = time.time()
            last_batch_time = batch_end - batch_start
            if verbose:
                elapsed = batch_end - start_time
                rate = indexed / elapsed if elapsed > 0 else 0
                remaining = (total - indexed) if total is not None else None
                eta = None
                if remaining is not None and rate > 0:
                    eta = remaining / rate
                eta_s = f" ETA ~ {eta:.1f}s" if eta is not None else ""
                print(f"Final batch indexed {len(batch)} docs in {last_batch_time:.2f}s; total indexed={indexed}.{eta_s}", flush=True)
        except Exception as exc:
            print(f"Final bulk indexing failure: {exc}", flush=True)

    # If tqdm was used, it already tracked progress; ensure final count printed
    print(f"Indexed {indexed} documents into '{idx}'.", flush=True)
