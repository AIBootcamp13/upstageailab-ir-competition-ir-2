"""Recompute embeddings and bulk-index helper (scaffolding)

This module provides a minimal, safe scaffolding for probing an embedding
dimension and streaming documents to an ES index while computing embeddings
with a project's embedding loader when available.

The implementation is conservative: it tries to import the repository's
embedding helper, falls back to a deterministic zero-vector if not present,
and supports a dry-run mode.
"""
import json
import os
import time
from typing import List, Optional

try:
    import numpy as np
except Exception:
    np = None

try:
    from elasticsearch import Elasticsearch, helpers
except Exception:
    Elasticsearch = None
    helpers = None


def _load_embedding_fn(model: Optional[str] = None):
    """Try to find an embedding function in the repo. Return a callable(batch)->List[List[float]]."""
    # Try common locations used in this repo. This is best-effort scaffolding.
    candidates = [
        'ir_core.embeddings.compute_embeddings',
        'ir_core.embeddings.embed_batch',
        'embeddings.compute_embeddings',
        'embeddings.embed_batch',
    ]
    for cand in candidates:
        parts = cand.split('.')
        try:
            mod = __import__('.'.join(parts[:-1]), fromlist=[parts[-1]])
            fn = getattr(mod, parts[-1], None)
            if callable(fn):
                return fn
        except Exception:
            continue

    # No real embedding function available; return a fallback that yields zeros
    def _fallback(batch: List[dict]):
        dim = 768
        if np is not None:
            return [list(np.zeros(dim).tolist()) for _ in batch]
        else:
            return [[0.0] * dim for _ in batch]

    return _fallback


def probe_embedding_dim(model: Optional[str] = None) -> int:
    """Probe embedding dimension by calling the embedding fn on a tiny sample.

    Returns the detected dimension (int). If the repo embedding loader isn't
    available, returns a conservative 768.
    """
    fn = _load_embedding_fn(model)
    try:
        emb = fn([{"text": "probe"}])
        if emb and isinstance(emb, list) and isinstance(emb[0], (list, tuple)):
            return len(emb[0])
    except Exception:
        pass
    return 768


def stream_and_index(es: str, documents_path: str, index_name: str, batch_size: int = 64, model: Optional[str] = None, dry_run: bool = False):
    """Stream documents from a JSONL file, compute embeddings, and bulk-index to ES.

    This is intentionally simple scaffolding: it uses the embedding function
    when available, otherwise falls back to zero vectors. It writes NDJSON to
    the Elasticsearch _bulk endpoint.
    """
    emb_fn = _load_embedding_fn(model)
    total = 0
    batch = []
    line_no = 0
    if not os.path.exists(documents_path):
        raise FileNotFoundError(documents_path)

    def _flush(b):
        nonlocal total
        if not b:
            return
        ids = [d.get('id') or str(i) for i, d in enumerate(b, start=1)]
        embeddings = list(emb_fn(b))
        if dry_run:
            emb_dim = len(embeddings[0]) if embeddings and isinstance(embeddings[0], (list, tuple)) else 0
            print(f"DRY RUN: would index {len(b)} docs into {index_name} with embedding dim={emb_dim}")
            total += len(b)
            return
        # Preferred: use elasticsearch.helpers.bulk when available
        actions = []
        for doc, emb in zip(b, embeddings):
            body = dict(doc)
            body['embedding'] = emb
            actions.append({
                '_index': index_name,
                '_source': body,
            })

        if helpers is not None and Elasticsearch is not None:
            # Use low-level client for bulk with retries
            client = Elasticsearch([es])
            success, failed = 0, []
            try:
                for ok, item in helpers.streaming_bulk(client, actions, chunk_size=len(actions), max_retries=2, initial_backoff=1):
                    if ok:
                        success += 1
                    else:
                        failed.append(item)
            except Exception as e:
                # Fallback: write all docs to failed file
                print(f"Bulk indexing raised: {e}")
                failed = actions

            # Write failed docs for inspection
            if failed:
                os.makedirs('outputs', exist_ok=True)
                path = os.path.join('outputs', 'failed_docs.jsonl')
                with open(path, 'a', encoding='utf-8') as fh:
                    for f in failed:
                        fh.write(json.dumps(f) + '\n')

            total += success
        else:
            # Fallback: build NDJSON and POST to _bulk
            lines = []
            for a in actions:
                lines.append(json.dumps({'index': {'_index': a['_index']}}))
                lines.append(json.dumps(a['_source']))
            data = "\n".join(lines) + "\n"
            import requests

            r = requests.post(es.rstrip('/') + '/_bulk', data=data.encode('utf-8'), headers={'Content-Type': 'application/x-ndjson'})
            if not (200 <= r.status_code < 300):
                raise RuntimeError(f"Bulk index failed: {r.status_code} {r.text}")
            total += len(b)

    with open(documents_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line_no += 1
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except Exception:
                print(f"Skipping invalid JSON on line {line_no}")
                continue
            batch.append(doc)
            if len(batch) >= batch_size:
                _flush(batch)
                batch = []

    if batch:
        _flush(batch)

    # Refresh the index to make documents visible immediately
    if not dry_run:
        import requests
        requests.post(es.rstrip('/') + f'/{index_name}/_refresh')

    print(f"Indexed {total} documents into {index_name}")
    return total
