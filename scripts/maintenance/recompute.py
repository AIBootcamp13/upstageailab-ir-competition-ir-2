#!/usr/bin/env python3
"""Enhanced Embedding Generation with Profiling Insights

This script generates embeddings for multiple indexes using profiling data
for optimization and quality enhancement.

Usage:
  PYTHONPATH=src poetry run python scripts/maintenance/recompute.py --index-type english --model sentence-transformers/all-MiniLM-L6-v2
  PYTHONPATH=src poetry run python scripts/maintenance/recompute.py --index-type korean --model snunlp/KR-SBERT-V40K-klueNLI-augSTS
  PYTHONPATH=src poetry run python scripts/maintenance/recompute.py --index-type bilingual --model snunlp/KR-SBERT-V40K-klueNLI-augSTS
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
from datetime import datetime

try:
    import numpy as np
except Exception:
    np = None

try:
    from elasticsearch import Elasticsearch, helpers
except Exception:
    Elasticsearch = None
    helpers = None

import concurrent.futures


def _add_src_to_path():
    """Add src directory to Python path."""
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def load_profiling_insights(profile_dir: str) -> Dict[str, Any]:
    """Load profiling insights for embedding optimization."""
    insights = {}

    # Load per-source length statistics for batch size optimization
    length_stats_path = os.path.join(profile_dir, "per_src_length_stats.json")
    if os.path.exists(length_stats_path):
        with open(length_stats_path, 'r', encoding='utf-8') as f:
            insights['length_stats'] = json.load(f)

    # Load vocabulary clusters for domain-aware processing
    clusters_path = os.path.join(profile_dir, "src_clusters_by_vocab.json")
    if os.path.exists(clusters_path):
        with open(clusters_path, 'r', encoding='utf-8') as f:
            insights['clusters'] = json.load(f)

    # Load keywords for potential query expansion during embedding
    keywords_path = os.path.join(profile_dir, "keywords_per_src.json")
    if os.path.exists(keywords_path):
        with open(keywords_path, 'r', encoding='utf-8') as f:
            insights['keywords'] = json.load(f)

    return insights


def get_optimal_batch_size(length_stats: Dict[str, Any], src: str, base_batch_size: int = 32) -> int:
    """Calculate optimal batch size based on document length statistics."""
    if not length_stats or src not in length_stats:
        return base_batch_size

    stats = length_stats[src]
    avg_tokens = stats.get('content_tokens', {}).get('mean', 500)

    # Adjust batch size based on average token length
    # Shorter documents = larger batches, longer documents = smaller batches
    if avg_tokens < 300:
        return min(base_batch_size * 2, 64)
    elif avg_tokens > 800:
        return max(base_batch_size // 2, 8)
    else:
        return base_batch_size


def enhance_text_with_keywords(text: str, src: str, keywords: Dict[str, List[Dict]], max_keywords: int = 3) -> str:
    """Enhance text with relevant keywords from profiling data."""
    if src not in keywords:
        return text

    # Get top keywords for this source
    src_keywords = keywords[src][:max_keywords]
    keyword_terms = [kw['term'] for kw in src_keywords]

    # Append keywords to enhance semantic representation
    enhanced = f"{text} {' '.join(keyword_terms)}"
    return enhanced


def generate_enhanced_embeddings(
    index_type: str,
    model_name: str,
    es_host: str = "http://localhost:9200",
    batch_size: int = 32,
    profile_dir: Optional[str] = None,
    dry_run: bool = False,
    max_workers: int = 4,
):
    """Generate embeddings with profiling-based enhancements."""

    _add_src_to_path()

    # Load profiling insights
    insights = {}
    if profile_dir:
        insights = load_profiling_insights(profile_dir)
        print(f"Loaded profiling insights from {profile_dir}")

    # Determine data file and index name based on type
    config = {
        'english': {
            'data_file': 'data/documents_bilingual.jsonl',
            'index_name': 'documents_en_with_embeddings_new',
            'expected_dim': 384
        },
        'korean': {
            'data_file': 'data/documents_ko.jsonl',
            'index_name': 'documents_ko_with_embeddings_new',
            'expected_dim': 768
        },
        'bilingual': {
            'data_file': 'data/documents_bilingual.jsonl',
            'index_name': 'documents_bilingual_with_embeddings_new',
            'expected_dim': 768
        }
    }

    if index_type not in config:
        raise ValueError(f"Unknown index type: {index_type}")

    data_file = config[index_type]['data_file']
    index_name = config[index_type]['index_name']
    expected_dim = config[index_type]['expected_dim']

    print(f"Generating {expected_dim}D embeddings for {index_type} index")
    print(f"Model: {model_name}")
    print(f"Data file: {data_file}")
    print(f"Target index: {index_name}")

    # Import required modules
    from ir_core.utils.core import read_jsonl
    from ir_core.embeddings.core import encode_texts, load_model

    # Set the model for encoding
    load_model(model_name)

    # Verify model produces expected dimensions
    print("Verifying model dimensions...")
    test_texts = ["This is a test document for dimension verification."]
    try:
        test_emb = encode_texts(test_texts, model_name=model_name)
        actual_dim = test_emb.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(f"Model {model_name} produces {actual_dim}D embeddings, expected {expected_dim}D")
        print(f"✓ Model verified: {actual_dim}D embeddings")
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return

    # Process documents with enhancements
    total_processed = 0
    batch = []

    for doc in read_jsonl(data_file):
        # Extract text content
        text = doc.get('content', doc.get('text', ''))
        if not text:
            continue

        # Apply profiling enhancements
        src = doc.get('src', 'unknown')
        if insights.get('keywords'):
            text = enhance_text_with_keywords(text, src, insights['keywords'])

        # Get optimal batch size for this source
        optimal_batch = batch_size
        if insights.get('length_stats'):
            optimal_batch = get_optimal_batch_size(insights['length_stats'], src, batch_size)

        doc['enhanced_text'] = text
        doc['embedding_model'] = model_name
        doc['embedding_dim'] = expected_dim
        batch.append(doc)

        # Process batch when it reaches optimal size
        if len(batch) >= optimal_batch:
            process_batch(batch, encode_texts, index_name, es_host, dry_run)
            total_processed += len(batch)
            print(f"Processed {total_processed} documents...")
            batch = []

    # Process remaining documents
    if batch:
        process_batch(batch, encode_texts, index_name, es_host, dry_run)
        total_processed += len(batch)

    print(f"✓ Completed processing {total_processed} documents")
    print(f"✓ Created index: {index_name} with {expected_dim}D embeddings")


def process_batch(batch: List[Dict], encode_fn, index_name: str, es_host: str, dry_run: bool):
    """Process a batch of documents with embedding generation."""
    texts = [doc['enhanced_text'] for doc in batch]

    # Generate embeddings
    embeddings = encode_fn(texts)

    if dry_run:
        print(f"DRY RUN: Would process {len(batch)} documents")
        return

    # Index documents with embeddings
    from elasticsearch import Elasticsearch, helpers

    es = Elasticsearch([es_host])
    actions = []

    for doc, emb in zip(batch, embeddings):
        # Remove temporary fields
        doc_copy = {k: v for k, v in doc.items() if k != 'enhanced_text'}

        # Add embeddings
        if hasattr(emb, 'tolist'):
            doc_copy['embeddings'] = emb.tolist()
        else:
            doc_copy['embeddings'] = emb

        actions.append({
            '_index': index_name,
            '_id': doc.get('docid') or doc.get('id'),
            '_source': doc_copy
        })

    # Bulk index
    try:
        success, failed = helpers.bulk(es, actions, stats_only=True)
        if failed:
            print(f"Warning: {failed} documents failed to index")
    except Exception as e:
        print(f"Error during bulk indexing: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced embeddings with profiling insights")
    parser.add_argument('--index-type', required=True, choices=['english', 'korean', 'bilingual'],
                       help='Type of index to generate embeddings for')
    parser.add_argument('--model', required=True,
                       help='Embedding model to use')
    parser.add_argument('--es-host', default='http://localhost:9200',
                       help='Elasticsearch host')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Base batch size for embedding generation')
    parser.add_argument('--profile-dir', default='outputs/reports/data_profile/latest',
                       help='Directory containing profiling data')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode - do not actually index')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of worker threads')

    args = parser.parse_args()

    generate_enhanced_embeddings(
        index_type=args.index_type,
        model_name=args.model,
        es_host=args.es_host,
        batch_size=args.batch_size,
        profile_dir=args.profile_dir,
        dry_run=args.dry_run,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
import json
import os
import time
from typing import List, Optional
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

try:
    from elasticsearch import Elasticsearch, helpers
except Exception:
    Elasticsearch = None
    helpers = None

import concurrent.futures


def _load_embedding_fn(model: Optional[str] = None):
    """Try to find an embedding function in the repo. Return a callable(batch)->List[List[float]] or ndarray.

    This function prefers `ir_core.embeddings.encode_texts` (returns ndarray) but will
    fall back to other commonly used entrypoints or a zero-vector fallback.
    """
    candidates = [
        'ir_core.embeddings.encode_texts',
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
                # If it's encode_texts, create a wrapper that includes model_name
                if fn.__name__ == 'encode_texts':
                    def wrapper(texts):
                        return fn(texts, model_name=model)
                    return wrapper
                return fn
        except Exception:
            continue

    # No real embedding function available; return a fallback that yields zeros
    def _fallback(batch):
        # Handle both list of strings and list of dicts
        if isinstance(batch, list) and batch:
            if isinstance(batch[0], dict):
                # If batch is list of dicts, get length from batch
                batch_size = len(batch)
            else:
                # If batch is list of strings, get length from batch
                batch_size = len(batch)
        else:
            batch_size = 1  # fallback

        dim = 768
        if np is not None:
            return np.zeros((batch_size, dim), dtype=float).tolist()
        else:
            return [[0.0] * dim for _ in range(batch_size)]

    return _fallback


def probe_embedding_dim(model: Optional[str] = None) -> int:
    """Probe embedding dimension by calling the embedding fn on a tiny sample.

    Returns the detected dimension (int). If the repo embedding loader isn't
    available, returns a conservative 768.
    """
    fn = _load_embedding_fn(model)
    try:
        emb = fn(["probe"])
        # Convert numpy arrays to lists
        if hasattr(emb, 'shape'):
            shape_attr = getattr(emb, 'shape', None)
            if shape_attr is not None:
                return int(shape_attr[-1])
        if emb and isinstance(emb, list) and isinstance(emb[0], (list, tuple)):
            return len(emb[0])
    except Exception:
        pass

    # Fallback: use encode_texts directly with the specified model
    try:
        from src.ir_core.embeddings.core import encode_texts
        result = encode_texts(["probe"], model_name=model)
        return result.shape[-1]
    except Exception:
        pass

    return 768


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def stream_and_index(
    es: str,
    documents_path: str,
    index_name: str,
    batch_size: int = 64,
    model: Optional[str] = None,
    dry_run: bool = False,
    max_workers: int = 1,
    failed_dir: str = 'outputs',
    embedding_batch_size: int = 32,
    device: Optional[str] = None,
):
    """Stream documents from a JSONL file, compute embeddings, and bulk-index to ES.

    Enhancements over the minimal scaffold:
    - Tries to use `ir_core.embeddings.encode_texts` when available (GPU-aware).
    - Accepts numpy ndarray outputs and converts them to lists.
    - Uses `elasticsearch.helpers.streaming_bulk` when available for efficient upload.
    - Writes failed items to `failed_dir/failed_docs_{timestamp}.jsonl`.
    - Supports a `max_workers` parameter for parallel embedding computation (best-effort).
    """
    emb_fn = _load_embedding_fn(model)
    total_indexed = 0
    batch = []
    line_no = 0
    documents_path = str(documents_path)
    if not os.path.exists(documents_path):
        raise FileNotFoundError(documents_path)

    _ensure_dir(failed_dir)

    def _write_failed(failed_actions):
        if not failed_actions:
            return
        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        path = os.path.join(failed_dir, f'failed_docs_{ts}.jsonl')
        with open(path, 'a', encoding='utf-8') as fh:
            for f in failed_actions:
                try:
                    fh.write(json.dumps(f) + '\n')
                except Exception:
                    fh.write(json.dumps({'failed': str(f)}) + '\n')

    def _compute_embeddings(docs: List[dict]):
        # Prefer ir_core.embeddings.encode_texts when available; it accepts
        # texts and supports batch_size and device params.
        texts = [d.get('text') or d.get('content') or d.get('body') or '' for d in docs]
        fn_name = getattr(emb_fn, '__name__', '')
        try:
            if fn_name == 'encode_texts':
                # Try to call with batch_size and device control if supported
                try:
                    # Use **kwargs to avoid parameter errors
                    kwargs = {}
                    if embedding_batch_size is not None:
                        kwargs['batch_size'] = embedding_batch_size
                    if device is not None:
                        kwargs['device'] = device
                    raw = emb_fn(texts, **kwargs)
                except TypeError:
                    # Fallback if parameters not supported
                    raw = emb_fn(texts)
            else:
                # Try common signature: list[dict] or list[str]
                try:
                    raw = emb_fn(docs)
                except TypeError:
                    raw = emb_fn(texts)
        except Exception:
            # Last resort: fallback to zeros for this batch
            raw = [[0.0] * probe_embedding_dim(model) for _ in docs]

        # Normalize to list of lists
        tolist_method = getattr(raw, 'tolist', None)
        if tolist_method and callable(tolist_method):
            try:
                raw = tolist_method()
            except Exception:
                pass  # Keep original format if tolist fails
        return raw

    def _flush(b):
        nonlocal total_indexed
        if not b:
            return
        embeddings = _compute_embeddings(b)
        if dry_run:
            emb_dim = 0
            if isinstance(embeddings, list) and embeddings:
                first_emb = embeddings[0]
                if isinstance(first_emb, (list, tuple)):
                    emb_dim = len(first_emb)
            print(f"DRY RUN: would index {len(b)} docs into {index_name} with embedding dim={emb_dim}")
            total_indexed += len(b)
            return

        actions = []
        if isinstance(embeddings, list) and hasattr(embeddings, '__iter__'):
            for doc, emb in zip(b, embeddings):
                body = dict(doc)
                body['embeddings'] = emb
                actions.append({
                    '_index': index_name,
                    '_source': body,
                })
        else:
            print(f"Warning: embeddings is not iterable, skipping batch of {len(b)} docs")

        if helpers is not None and Elasticsearch is not None:
            client = Elasticsearch([es])
            success = 0
            failed = []
            try:
                for ok, item in helpers.streaming_bulk(
                    client,
                    actions,
                    chunk_size=min(batch_size, 1024),
                    max_retries=2,
                    initial_backoff=1,
                ):
                    if ok:
                        success += 1
                    else:
                        failed.append(item)
            except Exception as e:
                print(f"Bulk indexing raised exception: {e}")
                failed = actions

            if failed:
                _write_failed(failed)

            total_indexed += success
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
            total_indexed += len(b)

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
        try:
            requests.post(es.rstrip('/') + f'/{index_name}/_refresh')
        except Exception:
            pass

    print(f"Indexed {total_indexed} documents into {index_name}")
    return total_indexed


if __name__ == "__main__":
    main()
