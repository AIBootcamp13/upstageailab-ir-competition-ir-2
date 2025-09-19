#!/usr/bin/env python3
"""Tiny smoke test: encode a few synthetic documents and run hybrid retrieval.

This script is intentionally self-contained and does not require a running
Elasticsearch instance: it monkeypatches the `sparse_retrieve` function in
`ir_core.retrieval` to return a synthetic BM25 candidate set, then calls
`hybrid_retrieve` to exercise the reranking logic which depends on the
embeddings implementation.

Run from the `refactor` directory (see README for dependency notes). It
prints shapes and a short summary of results.
"""
from __future__ import annotations

import os
import sys
import json
from typing import List


def _add_src_to_path():
    # Ensure `refactor/src` is importable when running the script directly.
    scripts_dir = os.path.dirname(__file__)
    refactor_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(refactor_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def main():
    _add_src_to_path()

    # Imports from the local package under refactor/src
    from ir_core import embeddings as emb_mod
    from ir_core import retrieval

    print("Running smoke test for embeddings + hybrid retrieval")

    # Sample synthetic documents
    texts: List[str] = [
        "서울의 날씨는 맑고 화창합니다.",
        "파이썬은 인기 있는 프로그래밍 언어입니다.",
        "엘라스틱서치는 분산 검색엔진입니다.",
        "한국의 수도는 서울입니다.",
        "머신러닝과 딥러닝은 관련 분야입니다.",
    ]

    # Encode texts and a query
    print(
        "Encoding sample documents (this will download model weights on first run)..."
    )
    doc_embs = emb_mod.encode_texts(texts, batch_size=2)
    print(f"Encoded {len(texts)} docs -> embeddings shape: {doc_embs.shape}")

    query = "서울의 날씨 알려줘"
    q_emb = emb_mod.encode_query(query)
    print(f"Query embedding dim: {q_emb.shape}")

    # Create a synthetic BM25 hit list (as Elasticsearch would return)
    bm25_hits = []
    for i, t in enumerate(texts):
        bm25_hits.append(
            {
                "_id": str(i),
                "_source": {"content": t},
                "_score": float(1.0 / (i + 1)),
            }
        )

    # Monkeypatch the sparse_retrieve used by hybrid_retrieve to return our synthetic hits
    retrieval.sparse_retrieve = lambda q, size=10, index=None: bm25_hits

    # Run hybrid retrieve (it will call encode_texts on the BM25 texts internally)
    print("Running hybrid_retrieve (uses synthetic BM25 candidates)...")
    results = retrieval.hybrid_retrieve(
        query, bm25_k=len(bm25_hits), rerank_k=5, alpha=None
    )

    # Print a compact summary
    print("Hybrid re-ranking results (top K):")
    for r in results:
        hit = r["hit"] if isinstance(r, dict) else r.get("hit")
        # ensure compatibility if retrieval returns list of dicts or hybrid format
        if isinstance(r, dict):
            cosine = r.get("cosine")
            score = r.get("score")
            print(
                f"id={hit.get('_id')} score={score:.4f} cosine={cosine:.4f} text={hit['_source']['content']}"
            )
        else:
            print(json.dumps(r, ensure_ascii=False))

    print("Smoke test completed.")


if __name__ == "__main__":
    main()
