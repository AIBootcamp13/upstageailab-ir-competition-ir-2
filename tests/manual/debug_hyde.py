#!/usr/bin/env python3
"""Debug HyDE retrieval for the failing query."""

import os
import sys

# Add src to path
scripts_dir = os.path.dirname(__file__)
repo_dir = os.path.dirname(scripts_dir)
src_dir = os.path.join(repo_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ir_core.query_enhancement.hyde import HyDE

def debug_hyde():
    hyde = HyDE()

    query = "기억 상실증 원인"

    print(f"Query: {query}")
    print("=" * 50)

    # Generate hypothetical answer
    hypothetical_answer = hyde.generate_hypothetical_answer(query)
    print(f"Hypothetical Answer:\n{hypothetical_answer}")
    print("=" * 50)

    # Get HyDE embedding
    hyde_embedding = hyde.get_hyde_embedding(query)
    print(f"HyDE embedding shape: {hyde_embedding.shape if hyde_embedding is not None else 'None'}")
    print("=" * 50)

    # Retrieve with HyDE
    results = hyde.retrieve_with_hyde(query, top_k=5)
    print(f"Retrieved {len(results)} documents:")
    for i, result in enumerate(results):
        print(f"\nRank {i+1}:")
        print(f"  ID: {result['id']}")
        print(f"  Score: {result['score']}")
        print(f"  Content: {result['content'][:200]}...")

if __name__ == "__main__":
    debug_hyde()