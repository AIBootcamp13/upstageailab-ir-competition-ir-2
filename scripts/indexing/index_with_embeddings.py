#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'src')

import json
import numpy as np
from ir_core.embeddings.core import encode_texts
from ir_core.infra import get_es
from elasticsearch.helpers import bulk
from ir_core.utils.core import read_jsonl

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/indexing/index_with_embeddings.py <jsonl_file> <index_name>")
        sys.exit(1)

    jsonl_file = sys.argv[1]
    index_name = sys.argv[2]

    print(f"Indexing {jsonl_file} with embeddings to {index_name}")

    es = get_es()
    batch = []
    batch_size = 32

    for doc in read_jsonl(jsonl_file):
        # Generate embedding for the document content
        content = doc.get('content', '')
        if content:
            embedding = encode_texts([content])[0]

            # Validate embedding for NaN/inf values
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                print(f"⚠️  Invalid embedding detected for doc {doc.get('docid', doc.get('id'))}, skipping...")
                continue

            doc['embeddings'] = embedding.tolist()
            doc['embedding_dim'] = len(embedding)
            doc['embedding_model'] = 'jhgan/ko-sbert-sts'

        # Prepare for bulk indexing
        doc_id = doc.get('docid') or doc.get('id')
        action = {
            '_index': index_name,
            '_id': doc_id,
            '_source': doc
        }
        batch.append(action)

        if len(batch) >= batch_size:
            success, _ = bulk(es, batch)
            print(f"Indexed {success} documents")
            batch = []

    # Index remaining documents
    if batch:
        success, _ = bulk(es, batch)
        print(f"Indexed {success} documents")

    print("Indexing complete")

if __name__ == "__main__":
    main()
