#!/usr/bin/env python3
"""
Re-indexes documents from a JSONL file into Elasticsearch,
computing and adding embeddings based on the current configuration.
"""
import sys
from pathlib import Path
from tqdm import tqdm
from elasticsearch.helpers import bulk

# Add src to python path to allow for project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ir_core.config import settings
from ir_core.embeddings import get_embedding_provider
from ir_core.infra import get_es
from ir_core.utils import read_jsonl

def reindex_with_embeddings(jsonl_path: str, index_name: str, batch_size: int = 64):
    """
    Reads documents, computes embeddings, and bulk-indexes them.

    Args:
        jsonl_path: Path to the input JSONL file.
        index_name: Name of the target Elasticsearch index.
        batch_size: Batch size for encoding and indexing.
    """
    try:
        es_client = get_es()
        if not es_client.ping():
            raise ConnectionError("Could not connect to Elasticsearch")
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        return

    # Initialize the embedding provider from the current settings
    print(f"▶️ Using embedding model: {settings.EMBEDDING_MODEL}")
    provider = get_embedding_provider()

    docs = list(read_jsonl(jsonl_path))
    total_docs = len(docs)
    print(f"✅ Found {total_docs} documents to process in {jsonl_path}")

    # Process documents in batches
    for i in tqdm(range(0, total_docs, batch_size), desc=f"Indexing to '{index_name}'"):
        batch_docs = docs[i : i + batch_size]

        # 1. Extract content to be encoded
        contents = [doc.get("content", "") for doc in batch_docs]

        # 2. Compute embeddings for the batch
        embeddings = provider.encode_texts(contents)

        # 3. Add embeddings to each document
        for doc, embedding in zip(batch_docs, embeddings):
            doc["embeddings"] = embedding.tolist()

        # 4. Prepare bulk actions for Elasticsearch
        actions = [
            {
                "_index": index_name,
                "_id": doc.get("docid") or doc.get("id"),
                "_source": doc,
            }
            for doc in batch_docs
        ]

        # 5. Execute bulk indexing
        try:
            success, failed = bulk(es_client, actions, raise_on_error=True)
        except Exception as e:
            print(f"\n❌ Error during bulk indexing: {e}")
            break

    print(f"\n✅ Successfully indexed {total_docs} documents with embeddings into '{index_name}'.")

if __name__ == "__main__":
    import fire
    fire.Fire(reindex_with_embeddings)
