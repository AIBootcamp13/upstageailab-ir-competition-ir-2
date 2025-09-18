#!/usr/bin/env python3
"""
Create an Elasticsearch index with a proper dense_vector mapping for 'embeddings'.

Usage:
  poetry run python scripts/indexing/create_index_with_mapping.py --index <name> --dims <int> [--overwrite]

This script ensures the 'embeddings' field can be used with cosineSimilarity in script_score queries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
from ir_core.infra import get_es


def create_index(index: str, dims: int, overwrite: bool = False, use_nori: bool = True):
    es = get_es()

    if overwrite and es.indices.exists(index=index):
        print(f"Deleting existing index: {index}")
        try:
            es.indices.delete(index=index)
        except Exception as e:
            print(f"Warning: failed to delete index '{index}': {e}")

    if es.indices.exists(index=index):
        print(f"Index already exists: {index}")
        # Print current mapping for visibility
        mapping = es.indices.get_mapping(index=index)
        print(mapping.get(index, {}).get('mappings', {}))
        return

    body = {
        "settings": {},
        "mappings": {
            "properties": {
                # ID fields
                "docid": {"type": "keyword"},
                "id": {"type": "keyword"},

                # Content fields used by BM25 (with Nori analyzer for Korean)
                "content": {"type": "text"},
                "summary": {"type": "text"},
                "keywords": {"type": "text"},
                "hypothetical_questions": {"type": "text"},

                # Metadata
                "src": {"type": "keyword"},
                "title": {"type": "text"},

                # Dense vector for ANN scoring
                "embeddings": {"type": "dense_vector", "dims": dims},
            }
        }
    }

    if use_nori:
        # Define an enhanced Nori-based analyzer with compound decomposition and POS filtering
        body["settings"]["analysis"] = {
            "tokenizer": {
                "korean_tokenizer": {
                    "type": "nori_tokenizer",
                    # decompound_mode: none|discard|mixed (mixed keeps both compound and decomposed forms)
                    "decompound_mode": "mixed",
                    # user_dictionary can be added in the future if we curate domain-specific terms
                }
            },
            "filter": {
                # Remove punctuation/symbols and optionally specific POS
                "korean_pos_filter": {
                    "type": "nori_part_of_speech",
                    # Exclude less informative POS; keep nouns, verbs, adjectives primarily
                    "stoptags": [
                        "E",  # ending
                        "IC", # interjection
                        "J",  # particle
                        "MAG","MAJ", # adverbs
                        "MM", # determiner
                        "SP","SSC","SSO","SC","SE", # punctuation/symbols
                        "XPN","XSN","XSV","XSA" # affixes
                    ]
                },
                # Normalize to reading form (useful for some scripts/variants)
                "korean_readingform": {
                    "type": "nori_readingform"
                },
                "lowercase": {"type": "lowercase"}
            },
            "analyzer": {
                "korean": {
                    "type": "custom",
                    "tokenizer": "korean_tokenizer",
                    "filter": ["lowercase", "korean_pos_filter", "korean_readingform"],
                }
            }
        }
        for field in ("content", "summary", "keywords", "hypothetical_questions", "title"):
            body["mappings"]["properties"][field]["analyzer"] = "korean"

    es.indices.create(index=index, body=body)
    print(f"Created index '{index}' with embeddings dims={dims}")


def main():
    parser = argparse.ArgumentParser(description="Create ES index with dense_vector mapping")
    parser.add_argument("--index", required=True, help="Index name")
    parser.add_argument("--dims", required=True, type=int, help="Embedding dimensions")
    parser.add_argument("--overwrite", action="store_true", help="Delete index if exists")
    parser.add_argument("--no-nori", action="store_true", help="Do not use Nori analyzer")
    args = parser.parse_args()

    create_index(args.index, args.dims, args.overwrite, use_nori=not args.no_nori)


if __name__ == "__main__":
    main()
