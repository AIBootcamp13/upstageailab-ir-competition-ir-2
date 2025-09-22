#!/usr/bin/env python3
"""
Create an Elasticsearch index with a proper dense_vector mapping for 'embeddings'.

Usage:
  uv run python scripts/indexing/create_index_with_mapping.py --index <name> --dims <int> [--overwrite]

This script ensures the 'embeddings' field can be used with cosineSimilarity in script_score queries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
from ir_core.infra import get_es


def create_index(index: str, dims: int, overwrite: bool = False, use_nori: bool = True, korean_fields: str = "content,summary,keywords,hypothetical_questions,title"):
    es = get_es()

    if overwrite and es.indices.exists(index=index):
        print(f"Deleting existing index: {index}")
        try:
            es.indices.delete(index=index)
        except Exception as e:
            print(f"Warning: failed to delete index '{index}': {e}")

        # Re-check if the index still exists after deletion attempt
        if es.indices.exists(index=index):
            print(f"Index '{index}' still exists after deletion attempt. Aborting creation.")
            mapping = es.indices.get_mapping(index=index)
            print(mapping.get(index, {}).get('mappings', {}))
            return

    if es.indices.exists(index=index):
        print(f"Index already exists: {index}")
        # Print current mapping for visibility
        mapping = es.indices.get_mapping(index=index)
        current_mapping = mapping.get(index, {}).get('mappings', {}).get('properties', {})
        print("Current mapping:")
        print(current_mapping)

        # Define the expected mapping for comparison
        expected_mapping = {
            "docid": {"type": "keyword"},
            "id": {"type": "keyword"},
            "content": {"type": "text"},
            "summary": {"type": "text"},
            "keywords": {"type": "text"},
            "hypothetical_questions": {"type": "text"},
            "src": {"type": "keyword"},
            "title": {"type": "text"},
            "embeddings": {"type": "dense_vector", "dims": dims}
        }

        # Check for mismatches in mapping keys or types
        mismatches = []
        for field, expected in expected_mapping.items():
            if field not in current_mapping:
                mismatches.append(f"Missing field: {field}")
            elif field == "embeddings":
                # Special check for embeddings field
                if current_mapping[field].get("type") != expected.get("type"):
                    mismatches.append(f"Field '{field}' type mismatch: expected {expected.get('type')}, got {current_mapping[field].get('type')}")
                elif current_mapping[field].get("dims") != expected.get("dims"):
                    mismatches.append(f"Field '{field}' dims mismatch: expected {expected.get('dims')}, got {current_mapping[field].get('dims')}")
            elif current_mapping[field].get("type") != expected.get("type"):
                mismatches.append(f"Field '{field}' type mismatch: expected {expected.get('type')}, got {current_mapping[field].get('type')}")

        if mismatches:
            print("WARNING: Index mapping does not match expected mapping!")
            for msg in mismatches:
                print("  -", msg)
            print("Consider deleting and recreating the index, or updating the mapping manually.")
        else:
            print("Index mapping matches expected mapping.")

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
                "embeddings": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine"
                },
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

        # Determine which fields should use the Korean analyzer
        korean_fields_set = set(f.strip() for f in korean_fields.split(",") if f.strip())

        for field in korean_fields_set:
            if field in body["mappings"]["properties"]:
                body["mappings"]["properties"][field]["analyzer"] = "korean"

    try:
        es.indices.create(index=index, body=body)
        print(f"Created index '{index}' with embeddings dims={dims}")
    except Exception as e:
        error_msg = str(e).lower()
        if "resource_already_exists_exception" in error_msg:
            print(f"Index '{index}' was created by another process. Checking mapping...")
            # Re-run the mapping validation logic
            mapping = es.indices.get_mapping(index=index)
            current_mapping = mapping.get(index, {}).get('mappings', {}).get('properties', {})

            # Check embeddings field specifically
            embeddings_field = current_mapping.get('embeddings', {})
            if embeddings_field.get('type') != 'dense_vector' or embeddings_field.get('dims') != dims:
                print(f"WARNING: Existing index '{index}' has incompatible embeddings configuration!")
                print(f"Expected: type=dense_vector, dims={dims}")
                print(f"Found: {embeddings_field}")
            else:
                print(f"Index '{index}' already exists with compatible configuration.")
        elif "nori_tokenizer" in error_msg or "nori" in error_msg:
            print(f"WARNING: Nori tokenizer not available in Elasticsearch. Creating index without Korean analyzer.")
            print("To use Korean text analysis, install the Elasticsearch Nori plugin.")

            # Remove nori-specific settings and try again
            if "analysis" in body.get("settings", {}):
                del body["settings"]["analysis"]

            # Remove analyzer assignments from text fields
            for field in ["content", "summary", "keywords", "hypothetical_questions", "title"]:
                if field in body["mappings"]["properties"] and "analyzer" in body["mappings"]["properties"][field]:
                    del body["mappings"]["properties"][field]["analyzer"]

            try:
                es.indices.create(index=index, body=body)
                print(f"Created index '{index}' with embeddings dims={dims} (without Korean analyzer)")
            except Exception as e2:
                print(f"Error creating index '{index}' even without analyzer: {e2}")
                raise
        else:
            print(f"Error creating index '{index}': {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Create ES index with dense_vector mapping")
    parser.add_argument("--index", required=True, help="Index name")
    parser.add_argument("--dims", required=True, type=int, help="Embedding dimensions")
    parser.add_argument("--overwrite", action="store_true", help="Delete index if exists")
    parser.add_argument("--no-nori", action="store_true", help="Do not use Nori analyzer")
    parser.add_argument(
        "--korean-fields",
        type=str,
        default="content,summary,keywords,hypothetical_questions,title",
        help="Comma-separated list of fields to apply the Korean analyzer to (default: all text fields)",
    )
    args = parser.parse_args()

    create_index(args.index, args.dims, args.overwrite, use_nori=not args.no_nori, korean_fields=args.korean_fields)


if __name__ == "__main__":
    main()
