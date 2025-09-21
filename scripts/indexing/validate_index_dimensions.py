#!/usr/bin/env python3
"""
Pre-flight validator: checks that the embedding provider's dimension matches the
Elasticsearch index mapping for the `embeddings` dense_vector field. Also can
verify that Nori analyzer is configured for Korean text fields.

Usage:
  PYTHONPATH=src uv run python scripts/indexing/validate_index_dimensions.py \
    --index <INDEX_NAME> [--provider <auto|polyglot|huggingface|solar|sentence_transformers>] [--expect-dims <int>] [--check-analyzer]

Exit codes:
  0 - OK
  1 - Mismatch or validation failure
  2 - Runtime error (ES unreachable, etc.)
"""
import sys
from pathlib import Path

# Ensure src is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
from typing import Optional

from ir_core.infra import get_es
from ir_core.config import settings


def get_provider_dim(provider_name: Optional[str]) -> int:
    """Determine expected embedding dimension without loading heavy models.

    Priority:
    1) settings.EMBEDDING_DIMENSION
    2) Known mappings for Polyglot-Ko model name
    3) Error
    """
    # 1) Use settings if present
    dim = int(getattr(settings, "EMBEDDING_DIMENSION", 0) or 0)
    if dim > 0:
        return dim

    # 2) Known mappings for Polyglot-Ko
    model_name = getattr(settings, 'POLYGLOT_MODEL', None)
    polyglot_map = {
        'EleutherAI/polyglot-ko-1.3b': 2048,
        'EleutherAI/polyglot-ko-3.8b': 3072,
        'EleutherAI/polyglot-ko-5.8b': 4096,
        'EleutherAI/polyglot-ko-12.8b': 5120,
    }
    if model_name in polyglot_map:
        return polyglot_map[model_name]

    raise RuntimeError("Unable to determine embedding dimension from settings; set EMBEDDING_DIMENSION or provide --expect-dims")


def get_index_dims(es, index: str) -> Optional[int]:
    mapping = es.indices.get_mapping(index=index)
    props = mapping.get(index, {}).get("mappings", {}).get("properties", {})
    emb = props.get("embeddings")
    if not emb:
        return None
    if emb.get("type") != "dense_vector":
        raise RuntimeError(
            f"Index '{index}' field 'embeddings' mapped as {emb.get('type')} not dense_vector"
        )
    return int(emb.get("dims", 0) or 0)


essential_fields = ("content", "summary", "keywords", "hypothetical_questions", "title")

def check_analyzer(es, index: str) -> bool:
    settings_resp = es.indices.get_settings(index=index)
    analysis = settings_resp.get(index, {}).get("settings", {}).get("index", {}).get("analysis", {})
    analyzers = analysis.get("analyzer", {})
    if "korean" not in analyzers:
        return False
    # Also ensure fields are wired to use it
    mapping = es.indices.get_mapping(index=index)
    props = mapping.get(index, {}).get("mappings", {}).get("properties", {})
    for f in essential_fields:
        field = props.get(f, {})
        if field.get("type") != "text" or field.get("analyzer") != "korean":
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pre-flight validator for embedding dims vs index mapping")
    parser.add_argument("--index", required=True, help="Elasticsearch index name")
    parser.add_argument("--provider", default="auto", help="Embedding provider (auto|polyglot|huggingface|solar|sentence_transformers)")
    parser.add_argument("--expect-dims", type=int, default=None, help="Expected dims (override) – optional")
    parser.add_argument("--check-analyzer", action="store_true", help="Also verify Nori analyzer wiring on text fields")
    args = parser.parse_args(argv)

    es = get_es()

    try:
        if not es.indices.exists(index=args.index):
            print(f"❌ Index does not exist: {args.index}")
            return 1

        index_dims = get_index_dims(es, args.index)
        if index_dims is None:
            print(f"❌ Index '{args.index}' has no 'embeddings' mapping. Expected dense_vector.")
            return 1

        provider_dim = args.expect_dims if args.expect_dims else get_provider_dim(args.provider)
        ok = True

        if int(provider_dim) != int(index_dims):
            print(f"❌ Dimension mismatch: provider={provider_dim} vs index={index_dims} @ {args.index}")
            ok = False
        else:
            print(f"✅ Dimensions match: {provider_dim} (provider) == {index_dims} (index)")

        if args.check_analyzer:
            if check_analyzer(es, args.index):
                print("✅ Nori analyzer configured for Korean fields (content/summary/keywords/hypothetical_questions/title)")
            else:
                print("❌ Nori analyzer missing or not applied to all required fields")
                ok = False

        return 0 if ok else 1
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
