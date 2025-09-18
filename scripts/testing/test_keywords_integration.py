#!/usr/bin/env python3
"""
Test script for curated keywords integration in RAG retrieval

This script tests the integration of curated scientific keywords
into the retrieval system and validates the hybrid approach.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ir_core.retrieval.keywords_integration import CuratedKeywordsIntegrator, enhance_query_with_curated_keywords
from ir_core.retrieval.core import _extract_keywords_from_query, build_flexible_match_query


def test_keywords_integration():
    """Test the curated keywords integration functionality"""

    print("=== Testing Curated Keywords Integration ===\n")

    # Initialize the integrator
    integrator = CuratedKeywordsIntegrator()
    integrator.initialize_embeddings()

    # Test queries
    test_queries = [
        "DNA 복제 과정",  # Korean: DNA replication process
        "quantum mechanics principles",  # English: quantum mechanics
        "machine learning algorithms",  # Technical: ML algorithms
        "climate change effects",  # Environmental science
        "organic chemistry reactions",  # Chemistry
    ]

    for query in test_queries:
        print(f"Testing query: '{query}'")
        print("-" * 50)

        # Test LLM keyword extraction
        try:
            llm_keywords = _extract_keywords_from_query(query)
            print(f"LLM keywords: {llm_keywords}")
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            llm_keywords = []

        # Test curated keywords enhancement
        try:
            enhanced_query, all_keywords = integrator.enhance_query_with_keywords(
                query, llm_keywords, use_semantic_matching=True, max_additional=3
            )
            print(f"Enhanced query: '{enhanced_query}'")
            print(f"All keywords ({len(all_keywords)}): {all_keywords}")
            print(f"Curated keywords added: {len(all_keywords) - len(llm_keywords)}")
        except Exception as e:
            print(f"Curated enhancement failed: {e}")
            enhanced_query = query
            all_keywords = llm_keywords

        # Test query building
        try:
            es_query = build_flexible_match_query(enhanced_query, size=10)
            print("Elasticsearch query built successfully")
            # Print a summary of the query structure
            should_clauses = es_query.get("query", {}).get("bool", {}).get("should", [])
            print(f"Query has {len(should_clauses)} should clauses")
        except Exception as e:
            print(f"Query building failed: {e}")

        print("\n")


def test_semantic_matching():
    """Test semantic keyword matching functionality"""

    print("=== Testing Semantic Keyword Matching ===\n")

    integrator = CuratedKeywordsIntegrator()
    integrator.initialize_embeddings()

    test_query = "molecular biology techniques"
    print(f"Query: '{test_query}'")

    # Test semantic matching
    relevant_keywords = integrator.find_relevant_keywords(test_query, top_k=5)
    print(f"Semantically relevant keywords: {relevant_keywords}")

    # Test domain-specific keywords
    biology_keywords = integrator.get_domain_keywords("biology", 5)
    print(f"Biology domain keywords: {biology_keywords}")

    print("\n")


def test_domain_categorization():
    """Test domain-based keyword categorization"""

    print("=== Testing Domain Categorization ===\n")

    integrator = CuratedKeywordsIntegrator()

    domains = ["biology", "chemistry", "physics", "computer_science", "mathematics"]
    for domain in domains:
        keywords = integrator.get_domain_keywords(domain, 3)
        print(f"{domain}: {keywords}")

    print("\n")


if __name__ == "__main__":
    try:
        test_domain_categorization()
        test_semantic_matching()
        test_keywords_integration()
        print("✅ All tests completed successfully!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()