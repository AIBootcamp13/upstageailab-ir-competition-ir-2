#!/usr/bin/env python3
"""
Domain Classification Validation Script

This script demonstrates how to validate the domain classification accuracy
of the QueryAnalyzer using both predefined and LLM-generated validation sets.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ir_core.analysis.query_analyzer import QueryAnalyzer


def main():
    """Run domain classification validation."""
    print("🔬 Domain Classification Validation")
    print("=" * 50)

    # Initialize analyzer
    analyzer = QueryAnalyzer()

    # Test with predefined queries
    print("\n📋 Testing with predefined validation set...")
    validation_set = analyzer.create_domain_validation_set(num_queries_per_domain=5, use_llm=False)
    results = analyzer.evaluate_domain_classification(validation_set)

    print(f"✅ Validation completed for {results['total_queries']} queries")
    print(".2%")
    print(".3f")
    print(".3f")
    print(".3f")

    print("\n📊 Per-domain accuracy:")
    for domain, metrics in results['domain_metrics'].items():
        print(".1%")

    # Test with LLM-generated queries (if available)
    print("\n🤖 Testing with LLM-generated validation set...")
    try:
        llm_validation_set = analyzer.create_domain_validation_set(num_queries_per_domain=3, use_llm=True)
        llm_results = analyzer.evaluate_domain_classification(llm_validation_set)

        print(f"✅ LLM validation completed for {llm_results['total_queries']} queries")
        print(".2%")
        print(".3f")

    except Exception as e:
        print(f"⚠️  LLM validation failed: {e}")
        print("💡 Make sure OpenAI API key is configured")

    # Show some examples
    print("\n📝 Sample validation results:")
    for i, result in enumerate(results['detailed_results'][:3]):
        print(f"\n{i+1}. Query: {result['query']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Predicted: {result['predicted']}")
        print(f"   Exact match: {result['exact_match']}")
        print(".3f")

    print("\n🎯 Validation Summary:")
    print("- Domain classification supports multi-domain queries")
    print("- Current accuracy provides good baseline for scientific QA")
    print("- LLM-generated queries enable scalable validation")
    print("- Results can guide further improvements to keyword matching")


if __name__ == "__main__":
    main()
