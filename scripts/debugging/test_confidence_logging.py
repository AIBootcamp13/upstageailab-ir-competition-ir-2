#!/usr/bin/env python3
"""
Test script to demonstrate confidence score logging functionality.

This script creates a QueryEnhancementManager and applies different techniques
to showcase the rich confidence score logging with color-coded output.
"""

import sys
import os
from typing import List, Dict, Any

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ir_core.query_enhancement.manager import QueryEnhancementManager


def test_confidence_logging(return_results: bool = False) -> List[Dict[str, Any]]:
    """Test confidence score logging with various query types."""

    results = []

    # Initialize the manager (this will use default settings)
    try:
        manager = QueryEnhancementManager()
        if not return_results:
            print("✅ QueryEnhancementManager initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize manager: {e}"
        if return_results:
            results.append({"error": error_msg, "initialization_failed": True})
            return results
        else:
            print(f"❌ {error_msg}")
            print("💡 Make sure OPENAI_API_KEY environment variable is set")
            return []

    # Test queries with different characteristics
    test_queries = [
        "What is machine learning?",  # Simple question - should use rewriting
        "Explain the process of photosynthesis in detail",  # Complex - might use decomposition
        "How does quantum computing work?",  # Technical - might use step-back
        "Hello, how are you today?",  # Conversational - should bypass
        "Translate 'Hello world' to French",  # Translation needed
    ]

    if not return_results:
        print("\n🧪 Running test queries...")
        print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        if not return_results:
            print(f"\n📝 Test Query {i}: {query}")
            print("-" * 40)

        try:
            # Test with specific techniques to force application
            techniques_to_test = ['rewriting', 'step_back', 'decomposition']

            for technique in techniques_to_test:
                if not return_results:
                    print(f"\n  🔧 Testing technique: {technique}")
                result = manager.enhance_query(query, technique=technique)

                test_result = {
                    "query": query,
                    "technique": technique,
                    "technique_used": result.get('technique_used', 'N/A'),
                    "confidence_score": result.get('confidence', 0.0),
                    "enhanced": result.get('enhanced', False),
                    "original_query": query,
                    "enhanced_query": result.get('enhanced_query', query),
                    "error": None
                }
                results.append(test_result)

                if not return_results:
                    print(f"     ✅ Technique: {result.get('technique_used', 'N/A')}")
                    print(f"     📊 Confidence: {result.get('confidence', 'N/A')}")
                    print(f"     ✨ Enhanced: {result.get('enhanced', False)}")
                    if result.get('enhanced_query') != query:
                        print(f"     📝 Original: {query}")
                        print(f"     🔄 Enhanced: {result.get('enhanced_query', 'N/A')}")

        except Exception as e:
            error_result = {
                "query": query,
                "technique": technique,
                "error": str(e)
            }
            results.append(error_result)
            if not return_results:
                print(f"❌ Enhancement failed: {e}")

    # Test invalid technique
    if not return_results:
        print(f"\n📝 Test Query {len(test_queries) + 1}: Testing invalid technique")
        print("-" * 40)

    try:
        invalid_result = manager.enhance_query("Test query", technique="invalid_technique")
        invalid_test_result = {
            "query": "Test query",
            "technique": "invalid_technique",
            "technique_used": invalid_result.get('technique_used', 'N/A'),
            "confidence_score": invalid_result.get('confidence', 0.0),
            "enhanced": invalid_result.get('enhanced', False),
            "original_query": "Test query",
            "enhanced_query": invalid_result.get('enhanced_query', "Test query"),
            "error": None
        }
        results.append(invalid_test_result)

        if not return_results:
            print("  🔧 Testing technique: invalid_technique")
            print(f"     ✅ Technique: {invalid_result.get('technique_used', 'N/A')}")
            print(f"     📊 Confidence: {invalid_result.get('confidence', 'N/A')}")
            print(f"     ✨ Enhanced: {invalid_result.get('enhanced', False)}")

    except Exception as e:
        invalid_error_result = {
            "query": "Test query",
            "technique": "invalid_technique",
            "error": str(e)
        }
        results.append(invalid_error_result)
        if not return_results:
            print(f"❌ Invalid technique test failed as expected: {e}")

    if not return_results:
        print("\n🎉 Confidence logging test completed!")
        print("Check the console output above for rich confidence score displays.")

    return results


def test_confidence_logging_runner():
    """
    Test function for confidence logging, refactored for automated test runners.
    Replaces print statements with assertions for automated verification.
    """
    # Capture results for assertions
    results = []
    try:
        results = test_confidence_logging(return_results=True)
    except Exception as e:
        assert False, f"Enhancement failed with exception: {e}"

    # Check if initialization failed
    if results and results[0].get("initialization_failed"):
        # Skip the test if we can't initialize (e.g., no API key)
        import pytest
        pytest.skip(f"Skipping test due to initialization failure: {results[0]['error']}")

    assert results, "No results returned from confidence logging test."

    # Check that we have results for all test queries and techniques
    expected_results_count = 5 * 3 + 1  # 5 queries * 3 techniques + 1 invalid technique test
    assert len(results) == expected_results_count, f"Expected {expected_results_count} results, got {len(results)}"

    for result in results:
        if "error" in result and result["error"]:
            # For error cases (like invalid technique), we expect an error
            if result["technique"] == "invalid_technique":
                assert result["error"] is not None, "Invalid technique should produce an error"
            else:
                assert False, f"Unexpected error in result: {result['error']}"
        else:
            # For successful cases
            assert "confidence_score" in result, "Result missing confidence_score."
            assert isinstance(result["confidence_score"], (float, int)), "Confidence score is not a number."
            assert 0.0 <= result["confidence_score"] <= 1.0, "Confidence score out of expected range."
            assert "enhanced_query" in result, "Result missing enhanced_query."
            assert isinstance(result["enhanced_query"], str), "Enhanced query is not a string."
            assert "technique_used" in result, "Result missing technique_used."
            assert "query" in result, "Result missing query."


if __name__ == "__main__":
    test_confidence_logging()