#!/usr/bin/env python3
"""
Test script to demonstrate confidence score logging functionality.

This script creates a QueryEnhancementManager and applies different techniques
to showcase the rich confidence score logging with color-coded output.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ir_core.query_enhancement.manager import QueryEnhancementManager


def test_confidence_logging():
    """Test confidence score logging with various query types."""

    print("🎯 Testing Confidence Score Logging")
    print("=" * 50)

    # Initialize the manager (this will use default settings)
    try:
        manager = QueryEnhancementManager()
        print("✅ QueryEnhancementManager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize manager: {e}")
        return

    # Test queries with different characteristics
    test_queries = [
        "What is machine learning?",  # Simple question - should use rewriting
        "Explain the process of photosynthesis in detail",  # Complex - might use decomposition
        "How does quantum computing work?",  # Technical - might use step-back
        "Hello, how are you today?",  # Conversational - should bypass
        "Translate 'Hello world' to French",  # Translation needed
    ]

    print("\n🧪 Running test queries...")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: {query}")
        print("-" * 40)

        try:
            # Test with specific techniques to force application
            techniques_to_test = ['rewriting', 'step_back', 'decomposition']

            for technique in techniques_to_test:
                print(f"\n  🔧 Testing technique: {technique}")
                result = manager.enhance_query(query, technique=technique)
                print(f"     ✅ Technique: {result.get('technique_used', 'N/A')}")
                print(f"     📊 Confidence: {result.get('confidence', 'N/A')}")
                print(f"     ✨ Enhanced: {result.get('enhanced', False)}")
                if result.get('enhanced_query') != query:
                    print(f"     📝 Original: {query}")
                    print(f"     🔄 Enhanced: {result.get('enhanced_query', 'N/A')}")

        except Exception as e:
            print(f"❌ Enhancement failed: {e}")

    print("\n🎉 Confidence logging test completed!")
    print("Check the console output above for rich confidence score displays.")


if __name__ == "__main__":
    test_confidence_logging()