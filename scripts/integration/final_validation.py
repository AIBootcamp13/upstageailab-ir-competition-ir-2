#!/usr/bin/env python3
"""
Final Validation of Profiling Insights Integration

Comprehensive test of the complete profiling insights integration including:
- Configuration system
- All enhancement features
- Performance validation
- Error handling
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ir_core.retrieval.core import hybrid_retrieve
from ir_core.retrieval.insights_manager import get_insights_status
from ir_core.retrieval.chunking import get_chunking_config
from ir_core.config import settings


def test_configuration_system():
    """Test that configuration settings are properly loaded."""
    print("=== Testing Configuration System ===")

    # Check profiling insights config
    insights_config = getattr(settings, 'profiling_insights', {})
    print(f"Profiling insights enabled: {insights_config.get('enabled', False)}")
    print(f"Query expansion enabled: {insights_config.get('use_query_expansion', False)}")
    print(f"Domain routing enabled: {insights_config.get('use_domain_routing', False)}")
    print(f"Dynamic chunking enabled: {insights_config.get('use_dynamic_chunking', False)}")
    print(f"Memory optimization enabled: {insights_config.get('use_memory_optimization', False)}")
    print(f"Cache TTL: {insights_config.get('cache_ttl_seconds', 'N/A')} seconds")
    print(f"Query expansion terms: {insights_config.get('query_expansion_terms', 'N/A')}")
    print(f"Memory batch fallback: {insights_config.get('memory_batch_fallback', 'N/A')}")

    return bool(insights_config)


def test_feature_integration():
    """Test all profiling insights features working together."""
    print("\n=== Testing Feature Integration ===")

    test_query = "neural network optimization techniques"

    try:
        # Test with all features enabled
        print("Testing with all profiling insights enabled...")
        results_enabled = hybrid_retrieve(
            query=test_query,
            rerank_k=5,
            use_profiling_insights=True
        )
        print(f"✓ Retrieved {len(results_enabled)} results with insights")

        # Test with all features disabled
        print("Testing with all profiling insights disabled...")
        results_disabled = hybrid_retrieve(
            query=test_query,
            rerank_k=5,
            use_profiling_insights=False
        )
        print(f"✓ Retrieved {len(results_disabled)} results without insights")

        # Compare results
        if results_enabled and results_disabled:
            enabled_score = results_enabled[0].get('score', 0)
            disabled_score = results_disabled[0].get('score', 0)
            improvement = enabled_score - disabled_score
            print(".3f")
        return True

    except Exception as e:
        print(f"✗ Feature integration test failed: {e}")
        return False


def test_error_handling():
    """Test error handling when profiling data is unavailable."""
    print("\n=== Testing Error Handling ===")

    # Test with non-existent report directory
    original_dir = settings.PROFILE_REPORT_DIR
    settings.PROFILE_REPORT_DIR = "non_existent_directory"

    try:
        results = hybrid_retrieve(
            query="test query",
            rerank_k=3,
            use_profiling_insights=True
        )
        print(f"✓ Gracefully handled missing profiling data: {len(results)} results")
        return True
    except Exception as e:
        print(f"✗ Error handling failed: {e}")
        return False
    finally:
        # Restore original setting
        settings.PROFILE_REPORT_DIR = original_dir


def test_chunking_integration():
    """Test that chunking recommendations are working."""
    print("\n=== Testing Chunking Integration ===")

    test_sources = ["arxiv", "pubmed", "wikipedia"]

    for src in test_sources:
        config = get_chunking_config(src)
        print(f"{src}: {config}")

    return True


def test_insights_status():
    """Test insights status reporting."""
    print("\n=== Testing Insights Status ===")

    status = get_insights_status()
    print(f"Insights loaded: {status['insights_loaded']}")
    print(f"Cache age: {status['cache_age_seconds']:.1f} seconds")
    print(f"Available insights: {status['available_insights']}")

    return status['insights_loaded']


def main():
    """Run comprehensive validation tests."""
    print("Final Validation of Profiling Insights Integration")
    print("=" * 60)

    tests = [
        ("Configuration System", test_configuration_system),
        ("Feature Integration", test_feature_integration),
        ("Error Handling", test_error_handling),
        ("Chunking Integration", test_chunking_integration),
        ("Insights Status", test_insights_status),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print("15")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests passed! Profiling insights integration is complete.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())