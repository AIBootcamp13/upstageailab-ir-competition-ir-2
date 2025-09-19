#!/usr/bin/env python3
"""
Parallel Processing Example Script

This script demonstrates how to use the parallel processing capabilities
of the analysis framework for high-performance query analysis.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ir_core.analysis.query_analyzer import QueryAnalyzer
from omegaconf import DictConfig
import time


def main():
    """Demonstrate parallel processing capabilities."""
    print("🚀 Parallel Processing Example")
    print("=" * 50)

    # Example queries for different scientific domains
    sample_queries = [
        "물체의 질량과 무게의 차이점은 무엇인가요?",
        "원자의 구조는 어떻게 되어 있나요?",
        "DNA 복제 과정은 어떻게 이루어지나요?",
        "화학 반응에서 촉매의 역할은 무엇인가요?",
        "태양계에서 가장 큰 행성은 무엇인가요?",
        "빅뱅 이론은 무엇인가요?",
        "피타고라스 정리는 무엇인가요?",
        "세포막의 기능은 무엇인가요?",
        "산과 염기의 차이점은 무엇인가요?",
        "지진의 원인은 무엇인가요?",
        "광합성 과정은 어떻게 이루어지나요?",
        "반응 속도에 영향을 미치는 요인은 무엇인가요?",
        "블랙홀의 특성은 무엇인가요?",
        "미적분의 기본 개념은 무엇인가요?",
        "지구의 대기층 구성은 어떻게 되어 있나요?",
    ] * 3  # Create larger batch for demonstration

    print(f"📊 Processing {len(sample_queries)} queries...")

    # Test 1: Default configuration (automatic parallel)
    print("\n🔄 Test 1: Default configuration (automatic parallel)")
    analyzer = QueryAnalyzer()

    start_time = time.time()
    results = analyzer.analyze_batch(sample_queries)
    elapsed = time.time() - start_time

    print(".2f")
    print(f"✅ Processed {len(results)} queries")
    print(f"📈 Throughput: {len(results)/elapsed:.1f} queries/second")

    # Test 2: Custom configuration
    print("\n🔧 Test 2: Custom configuration (4 workers)")
    config = DictConfig({"analysis": {"max_workers": 4, "enable_parallel": True}})
    analyzer_custom = QueryAnalyzer(config)

    start_time = time.time()
    results_custom = analyzer_custom.analyze_batch(sample_queries, max_workers=4)
    elapsed_custom = time.time() - start_time

    print(".2f")
    print(f"✅ Processed {len(results_custom)} queries")
    print(f"📈 Throughput: {len(results_custom)/elapsed_custom:.1f} queries/second")

    # Test 3: Sequential processing (for comparison)
    print("\n🐌 Test 3: Sequential processing (for comparison)")
    config_sequential = DictConfig(
        {"analysis": {"max_workers": 1, "enable_parallel": False}}
    )
    analyzer_seq = QueryAnalyzer(config_sequential)

    start_time = time.time()
    results_seq = analyzer_seq.analyze_batch(sample_queries, max_workers=0)
    elapsed_seq = time.time() - start_time

    print(".2f")
    print(f"✅ Processed {len(results_seq)} queries")
    print(f"📈 Throughput: {len(results_seq)/elapsed_seq:.1f} queries/second")

    # Performance comparison
    print("\n📊 Performance Comparison")
    print("=" * 30)
    speedup_parallel = elapsed_seq / elapsed if elapsed > 0 else 1
    speedup_custom = elapsed_seq / elapsed_custom if elapsed_custom > 0 else 1

    print(".1f")
    print(".1f")
    print(".1f")

    # Show sample results
    print("\n🎯 Sample Analysis Results")
    print("=" * 25)
    for i, result in enumerate(results[:3]):
        print(f"Query {i+1}: {sample_queries[i][:50]}...")
        print(f"  Domain: {result.domain}")
        print(f"  Complexity: {result.complexity_score:.2f}")
        print(f"  Word count: {result.word_count}")
        print()

    print("✅ Parallel processing demonstration completed!")
    print("\n💡 Tips:")
    print("  - Use parallel processing for batches > 10 queries")
    print("  - Adjust max_workers based on your CPU cores")
    print("  - Monitor memory usage for very large batches")
    print("  - Use sequential processing for debugging")


if __name__ == "__main__":
    main()
