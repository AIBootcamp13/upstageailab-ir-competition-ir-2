#!/usr/bin/env python3
# scripts/test_report_generator.py

"""
Test script for the AnalysisReportGenerator component.
Tests all report generation types with sample data.
"""

import sys
import os
from pathlib import Path

# Add src to path
repo_dir = Path(__file__).parent.parent
src_dir = repo_dir / "src"
sys.path.insert(0, str(src_dir))

from ir_core.analysis.report_generator import AnalysisReportGenerator
from ir_core.analysis.core import AnalysisResult, QueryAnalysis, RetrievalResult
import numpy as np


def create_sample_analysis_result() -> AnalysisResult:
    """Create a sample AnalysisResult for testing."""
    # Create sample query analyses
    query_analyses = []
    retrieval_results = []

    domains = ['biology', 'chemistry', 'physics', 'mathematics']
    np.random.seed(42)  # For reproducible results

    for i in range(50):
        # Sample query
        query = f"Sample scientific question {i+1}"
        query_length = len(query) + np.random.randint(20, 100)

        # Random domain assignment (can have multiple)
        num_domains = np.random.randint(1, 3)
        query_domains = np.random.choice(domains, num_domains, replace=False).tolist()

        # Query analysis
        qa = QueryAnalysis(
            original_query=query,
            rewritten_query=query + " (rewritten)",
            query_length=query_length,
            domain=query_domains,
            complexity_score=np.random.uniform(0.1, 1.0),
            processing_time=np.random.uniform(0.01, 0.1),
            rewrite_effective=np.random.choice([True, False])
        )
        query_analyses.append(qa)

        # Retrieval result
        ground_truth = f"doc_{i}"
        predicted_ids = [f"doc_{i}"] + [f"doc_{np.random.randint(0, 100)}" for _ in range(9)]
        predicted_scores = np.random.uniform(0, 1, 10)
        predicted_scores[0] = np.random.uniform(0.8, 1.0)  # Make first one more likely to be correct

        # Calculate AP score (simplified)
        ap_score = np.random.uniform(0, 1)

        rr = RetrievalResult(
            query=query,
            ground_truth_id=ground_truth,
            predicted_ids=predicted_ids,
            predicted_scores=predicted_scores.tolist(),
            ap_score=ap_score,
            rank_of_ground_truth=1 if np.random.random() > 0.2 else None,
            top_k_precision={1: 1.0 if np.random.random() > 0.3 else 0.0,
                           3: np.random.uniform(0.3, 1.0),
                           5: np.random.uniform(0.2, 1.0),
                           10: np.random.uniform(0.1, 1.0)},
            retrieval_time=np.random.uniform(0.01, 0.5)
        )
        retrieval_results.append(rr)

    # Create analysis result
    result = AnalysisResult(
        map_score=float(np.mean([rr.ap_score for rr in retrieval_results])),
        mean_ap=float(np.mean([rr.ap_score for rr in retrieval_results])),
        precision_at_k={k: float(np.mean([rr.top_k_precision.get(k, 0) for rr in retrieval_results]))
                       for k in [1, 3, 5, 10]},
        recall_at_k={k: float(np.random.uniform(0.1, 0.8)) for k in [1, 3, 5, 10]},
        total_queries=len(query_analyses),
        avg_query_length=float(np.mean([qa.query_length for qa in query_analyses])),
        rewrite_rate=float(np.mean([qa.rewrite_effective for qa in query_analyses])),
        domain_distribution={domain: np.random.randint(5, 20) for domain in domains},
        retrieval_success_rate=float(np.mean([1 if rr.rank_of_ground_truth else 0 for rr in retrieval_results])),
        avg_retrieval_time=float(np.mean([rr.retrieval_time for rr in retrieval_results])),
        error_categories={'timeout': 2, 'parsing_error': 1, 'other': 3},
        recommendations=[
            "Consider increasing the retrieval timeout for complex queries",
            "Improve query preprocessing to handle scientific terminology better",
            "Optimize the ranking algorithm for better precision at higher K values"
        ],
        query_analyses=query_analyses,
        retrieval_results=retrieval_results,
        domain_error_rates={domain: np.random.uniform(0.1, 0.8) for domain in domains}
    )

    return result


def test_report_generation():
    """Test all report generation types."""
    print("Creating sample analysis data...")
    sample_result = create_sample_analysis_result()
    results = [sample_result]  # List with one result for testing

    generator = AnalysisReportGenerator()

    print("Testing performance report generation...")

    # Test HTML performance report
    print("  - Generating HTML performance report...")
    html_path = generator.generate_performance_report(sample_result, format='html')
    print(f"    ✓ HTML report saved to: {html_path}")

    # Test Markdown performance report
    print("  - Generating Markdown performance report...")
    md_path = generator.generate_performance_report(sample_result, format='markdown')
    print(f"    ✓ Markdown report saved to: {md_path}")

    # Test JSON performance report
    print("  - Generating JSON performance report...")
    json_path = generator.generate_performance_report(sample_result, format='json')
    print(f"    ✓ JSON report saved to: {json_path}")

    print("\nTesting error analysis report generation...")

    # Test HTML error report
    print("  - Generating HTML error analysis report...")
    error_html_path = generator.generate_error_analysis_report(sample_result, format='html')
    print(f"    ✓ HTML error report saved to: {error_html_path}")

    print("\nTesting trend analysis report generation...")

    # Test HTML trend report
    print("  - Generating HTML trend analysis report...")
    trend_html_path = generator.generate_trend_analysis_report(results, format='html')
    print(f"    ✓ HTML trend report saved to: {trend_html_path}")

    print("\nTesting convenience functions...")

    # Test convenience functions
    from ir_core.analysis import generate_performance_report, generate_error_report, generate_trend_report

    print("  - Testing generate_performance_report convenience function...")
    conv_perf_path = generate_performance_report(sample_result, 'html')
    print(f"    ✓ Convenience function works: {conv_perf_path}")

    print("  - Testing generate_error_report convenience function...")
    conv_error_path = generate_error_report(sample_result, 'html')
    print(f"    ✓ Convenience function works: {conv_error_path}")

    print("  - Testing generate_trend_report convenience function...")
    conv_trend_path = generate_trend_report(results, 'html')
    print(f"    ✓ Convenience function works: {conv_trend_path}")

    print("\n✅ All report generation tests completed successfully!")
    print(f"📊 Reports saved to: {generator.output_dir}")

    # Print summary of generated files
    print("\n📋 Generated Files Summary:")
    for file_path in generator.output_dir.glob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            print(f"  - {file_path.name} ({size} bytes)")


if __name__ == "__main__":
    test_report_generation()
