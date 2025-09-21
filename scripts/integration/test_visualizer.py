#!/usr/bin/env python3
# scripts/test_visualizer.py

"""
Test script for the new AnalysisVisualizer component.
Tests all visualization types with sample data.
"""

import sys
import os
from pathlib import Path

# Add src to path
repo_dir = Path(__file__).parent.parent
src_dir = repo_dir / "src"
sys.path.insert(0, str(src_dir))

from ir_core.analysis.visualizer import AnalysisVisualizer
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
        query_analyses=query_analyses,
        retrieval_results=retrieval_results
    )

    return result


def test_visualizations():
    """Test all visualization types."""
    print("Creating sample analysis data...")
    sample_result = create_sample_analysis_result()
    results = [sample_result]  # List with one result for testing

    print("Testing matplotlib backend...")
    viz_matplotlib = AnalysisVisualizer(backend='matplotlib')

    # Test performance distribution
    print("  - Performance distribution histogram...")
    fig1 = viz_matplotlib.create_performance_distribution_histogram(results, "performance_dist")
    print("    ✓ Created performance distribution histogram")

    # Test query length vs performance
    print("  - Query length vs performance scatter...")
    fig2 = viz_matplotlib.create_query_length_performance_scatter(results, "query_length_perf")
    print("    ✓ Created query length vs performance scatter")

    # Test domain performance comparison
    print("  - Domain performance comparison...")
    fig3 = viz_matplotlib.create_domain_performance_comparison(results, "domain_comparison")
    print("    ✓ Created domain performance comparison")

    # Test error pattern heatmap
    print("  - Error pattern heatmap...")
    fig4 = viz_matplotlib.create_error_pattern_heatmap(results, "error_heatmap")
    print("    ✓ Created error pattern heatmap")

    # Test time series trends
    print("  - Time series performance trends...")
    timestamps = ["2025-01-01", "2025-01-02", "2025-01-03"]
    results_trend = [sample_result] * 3  # Duplicate for trend test
    fig5 = viz_matplotlib.create_time_series_performance_trends(results_trend, timestamps, "performance_trends")
    print("    ✓ Created time series performance trends")

    # Test comprehensive dashboard
    print("  - Comprehensive dashboard...")
    fig6 = viz_matplotlib.create_comprehensive_dashboard(results, "comprehensive_dashboard")
    print("    ✓ Created comprehensive dashboard")

    print("\nTesting plotly backend...")
    viz_plotly = AnalysisVisualizer(backend='plotly')

    # Test a few with plotly
    print("  - Performance distribution (plotly)...")
    fig7 = viz_plotly.create_performance_distribution_histogram(results, "performance_dist_plotly")
    print("    ✓ Created plotly performance distribution")

    print("  - Query length vs performance (plotly)...")
    fig8 = viz_plotly.create_query_length_performance_scatter(results, "query_length_perf_plotly")
    print("    ✓ Created plotly query length vs performance")

    print("\nTesting convenience functions...")
    from ir_core.analysis import plot_performance_distribution, plot_query_performance_correlation, plot_domain_comparison

    print("  - Convenience function: plot_performance_distribution...")
    fig9 = plot_performance_distribution(results, 'matplotlib')
    print("    ✓ Convenience function works")

    print("  - Convenience function: plot_query_performance_correlation...")
    fig10 = plot_query_performance_correlation(results, 'matplotlib')
    print("    ✓ Convenience function works")

    print("  - Convenience function: plot_domain_comparison...")
    fig11 = plot_domain_comparison(results, 'matplotlib')
    print("    ✓ Convenience function works")

    print("\n✅ All visualization tests completed successfully!")
    print(f"📊 Visualizations saved to: {viz_matplotlib.output_dir}")


if __name__ == "__main__":
    test_visualizations()
