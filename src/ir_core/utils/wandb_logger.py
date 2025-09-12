# src/ir_core/utils/wandb_logger.py

"""
Wandb-specific logging utilities for the Scientific QA analysis framework.

This module provides specialized logging functions for Wandb integration,
keeping the core analysis logic framework-agnostic.
"""

import wandb
from typing import Dict, List, Any, Optional
from ..analysis.core import AnalysisResult, QueryAnalysis, RetrievalResult


class WandbAnalysisLogger:
    """
    Specialized logger for Wandb integration with analysis results.

    This class handles the conversion of analysis results to Wandb-compatible
    formats and provides enhanced visualization capabilities.
    """

    def __init__(self, project_name: str = "scientific-qa-rag", entity: Optional[str] = None):
        """
        Initialize the Wandb logger.

        Args:
            project_name: Wandb project name
            entity: Wandb entity (user or team)
        """
        self.project_name = project_name
        self.entity = entity

    def log_analysis_result(
        self,
        result: AnalysisResult,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log comprehensive analysis results to Wandb.

        Args:
            result: AnalysisResult object containing all metrics
            run_name: Optional custom run name
            config: Optional configuration to log
        """
        # Initialize Wandb run if not already initialized
        if wandb.run is None:
            wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=config
            )

        # Log core metrics
        self._log_core_metrics(result)

        # Log query analysis
        self._log_query_analysis(result)

        # Log retrieval analysis
        self._log_retrieval_analysis(result)

        # Log detailed tables
        self._log_detailed_tables(result)

        # Log recommendations
        self._log_recommendations(result)

    def _log_core_metrics(self, result: AnalysisResult) -> None:
        """Log the core performance metrics."""
        wandb.log({
            "map_score": result.map_score,
            "mean_ap": result.mean_ap,
            "retrieval_success_rate": result.retrieval_success_rate,
            "rewrite_rate": result.rewrite_rate,
            "avg_query_length": result.avg_query_length,
            "total_queries": result.total_queries
        })

        # Log precision@K metrics
        for k, precision in result.precision_at_k.items():
            wandb.log({f"precision_at_{k}": precision})

        # Log recall@K metrics
        for k, recall in result.recall_at_k.items():
            wandb.log({f"recall_at_{k}": recall})

    def _log_query_analysis(self, result: AnalysisResult) -> None:
        """Log query-level analysis metrics."""
        # Query statistics
        wandb.log({
            "query_stats": {
                "avg_query_length": result.avg_query_length,
                "rewrite_rate": result.rewrite_rate,
                "total_queries": result.total_queries
            }
        })

        # Domain distribution
        if result.domain_distribution:
            domain_table = wandb.Table(
                columns=["Domain", "Count"],
                data=[[domain, count] for domain, count in result.domain_distribution.items()]
            )
            wandb.log({"domain_distribution": domain_table})

    def _log_retrieval_analysis(self, result: AnalysisResult) -> None:
        """Log retrieval-specific analysis metrics."""
        # Error categories
        if result.error_categories:
            error_table = wandb.Table(
                columns=["Error Type", "Count"],
                data=[[error_type, count] for error_type, count in result.error_categories.items()]
            )
            wandb.log({"error_categories": error_table})

        # Retrieval timing (if available)
        if hasattr(result, 'avg_retrieval_time') and result.avg_retrieval_time > 0:
            wandb.log({"avg_retrieval_time": result.avg_retrieval_time})

    def _log_detailed_tables(self, result: AnalysisResult) -> None:
        """Log detailed analysis tables."""
        # Main results table
        table_data = []
        for i, (query_analysis, retrieval_result) in enumerate(
            zip(result.query_analyses, result.retrieval_results)
        ):
            # Shorten IDs for display
            gt_short = retrieval_result.ground_truth_id[:8] if retrieval_result.ground_truth_id else ""
            pred_short = [pid[:8] for pid in retrieval_result.predicted_ids[:10]]

            table_data.append([
                query_analysis.original_query,
                query_analysis.rewritten_query,
                gt_short,
                pred_short,
                f"{retrieval_result.ap_score:.4f}",
                retrieval_result.rank_of_ground_truth or 0,
                query_analysis.query_length,
                ", ".join(query_analysis.domain) if isinstance(query_analysis.domain, list) else query_analysis.domain
            ])

        results_table = wandb.Table(
            columns=[
                "Original Query", "Rewritten Query", "GT ID (8char)",
                "Pred IDs (8char)", "AP Score", "GT Rank", "Query Length", "Domain"
            ],
            data=table_data
        )
        wandb.log({"detailed_results": results_table})

    def _log_recommendations(self, result: AnalysisResult) -> None:
        """Log analysis recommendations."""
        if result.recommendations:
            # Use HTML entities for proper bullet point rendering
            recommendations_html = "<ul style='margin: 0; padding-left: 20px;'>"
            for rec in result.recommendations:
                recommendations_html += f"<li>{rec}</li>"
            recommendations_html += "</ul>"
            wandb.log({"recommendations": wandb.Html(recommendations_html)})

    def create_performance_dashboard(self, results: List[AnalysisResult]) -> None:
        """
        Create a comprehensive performance dashboard from multiple analysis results.

        Args:
            results: List of AnalysisResult objects from different runs
        """
        if not results:
            return

        # Aggregate metrics across runs
        map_scores = [r.map_score for r in results]
        precision_trends = {}

        for k in [1, 3, 5, 10]:
            precision_trends[k] = [r.precision_at_k.get(k, 0) for r in results]

        # Log aggregated metrics
        wandb.log({
            "performance_summary": {
                "avg_map": sum(map_scores) / len(map_scores),
                "best_map": max(map_scores),
                "worst_map": min(map_scores),
                "map_std": (sum((x - sum(map_scores)/len(map_scores))**2 for x in map_scores) / len(map_scores))**0.5
            }
        })

        # Create trend charts
        for k, precisions in precision_trends.items():
            trend_table = wandb.Table(
                columns=["Run", f"P@{k}"],
                data=[[f"Run_{i+1}", p] for i, p in enumerate(precisions)]
            )
            wandb.log({f"precision_at_{k}_trend": trend_table})

    def log_query_analysis_table(self, queries: List[Dict[str, Any]]) -> None:
        """
        Log a detailed table of query analysis for manual inspection.

        Args:
            queries: List of query dictionaries with analysis
        """
        if not queries:
            return

        table_data = []
        for query in queries:
            table_data.append([
                query.get("original_query", ""),
                query.get("rewritten_query", ""),
                query.get("query_length", 0),
                query.get("domain", "unknown"),
                query.get("complexity_score", 0.0),
                query.get("rewrite_effective", False)
            ])

        query_table = wandb.Table(
            columns=[
                "Original Query", "Rewritten Query", "Length",
                "Domain", "Complexity", "Rewrite Effective"
            ],
            data=table_data
        )
        wandb.log({"query_analysis_table": query_table})


def create_enhanced_run_name(result: AnalysisResult, prefix: str = "analysis") -> str:
    """
    Create an enhanced run name based on analysis results.

    Args:
        result: AnalysisResult object
        prefix: Prefix for the run name

    Returns:
        str: Enhanced run name
    """
    map_str = f"MAP_{result.map_score:.3f}"
    success_str = f"Success_{result.retrieval_success_rate:.1%}"
    queries_str = f"Queries_{result.total_queries}"

    return f"{prefix}-{map_str}-{success_str}-{queries_str}"


def log_model_comparison(
    model_results: Dict[str, AnalysisResult],
    metric: str = "map_score"
) -> None:
    """
    Log a comparison of different models or configurations.

    Args:
        model_results: Dictionary mapping model names to AnalysisResult objects
        metric: Metric to compare (default: map_score)
    """
    comparison_data = []
    for model_name, result in model_results.items():
        if hasattr(result, metric):
            value = getattr(result, metric)
            comparison_data.append([model_name, value])

    if comparison_data:
        comparison_table = wandb.Table(
            columns=["Model", metric.replace("_", " ").title()],
            data=comparison_data
        )
        wandb.log({f"model_comparison_{metric}": comparison_table})
