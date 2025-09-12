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

        # Log Phase 3: Retrieval Quality Assessment
        # self._log_retrieval_quality_metrics(result)  # Temporarily commented out due to duplicate method

        # Log detailed tables
        self._log_detailed_tables(result)

        # Log interactive charts
        self._log_interactive_charts(result)

        # Log custom visualizations
        self._log_custom_visualizations(result)

    def _log_core_metrics(self, result: AnalysisResult) -> None:
        """Log the core performance metrics."""
        core_metrics = {
            "map_score": result.map_score,
            "mean_ap": result.mean_ap,
            "retrieval_success_rate": result.retrieval_success_rate,
            "rewrite_rate": result.rewrite_rate,
            "avg_query_length": result.avg_query_length,
            "total_queries": result.total_queries
        }

        # Add precision@K and recall@K to the same dictionary
        for k, precision in result.precision_at_k.items():
            core_metrics[f"precision_at_{k}"] = precision
        for k, recall in result.recall_at_k.items():
            core_metrics[f"recall_at_{k}"] = recall

        # Log everything in a single, atomic call
        wandb.log(core_metrics)

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


    def _log_retrieval_quality_metrics(self, result: AnalysisResult) -> None:
        """Log Phase 3: Retrieval Quality Assessment metrics."""
        # Performance segmentation
        if result.performance_segmentation:
            segmentation_table = wandb.Table(
                columns=["Performance Level", "Count"],
                data=[[level, count] for level, count in result.performance_segmentation.items()]
            )
            wandb.log({"performance_segmentation": segmentation_table})

        # Ranking quality metrics
        if result.ranking_quality_metrics:
            ranking_metrics = result.ranking_quality_metrics

            # Log NDCG@K
            if "ndcg_at_k" in ranking_metrics:
                for k, ndcg in ranking_metrics["ndcg_at_k"].items():
                    wandb.log({f"ndcg_at_{k}": ndcg})

            # Log other ranking metrics
            wandb.log({
                "reciprocal_rank": ranking_metrics.get("reciprocal_rank", 0.0),
                "score_variance": ranking_metrics.get("score_variance", 0.0)
            })

            # Rank distribution table
            if "rank_distribution" in ranking_metrics and ranking_metrics["rank_distribution"]:
                rank_table = wandb.Table(
                    columns=["Rank", "Count"],
                    data=[[rank, count] for rank, count in ranking_metrics["rank_distribution"].items()]
                )
                wandb.log({"rank_distribution": rank_table})

        # Score distribution metrics
        if result.score_distribution_metrics:
            score_metrics = result.score_distribution_metrics
            wandb.log({
                "mean_retrieval_score": score_metrics.get("mean_score", 0.0),
                "median_retrieval_score": score_metrics.get("median_score", 0.0),
                "score_std": score_metrics.get("score_std", 0.0),
                "score_skewness": score_metrics.get("score_skewness", 0.0)
            })

            # Score percentiles
            if "score_percentiles" in score_metrics:
                for percentile, value in score_metrics["score_percentiles"].items():
                    wandb.log({f"score_p{percentile}": value})

        # Consistency metrics
        if result.consistency_metrics:
            consistency = result.consistency_metrics
            wandb.log({
                "jaccard_similarity": consistency.get("jaccard_similarity", 0.0),
                "rank_correlation": consistency.get("rank_correlation", 0.0),
                "score_stability": consistency.get("score_stability", 0.0)
            })

            # Top-K overlap
            if "top_k_overlap" in consistency:
                for k, overlap in consistency["top_k_overlap"].items():
                    wandb.log({f"top_{k}_overlap": overlap})

        # False positive/negative analysis
        if result.false_positive_analysis:
            fp_analysis = result.false_positive_analysis
            wandb.log({
                "false_positive_rate": fp_analysis.get("false_positive_rate", 0.0),
                "false_negative_rate": fp_analysis.get("false_negative_rate", 0.0)
            })

            # Precision by rank
            if "precision_by_rank" in fp_analysis:
                for rank, precision in fp_analysis["precision_by_rank"].items():
                    wandb.log({f"precision_at_rank_{rank}": precision})

        # Confidence analysis
        if result.confidence_analysis:
            confidence = result.confidence_analysis
            wandb.log({
                "score_calibration_error": confidence.get("score_calibration_error", 0.0)
            })

            # Confidence by rank
            if "confidence_by_rank" in confidence:
                for rank, conf in confidence["confidence_by_rank"].items():
                    wandb.log({f"confidence_at_rank_{rank}": conf})

            # Uncertainty scores summary
            if "uncertainty_scores" in confidence and confidence["uncertainty_scores"]:
                uncertainty_scores = confidence["uncertainty_scores"]
                wandb.log({
                    "mean_uncertainty": sum(uncertainty_scores) / len(uncertainty_scores),
                    "max_uncertainty": max(uncertainty_scores),
                    "min_uncertainty": min(uncertainty_scores)
                })

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

    def _log_enhanced_error_analysis(self, result: AnalysisResult) -> None:
        """Log Phase 4 enhanced error analysis results."""
        # Log error category breakdowns
        if result.query_understanding_failures:
            wandb.log({"query_understanding_failures": result.query_understanding_failures})

        if result.retrieval_failures:
            wandb.log({"retrieval_failures": result.retrieval_failures})

        if result.system_failures:
            wandb.log({"system_failures": result.system_failures})

        # Log domain error rates
        if result.domain_error_rates:
            domain_error_table = wandb.Table(
                columns=["Domain", "Error Rate"],
                data=[[domain, rate] for domain, rate in result.domain_error_rates.items()]
            )
            wandb.log({"domain_error_rates": domain_error_table})

        # Log error patterns
        if result.error_patterns:
            # Log query length correlation
            if "query_length_correlation" in result.error_patterns:
                wandb.log({"query_length_success_correlation": result.error_patterns["query_length_correlation"]})

            # Log domain performance patterns
            if result.error_patterns.get("domain_performance_patterns"):
                domain_patterns = result.error_patterns["domain_performance_patterns"]
                patterns_data = []
                for domain, pattern in domain_patterns.items():
                    patterns_data.append([
                        domain,
                        pattern.get("success_rate", 0),
                        pattern.get("sample_size", 0),
                        pattern.get("needs_improvement", False)
                    ])

                if patterns_data:
                    patterns_table = wandb.Table(
                        columns=["Domain", "Success Rate", "Sample Size", "Needs Improvement"],
                        data=patterns_data
                    )
                    wandb.log({"domain_performance_patterns": patterns_table})

        # Log enhanced recommendations
        if result.error_recommendations:
            enhanced_rec_html = "<h4>Enhanced Error Analysis Recommendations</h4>"
            enhanced_rec_html += "<ul style='margin: 0; padding-left: 20px;'>"
            for rec in result.error_recommendations:
                enhanced_rec_html += f"<li>{rec}</li>"
            enhanced_rec_html += "</ul>"
            wandb.log({"enhanced_error_recommendations": wandb.Html(enhanced_rec_html)})

    def _log_interactive_charts(self, result: AnalysisResult) -> None:
        """Log interactive charts for better analysis visualization."""
        # Performance distribution histogram
        if result.retrieval_results:
            ap_scores = [rr.ap_score for rr in result.retrieval_results]
            perf_hist = wandb.plot.histogram(
                wandb.Table(data=[[s] for s in ap_scores], columns=["AP Score"]),
                "AP Score",
                title="Retrieval Performance Distribution"
            )
            wandb.log({"performance_distribution": perf_hist})

        # Query length vs performance scatter
        if result.query_analyses and result.retrieval_results:
            scatter_data = []
            for qa, rr in zip(result.query_analyses, result.retrieval_results):
                scatter_data.append([qa.query_length, rr.ap_score])

            if scatter_data:
                scatter_table = wandb.Table(data=scatter_data, columns=["Query Length", "AP Score"])
                scatter_plot = wandb.plot.scatter(
                    scatter_table,
                    "Query Length",
                    "AP Score",
                    title="Query Length vs Performance"
                )
                wandb.log({"query_length_performance": scatter_plot})

        # Domain performance comparison
        if result.domain_distribution:
            domain_data = [[domain, count] for domain, count in result.domain_distribution.items()]
            if domain_data:
                domain_table = wandb.Table(data=domain_data, columns=["Domain", "Count"])
                domain_bar = wandb.plot.bar(
                    domain_table,
                    "Domain",
                    "Count",
                    title="Queries by Scientific Domain"
                )
                wandb.log({"domain_distribution_chart": domain_bar})

    def _log_custom_visualizations(self, result: AnalysisResult) -> None:
        """Log custom visualizations for advanced analysis."""
        # Precision@K comparison
        if result.precision_at_k:
            precision_data = []
            for k in sorted(result.precision_at_k.keys()):
                precision_data.append([f"P@{k}", result.precision_at_k[k]])

            if precision_data:
                prec_table = wandb.Table(data=precision_data, columns=["Metric", "Value"])
                prec_bar = wandb.plot.bar(
                    prec_table,
                    "Metric",
                    "Value",
                    title="Precision@K Comparison"
                )
                wandb.log({"precision_at_k_comparison": prec_bar})

        # Error categories breakdown
        if result.error_categories:
            error_data = [[err_type, count] for err_type, count in result.error_categories.items()]
            if error_data:
                error_table = wandb.Table(data=error_data, columns=["Error Type", "Count"])
                error_bar = wandb.plot.bar(
                    error_table,
                    "Error Type",
                    "Count",
                    title="Error Categories Distribution"
                )
                wandb.log({"error_categories_chart": error_bar})

        # Performance segmentation (Phase 3)
        if result.performance_segmentation:
            segment_data = [[level, count] for level, count in result.performance_segmentation.items()]
            if segment_data:
                segment_table = wandb.Table(data=segment_data, columns=["Performance Level", "Count"])
                segment_bar = wandb.plot.bar(
                    segment_table,
                    "Performance Level",
                    "Count",
                    title="Performance Segmentation"
                )
                wandb.log({"performance_segmentation_chart": segment_bar})

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

        # Create interactive trend charts
        self._create_trend_charts(results, precision_trends)

        # Create comparative analysis
        self._create_comparative_analysis(results)

        # Create drill-down capabilities
        self._create_drill_down_views(results)

    def _create_trend_charts(self, results: List[AnalysisResult], precision_trends: Dict[int, List[float]]) -> None:
        """Create interactive trend charts for performance metrics."""
        # MAP score trend
        map_data = [[f"Run_{i+1}", score] for i, score in enumerate([r.map_score for r in results])]
        if map_data:
            map_table = wandb.Table(data=map_data, columns=["Run", "MAP Score"])
            map_line = wandb.plot.line(
                map_table,
                "Run",
                "MAP Score",
                title="MAP Score Trend Across Runs"
            )
            wandb.log({"map_score_trend": map_line})

        # Precision@K trends
        for k, precisions in precision_trends.items():
            trend_data = [[f"Run_{i+1}", p] for i, p in enumerate(precisions)]
            if trend_data:
                trend_table = wandb.Table(data=trend_data, columns=["Run", f"P@{k}"])
                trend_line = wandb.plot.line(
                    trend_table,
                    "Run",
                    f"P@{k}",
                    title=f"Precision@{k} Trend Across Runs"
                )
                wandb.log({f"precision_at_{k}_trend": trend_line})

    def _create_comparative_analysis(self, results: List[AnalysisResult]) -> None:
        """Create comparative analysis views across different runs."""
        if len(results) < 2:
            return

        # Compare key metrics across runs
        comparison_data = []
        for i, result in enumerate(results):
            comparison_data.append([
                f"Run_{i+1}",
                result.map_score,
                result.retrieval_success_rate,
                result.avg_query_length,
                result.rewrite_rate
            ])

        if comparison_data:
            comp_table = wandb.Table(
                data=comparison_data,
                columns=["Run", "MAP", "Success Rate", "Avg Query Length", "Rewrite Rate"]
            )

            # Log as bar chart for comparison
            for metric in ["MAP", "Success Rate", "Avg Query Length", "Rewrite Rate"]:
                comp_bar = wandb.plot.bar(
                    comp_table,
                    "Run",
                    metric,
                    title=f"{metric} Comparison Across Runs"
                )
                wandb.log({f"comparison_{metric.lower().replace(' ', '_')}": comp_bar})

    def _create_drill_down_views(self, results: List[AnalysisResult]) -> None:
        """Create drill-down views for detailed analysis."""
        # Domain performance across runs
        domain_trends = {}
        for i, result in enumerate(results):
            if result.domain_distribution:
                for domain, count in result.domain_distribution.items():
                    if domain not in domain_trends:
                        domain_trends[domain] = []
                    domain_trends[domain].append((f"Run_{i+1}", count))

        # Create domain trend charts
        for domain, trend_data in domain_trends.items():
            if len(trend_data) > 1:
                domain_data = [[run, count] for run, count in trend_data]
                domain_table = wandb.Table(data=domain_data, columns=["Run", "Count"])
                domain_line = wandb.plot.line(
                    domain_table,
                    "Run",
                    "Count",
                    title=f"{domain} Query Distribution Trend"
                )
                wandb.log({f"domain_trend_{domain.lower()}": domain_line})

        # Error analysis trends
        error_trends = {}
        for i, result in enumerate(results):
            if result.error_categories:
                for error_type, count in result.error_categories.items():
                    if error_type not in error_trends:
                        error_trends[error_type] = []
                    error_trends[error_type].append((f"Run_{i+1}", count))

        # Create error trend charts
        for error_type, trend_data in error_trends.items():
            if len(trend_data) > 1:
                error_data = [[run, count] for run, count in trend_data]
                error_table = wandb.Table(data=error_data, columns=["Run", "Count"])
                error_line = wandb.plot.line(
                    error_table,
                    "Run",
                    "Count",
                    title=f"{error_type} Error Trend"
                )
                wandb.log({f"error_trend_{error_type.lower().replace(' ', '_')}": error_line})

    def create_custom_dashboard(
        self,
        results: List[AnalysisResult],
        chart_configs: List[Dict[str, Any]]
    ) -> None:
        """
        Create custom dashboard based on user-specified chart configurations.

        Args:
            results: List of AnalysisResult objects
            chart_configs: List of chart configuration dictionaries
        """
        for config in chart_configs:
            chart_type = config.get("type", "bar")
            title = config.get("title", "Custom Chart")
            x_field = config.get("x_field")
            y_field = config.get("y_field")
            data_source = config.get("data_source", "results")

            if not x_field or not y_field:
                continue

            # Extract data based on configuration
            chart_data = self._extract_chart_data(results, config)

            if chart_data:
                chart_table = wandb.Table(data=chart_data, columns=[x_field, y_field])

                if chart_type == "bar":
                    chart = wandb.plot.bar(chart_table, x_field, y_field, title=title)
                elif chart_type == "line":
                    chart = wandb.plot.line(chart_table, x_field, y_field, title=title)
                elif chart_type == "scatter":
                    chart = wandb.plot.scatter(chart_table, x_field, y_field, title=title)
                else:
                    continue

                # Create a safe name for logging
                safe_name = title.lower().replace(" ", "_").replace("@", "at")
                wandb.log({f"custom_{safe_name}": chart})

    def _extract_chart_data(self, results: List[AnalysisResult], config: Dict[str, Any]) -> List[List]:
        """Extract data for custom charts based on configuration."""
        data_source = config.get("data_source", "results")
        x_field = config.get("x_field")
        y_field = config.get("y_field")

        chart_data = []

        if data_source == "results":
            for i, result in enumerate(results):
                if x_field and y_field:
                    x_value = self._get_nested_value(result, x_field)
                    y_value = self._get_nested_value(result, y_field)

                    if x_value is not None and y_value is not None:
                        if x_field == "run":
                            x_value = f"Run_{i+1}"
                        chart_data.append([x_value, y_value])

        elif data_source == "queries":
            for result in results:
                for qa, rr in zip(result.query_analyses, result.retrieval_results):
                    if x_field and y_field:
                        x_value = self._get_nested_value(qa, x_field) or self._get_nested_value(rr, x_field)
                        y_value = self._get_nested_value(qa, y_field) or self._get_nested_value(rr, y_field)

                        if x_value is not None and y_value is not None:
                            chart_data.append([x_value, y_value])

        return chart_data

    def _get_nested_value(self, obj: Any, field_path: str) -> Any:
        """Get nested value from object using dot notation."""
        try:
            for part in field_path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    return None
            return obj
        except:
            return None

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
