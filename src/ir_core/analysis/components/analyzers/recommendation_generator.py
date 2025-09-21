# src/ir_core/analysis/components/analyzers/recommendation_generator.py

"""
Recommendation Generator Component

Generates actionable recommendations based on error analysis and patterns
for the Scientific QA retrieval system.
"""

from typing import Dict, List, Any, Optional
from omegaconf import DictConfig

from ...constants import (
    ERROR_ANALYSIS_THRESHOLDS,
    PATTERN_DETECTION_CONFIG
)


class RecommendationGenerator:
    """
    Generates actionable recommendations based on analysis results.

    Analyzes error patterns and performance metrics to provide:
    - Query understanding improvements
    - Retrieval algorithm optimizations
    - System performance enhancements
    - Domain-specific recommendations
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the recommendation generator.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})

    def generate_error_recommendations(
        self,
        query_understanding_failures: Dict[str, int],
        retrieval_failures: Dict[str, int],
        system_failures: Dict[str, int],
        error_patterns: Dict[str, Any],
        domain_error_rates: Dict[str, float],
        success_rate: float
    ) -> List[str]:
        """
        Generate actionable recommendations based on error analysis and patterns.

        Args:
            query_understanding_failures: Query understanding error counts
            retrieval_failures: Retrieval error counts
            system_failures: System error counts
            error_patterns: Detected patterns
            domain_error_rates: Domain-specific error rates
            success_rate: Overall success rate

        Returns:
            List[str]: Comprehensive recommendations
        """
        recommendations = []

        # Overall performance recommendations
        if success_rate < 0.7:
            recommendations.append("Overall retrieval success rate is below 70%. Consider improving the retrieval algorithm or expanding the document collection.")

        # Query understanding recommendations
        if query_understanding_failures.get("ambiguous_query", 0) > 0:
            recommendations.append("Detected ambiguous queries. Consider implementing query clarification or multi-turn dialogue support.")

        if query_understanding_failures.get("out_of_domain", 0) > 0:
            recommendations.append("Some queries fall outside the system's knowledge domain. Consider expanding domain coverage or implementing out-of-domain detection.")

        if query_understanding_failures.get("complex_multi_concept", 0) > 0:
            recommendations.append("Complex multi-concept queries are failing. Consider query decomposition or specialized handling for compound scientific questions.")

        # Retrieval recommendations
        if retrieval_failures.get("false_positive", 0) > 0:
            recommendations.append("High false positive rate detected. Review document ranking algorithm and relevance scoring.")

        if retrieval_failures.get("false_negative", 0) > 0:
            recommendations.append("False negatives detected. Consider improving document indexing or search term expansion.")

        if retrieval_failures.get("ranking_error", 0) > 0:
            recommendations.append("Ranking errors found. Ground truth documents not appearing in top positions. Review ranking algorithm.")

        # System recommendations
        if system_failures.get("timeout_error", 0) > 0:
            recommendations.append("Timeout errors detected. Consider optimizing query processing time or increasing system resources.")

        if system_failures.get("parsing_error", 0) > 0:
            recommendations.append("Parsing errors found. Review input validation and error handling in query processing.")

        if system_failures.get("infrastructure_error", 0) > 0:
            recommendations.append("Infrastructure errors detected. Check system connectivity, resource availability, and service health.")

        # Pattern-based recommendations
        query_length_corr = error_patterns.get("query_length_correlation", 0.0)
        if abs(query_length_corr) > 0.3:
            if query_length_corr > 0:
                recommendations.append("Longer queries tend to succeed more. Consider encouraging detailed query formulation.")
            else:
                recommendations.append("Longer queries tend to fail more. Consider query simplification or segmentation for complex queries.")

        # Domain-specific recommendations
        high_error_domains = [domain for domain, rate in domain_error_rates.items()
                            if rate > ERROR_ANALYSIS_THRESHOLDS["domain_error_threshold"]]
        if high_error_domains:
            recommendations.append(f"High error rates detected in domains: {', '.join(high_error_domains)}. Consider domain-specific improvements.")

        # Query type performance recommendations
        query_type_patterns = error_patterns.get("query_type_performance", {})
        if query_type_patterns:
            low_performance_types = []
            for q_type, pattern in query_type_patterns.items():
                if pattern.get("success_rate", 1.0) < 0.6 and pattern.get("sample_size", 0) >= PATTERN_DETECTION_CONFIG["min_pattern_occurrences"]:
                    low_performance_types.append(q_type)

            if low_performance_types:
                recommendations.append(f"Poor performance detected for query types: {', '.join(low_performance_types)}. Consider specialized handling for these query types.")

        # Failure cluster recommendations
        failure_clusters = error_patterns.get("failure_mode_clusters", [])
        if failure_clusters:
            for cluster in failure_clusters:
                if cluster.get("size", 0) >= PATTERN_DETECTION_CONFIG["min_pattern_occurrences"]:
                    cluster_type = cluster.get("cluster_type", "")
                    domain = cluster.get("domain", "")
                    if cluster_type == "domain_failure":
                        recommendations.append(f"Cluster of failures detected in {domain} domain. Consider reviewing domain-specific document quality and indexing.")

        # A/B testing recommendations
        if len(high_error_domains) > 1:
            recommendations.append("Multiple domains showing high error rates. Consider A/B testing different retrieval strategies across domains.")

        if query_length_corr != 0.0 and abs(query_length_corr) > 0.2:
            recommendations.append("Query length shows correlation with success. Consider A/B testing query preprocessing strategies.")

        if not recommendations:
            recommendations.append("Error analysis complete. No critical issues detected. Consider monitoring for emerging patterns.")

        return recommendations

    def generate_performance_recommendations(
        self,
        map_score: float,
        retrieval_success_rate: float,
        rewrite_rate: float
    ) -> List[str]:
        """
        Generate recommendations based on overall performance metrics.

        Args:
            map_score: Mean Average Precision
            retrieval_success_rate: Success rate
            rewrite_rate: Rewrite rate

        Returns:
            List[str]: Performance recommendations
        """
        from ...constants import ANALYSIS_THRESHOLDS

        recommendations = []

        if map_score < ANALYSIS_THRESHOLDS["map_score_low"]:
            recommendations.append("MAP score is below 0.5. Consider improving retrieval algorithm or expanding document collection.")

        if retrieval_success_rate < ANALYSIS_THRESHOLDS["retrieval_success_rate_low"]:
            recommendations.append("Retrieval success rate is below 70%. Focus on improving top-10 retrieval accuracy.")

        if rewrite_rate > ANALYSIS_THRESHOLDS["rewrite_rate_high"]:
            recommendations.append("High rewrite rate detected. Verify that query rewriting is improving rather than degrading performance.")

        if rewrite_rate < ANALYSIS_THRESHOLDS["rewrite_rate_low"]:
            recommendations.append("Low rewrite rate. Consider enabling query rewriting to improve retrieval for conversational queries.")

        if not recommendations:
            recommendations.append("Overall performance looks good. Consider fine-tuning hyperparameters for marginal improvements.")

        return recommendations