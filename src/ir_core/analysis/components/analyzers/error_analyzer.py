# src/ir_core/analysis/components/analyzers/error_analyzer.py

"""
Error Analyzer Component

Comprehensive error analyzer for Scientific QA retrieval system with
pattern detection, domain analysis, and automated recommendations.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Sequence
from dataclasses import dataclass
from omegaconf import DictConfig

from ...constants import (
    ERROR_ANALYSIS_THRESHOLDS,
    ERROR_CATEGORIES,
    PATTERN_DETECTION_CONFIG,
    SCIENTIFIC_TERMS
)


@dataclass
class ErrorAnalysisResult:
    """Comprehensive result of error analysis with detailed categorization."""
    retrieval_success_rate: float
    error_categories: Dict[str, int]
    query_understanding_failures: Dict[str, int]
    retrieval_failures: Dict[str, int]
    system_failures: Dict[str, int]
    error_patterns: Dict[str, Any]
    domain_error_rates: Dict[str, float]
    temporal_trends: Dict[str, Any]
    error_recommendations: List[str]


class ErrorAnalyzer:
    """
    Comprehensive error analyzer for Scientific QA retrieval system.

    Analyzes retrieval errors across multiple dimensions:
    - Query Understanding Failures (ambiguous, out-of-domain, complex queries)
    - Retrieval Failures (false positives, false negatives, ranking errors)
    - System Failures (timeouts, parsing errors, infrastructure issues)
    - Pattern Detection (common failure patterns, correlations)
    - Temporal Analysis (performance trends over time)
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the error analyzer.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})

    def analyze_errors(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str],
        original_queries: List[str],
        query_domains: Optional[List[List[str]]] = None,
        timestamps: Optional[List[str]] = None
    ) -> ErrorAnalysisResult:
        """
        Perform comprehensive error analysis.

        Args:
            predicted_docs_list: Predicted documents for each query
            ground_truth_ids: Ground truth IDs
            original_queries: Original query strings
            query_domains: Domain classifications for each query
            timestamps: Timestamps for temporal analysis

        Returns:
            ErrorAnalysisResult: Comprehensive error analysis results
        """
        # Initialize error tracking
        query_understanding_failures = {
            "ambiguous_query": 0,
            "out_of_domain": 0,
            "complex_multi_concept": 0
        }

        retrieval_failures = {
            "false_positive": 0,
            "false_negative": 0,
            "ranking_error": 0
        }

        system_failures = {
            "timeout_error": 0,
            "parsing_error": 0,
            "infrastructure_error": 0
        }

        success_count = 0
        total_queries = len(predicted_docs_list)

        # Analyze each query
        for i, (pred_docs, gt_id, query) in enumerate(zip(predicted_docs_list, ground_truth_ids, original_queries)):
            query_domain = query_domains[i] if query_domains and i < len(query_domains) else ["unknown"]

            # Check for system failures first
            if not pred_docs:
                system_failures["infrastructure_error"] += 1
                continue

            # Check if ground truth is in top 10
            top_10_ids = [doc.get("id", "") for doc in pred_docs[:10]]
            is_successful = gt_id in top_10_ids

            if is_successful:
                success_count += 1
                continue

            # Analyze failure reasons
            error_type = self._categorize_error(pred_docs, gt_id, query, query_domain)
            category = ERROR_CATEGORIES[error_type]["category"]

            if category == "query_understanding":
                query_understanding_failures[error_type] += 1
            elif category == "retrieval":
                retrieval_failures[error_type] += 1
            elif category == "system":
                system_failures[error_type] += 1

        # Calculate rates
        retrieval_success_rate = success_count / total_queries if total_queries > 0 else 0

        # Legacy error categories for backward compatibility
        error_categories = {
            "successful": success_count,
            "failed": total_queries - success_count
        }

        # Add non-zero error types
        for error_type, count in query_understanding_failures.items():
            if count > 0:
                error_categories[f"query_{error_type}"] = count

        for error_type, count in retrieval_failures.items():
            if count > 0:
                error_categories[f"retrieval_{error_type}"] = count

        for error_type, count in system_failures.items():
            if count > 0:
                error_categories[f"system_{error_type}"] = count

        # Pattern detection
        error_patterns = self._detect_patterns(predicted_docs_list, ground_truth_ids, original_queries, query_domains)

        # Domain-specific error rates
        domain_error_rates = self._calculate_domain_error_rates(predicted_docs_list, ground_truth_ids, query_domains)

        # Temporal trends analysis
        temporal_trends = self._analyze_temporal_trends(retrieval_success_rate, error_patterns)

        # Generate recommendations
        recommendations = self._generate_error_recommendations(
            query_understanding_failures, retrieval_failures, system_failures,
            error_patterns, domain_error_rates, retrieval_success_rate
        )

        return ErrorAnalysisResult(
            retrieval_success_rate=retrieval_success_rate,
            error_categories=error_categories,
            query_understanding_failures=query_understanding_failures,
            retrieval_failures=retrieval_failures,
            system_failures=system_failures,
            error_patterns=error_patterns,
            domain_error_rates=domain_error_rates,
            temporal_trends=temporal_trends,
            error_recommendations=recommendations
        )

    def _categorize_error(
        self,
        pred_docs: List[Dict[str, Any]],
        gt_id: str,
        query: str,
        query_domain: List[str]
    ) -> str:
        """
        Categorize the type of error for a failed retrieval.

        Args:
            pred_docs: Predicted documents
            gt_id: Ground truth ID
            query: Original query
            query_domain: Query domain classification

        Returns:
            str: Error category key
        """
        query_lower = query.lower()

        # Check for ambiguous query (multiple domains or vague terms)
        domain_count = len([d for d in query_domain if d != "unknown"])
        if domain_count > 2:
            return "complex_multi_concept"

        # Check for out-of-domain
        if "unknown" in query_domain or domain_count == 0:
            return "out_of_domain"

        # Check for ambiguous terms
        ambiguous_indicators = ["무엇", "어떻게", "왜", "어디", "언제"]
        if any(indicator in query_lower for indicator in ambiguous_indicators):
            return "ambiguous_query"

        # Check retrieval quality
        top_scores = [doc.get("score", 0.0) for doc in pred_docs[:3]]
        avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        # Check if ground truth has low score (false negative)
        gt_doc = next((doc for doc in pred_docs if doc.get("id") == gt_id), None)
        if gt_doc and gt_doc.get("score", 0.0) < ERROR_ANALYSIS_THRESHOLDS["false_negative_threshold"]:
            return "false_negative"

        # Check for false positives (high score but wrong)
        if avg_top_score > ERROR_ANALYSIS_THRESHOLDS["false_positive_threshold"]:
            return "false_positive"

        # Check ranking error
        gt_rank = next((i for i, doc in enumerate(pred_docs) if doc.get("id") == gt_id), -1)
        if gt_rank > 5:  # Ground truth not in top 5
            return "ranking_error"

        # Default to false negative
        return "false_negative"

    def _detect_patterns(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str],
        original_queries: List[str],
        query_domains: Optional[List[List[str]]]
    ) -> Dict[str, Any]:
        """
        Detect comprehensive error patterns and correlations.

        Args:
            predicted_docs_list: All predicted documents
            ground_truth_ids: All ground truth IDs
            original_queries: All queries
            query_domains: All query domains

        Returns:
            Dict[str, Any]: Comprehensive pattern analysis
        """
        patterns = {
            "query_length_correlation": 0.0,
            "complexity_success_correlation": 0.0,
            "domain_performance_patterns": {},
            "query_type_performance": {},
            "temporal_patterns": {},
            "failure_mode_clusters": [],
            "recommendation_patterns": []
        }

        if not original_queries or len(original_queries) < PATTERN_DETECTION_CONFIG["min_pattern_occurrences"]:
            return patterns

        # Query length vs success correlation
        query_lengths = [len(q) for q in original_queries]
        successes = []

        for pred_docs, gt_id in zip(predicted_docs_list, ground_truth_ids):
            top_10_ids = [doc.get("id", "") for doc in pred_docs[:10]] if pred_docs else []
            successes.append(1 if gt_id in top_10_ids else 0)

        if query_lengths and successes:
            patterns["query_length_correlation"] = self._calculate_correlation(query_lengths, [float(s) for s in successes])

        # Domain-specific patterns
        if query_domains:
            patterns["domain_performance_patterns"] = self._analyze_domain_patterns(
                predicted_docs_list, ground_truth_ids, query_domains
            )

        # Query type analysis
        patterns["query_type_performance"] = self._analyze_query_type_patterns(
            original_queries, successes
        )

        # Failure mode clustering
        patterns["failure_mode_clusters"] = self._cluster_failure_modes(
            predicted_docs_list, ground_truth_ids, original_queries, query_domains
        )

        return patterns

    def _calculate_domain_error_rates(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str],
        query_domains: Optional[List[List[str]]]
    ) -> Dict[str, float]:
        """
        Calculate error rates by domain.

        Args:
            predicted_docs_list: All predicted documents
            ground_truth_ids: All ground truth IDs
            query_domains: All query domains

        Returns:
            Dict[str, float]: Domain-specific error rates
        """
        domain_stats = {}

        if not query_domains:
            return domain_stats

        for pred_docs, gt_id, domains in zip(predicted_docs_list, ground_truth_ids, query_domains):
            top_10_ids = [doc.get("id", "") for doc in pred_docs[:10]] if pred_docs else []
            is_success = gt_id in top_10_ids

            for domain in domains:
                if domain not in domain_stats:
                    domain_stats[domain] = {"total": 0, "errors": 0}
                domain_stats[domain]["total"] += 1
                if not is_success:
                    domain_stats[domain]["errors"] += 1

        # Calculate error rates
        domain_error_rates = {}
        for domain, stats in domain_stats.items():
            if stats["total"] > 0:
                domain_error_rates[domain] = stats["errors"] / stats["total"]

        return domain_error_rates

    def _generate_error_recommendations(
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

    def _analyze_temporal_trends(self, current_success_rate: float, error_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze temporal trends in performance (framework for future implementation).

        Args:
            current_success_rate: Current retrieval success rate
            error_patterns: Current error patterns

        Returns:
            Dict[str, Any]: Temporal analysis results
        """
        # This is a placeholder for temporal analysis
        # In a full implementation, this would:
        # 1. Load historical performance data
        # 2. Calculate trends over time
        # 3. Detect performance degradation
        # 4. Identify seasonal patterns

        trends = {
            "current_success_rate": current_success_rate,
            "success_rate_trend": [current_success_rate],  # Would be historical data
            "performance_stability": "stable",  # stable, improving, degrading
            "degradation_detected": False,
            "trend_direction": "stable",  # improving, degrading, stable
            "recommendations": []
        }

        # Simple degradation detection based on current patterns
        # In practice, this would compare against historical baselines
        if error_patterns.get("query_length_correlation", 0.0) < -0.5:
            trends["performance_stability"] = "degrading"
            trends["degradation_detected"] = True
            trends["trend_direction"] = "degrading"
            trends["recommendations"].append("Performance degradation detected. Query length correlation suggests worsening performance for complex queries.")

        # Check for domain-specific degradation
        domain_patterns = error_patterns.get("domain_performance_patterns", {})
        degrading_domains = []
        for domain, pattern in domain_patterns.items():
            if pattern.get("needs_improvement", False):
                degrading_domains.append(domain)

        if degrading_domains:
            trends["degradation_detected"] = True
            trends["recommendations"].append(f"Domain-specific degradation detected in: {', '.join(degrading_domains)}")

        return trends

    def _calculate_correlation(self, x_values: Sequence[Union[int, float]], y_values: Sequence[Union[int, float]]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        mean_x = sum(x_values) / len(x_values)
        mean_y = sum(y_values) / len(y_values)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        denominator = (sum((x - mean_x)**2 for x in x_values) *
                      sum((y - mean_y)**2 for y in y_values))**0.5

        return numerator / denominator if denominator != 0 else 0.0

    def _analyze_domain_patterns(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str],
        query_domains: List[List[str]]
    ) -> Dict[str, Any]:
        """Analyze performance patterns by domain."""
        domain_stats = {}

        for pred_docs, gt_id, domains in zip(predicted_docs_list, ground_truth_ids, query_domains):
            top_10_ids = [doc.get("id", "") for doc in pred_docs[:10]] if pred_docs else []
            is_success = gt_id in top_10_ids

            for domain in domains:
                if domain not in domain_stats:
                    domain_stats[domain] = {"total": 0, "successes": 0, "failures": 0}
                domain_stats[domain]["total"] += 1
                if is_success:
                    domain_stats[domain]["successes"] += 1
                else:
                    domain_stats[domain]["failures"] += 1

        # Calculate patterns
        patterns = {}
        for domain, stats in domain_stats.items():
            if stats["total"] >= PATTERN_DETECTION_CONFIG["min_pattern_occurrences"]:
                success_rate = stats["successes"] / stats["total"]
                patterns[domain] = {
                    "success_rate": success_rate,
                    "sample_size": stats["total"],
                    "needs_improvement": success_rate < ERROR_ANALYSIS_THRESHOLDS["domain_error_threshold"]
                }

        return patterns

    def _analyze_query_type_patterns(
        self,
        queries: List[str],
        successes: List[int]
    ) -> Dict[str, Any]:
        """Analyze performance patterns by query type."""
        from ...constants import QUERY_TYPE_PATTERNS

        query_types = []
        for query in queries:
            query_type = "general"
            for type_name, pattern in QUERY_TYPE_PATTERNS.items():
                if pattern.search(query):
                    query_type = type_name
                    break
            query_types.append(query_type)

        # Calculate success rates by query type
        type_stats = {}
        for q_type, success in zip(query_types, successes):
            if q_type not in type_stats:
                type_stats[q_type] = {"total": 0, "successes": 0}
            type_stats[q_type]["total"] += 1
            type_stats[q_type]["successes"] += success

        patterns = {}
        for q_type, stats in type_stats.items():
            if stats["total"] >= PATTERN_DETECTION_CONFIG["min_pattern_occurrences"]:
                patterns[q_type] = {
                    "success_rate": stats["successes"] / stats["total"],
                    "sample_size": stats["total"]
                }

        return patterns

    def _cluster_failure_modes(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str],
        queries: List[str],
        query_domains: Optional[List[List[str]]]
    ) -> List[Dict[str, Any]]:
        """Cluster similar failure modes for pattern analysis."""
        failures = []

        for i, (pred_docs, gt_id, query) in enumerate(zip(predicted_docs_list, ground_truth_ids, queries)):
            top_10_ids = [doc.get("id", "") for doc in pred_docs[:10]] if pred_docs else []
            if gt_id not in top_10_ids:
                # This is a failure
                domain = query_domains[i][0] if query_domains and i < len(query_domains) else "unknown"
                failures.append({
                    "query_length": len(query),
                    "domain": domain,
                    "has_numbers": any(c.isdigit() for c in query),
                    "is_scientific": any(term in query.lower() for term in SCIENTIFIC_TERMS),
                    "query_type": self._classify_query_type(query)
                })

        # Simple clustering based on common characteristics
        clusters = []
        if len(failures) >= PATTERN_DETECTION_CONFIG["min_pattern_occurrences"]:
            # Group by domain
            domain_groups = {}
            for failure in failures:
                domain = failure["domain"]
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(failure)

            for domain, group in domain_groups.items():
                if len(group) >= PATTERN_DETECTION_CONFIG["min_pattern_occurrences"]:
                    clusters.append({
                        "cluster_type": "domain_failure",
                        "domain": domain,
                        "size": len(group),
                        "avg_query_length": sum(f["query_length"] for f in group) / len(group),
                        "scientific_ratio": sum(f["is_scientific"] for f in group) / len(group)
                    })

        return clusters

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for pattern analysis."""
        from ...constants import QUERY_TYPE_PATTERNS

        for type_name, pattern in QUERY_TYPE_PATTERNS.items():
            if pattern.search(query):
                return type_name
        return "general"
