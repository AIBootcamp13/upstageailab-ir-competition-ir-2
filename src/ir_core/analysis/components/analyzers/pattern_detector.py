# src/ir_core/analysis/components/analyzers/pattern_detector.py

"""
Pattern Detector Component

Handles pattern detection and analysis for the Scientific QA retrieval system.
Identifies common failure patterns, correlations, and performance trends.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Sequence
from omegaconf import DictConfig

from ...constants import (
    PATTERN_DETECTION_CONFIG,
    SCIENTIFIC_TERMS
)


class PatternDetector:
    """
    Detects patterns in retrieval errors and performance data.

    Analyzes retrieval results to identify:
    - Query length vs success correlations
    - Domain-specific performance patterns
    - Query type performance patterns
    - Failure mode clusters
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the pattern detector.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})

    def detect_patterns(
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
                    "needs_improvement": success_rate < 0.7  # Using 70% as threshold
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
                    "query_type": self.classify_query_type(query)
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

    def calculate_domain_error_rates(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str],
        query_domains: Optional[List[List[str]]],
        top_k_check: int = 10
    ) -> Dict[str, float]:
        """
        Calculate error rates by domain.

        Args:
            predicted_docs_list: All predicted documents
            ground_truth_ids: All ground truth IDs
            query_domains: All query domains
            top_k_check: Number of top results to check for success

        Returns:
            Dict[str, float]: Domain-specific error rates
        """
        domain_stats = {}

        if not query_domains:
            return domain_stats

        for pred_docs, gt_id, domains in zip(predicted_docs_list, ground_truth_ids, query_domains):
            top_k_ids = []
            if pred_docs:
                for doc in pred_docs[:top_k_check]:
                    if isinstance(doc, dict):
                        top_k_ids.append(doc.get("id", ""))
                    elif isinstance(doc, str):
                        top_k_ids.append(doc)
                    else:
                        top_k_ids.append(str(doc) if doc else "")
            is_success = gt_id in top_k_ids

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

    def classify_query_type(self, query: str) -> str:
        """Classify query type for pattern analysis."""
        from ...constants import QUERY_TYPE_PATTERNS

        for type_name, pattern in QUERY_TYPE_PATTERNS.items():
            if pattern.search(query):
                return type_name
        return "general"