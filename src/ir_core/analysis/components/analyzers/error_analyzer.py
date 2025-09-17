# src/ir_core/analysis/components/analyzers/error_analyzer.py

"""
Error Analyzer Component
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from omegaconf import DictConfig

# Import extracted components
from .error_categorizer import ErrorCategorizer
from .pattern_detector import PatternDetector
from .recommendation_generator import RecommendationGenerator
from .temporal_analyzer import TemporalAnalyzer


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
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the error analyzer.
        """
        self.config = config or DictConfig({})
        self.top_k_check = 10

        # Initialize extracted components
        self.error_categorizer = ErrorCategorizer(config)
        self.pattern_detector = PatternDetector(config)
        self.recommendation_generator = RecommendationGenerator(config)
        self.temporal_analyzer = TemporalAnalyzer(config)

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

            # Check if ground truth is in top K
            top_k_ids = []
            for doc in pred_docs[:self.top_k_check]:
                if isinstance(doc, dict):
                    top_k_ids.append(doc.get("id", ""))
                elif isinstance(doc, str):
                    top_k_ids.append(doc)
                else:
                    top_k_ids.append(str(doc) if doc else "")
            is_successful = gt_id in top_k_ids

            if is_successful:
                success_count += 1
                continue

            # Analyze failure reasons using extracted component
            error_type = self.error_categorizer.categorize_error(pred_docs, gt_id, query, query_domain)

            # Simple categorization for now
            if "query" in error_type:
                query_understanding_failures[error_type] += 1
            elif "false" in error_type or "ranking" in error_type:
                retrieval_failures[error_type] += 1
            else:
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

        # Use extracted components for complex analysis
        error_patterns = self.pattern_detector.detect_patterns(predicted_docs_list, ground_truth_ids, original_queries, query_domains)
        domain_error_rates = self.pattern_detector.calculate_domain_error_rates(predicted_docs_list, ground_truth_ids, query_domains, self.top_k_check)
        temporal_trends = self.temporal_analyzer.analyze_temporal_trends(retrieval_success_rate, error_patterns)

        recommendations = self.recommendation_generator.generate_error_recommendations(
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
