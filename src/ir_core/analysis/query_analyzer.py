# src/ir_core/analysis/query_analyzer.py

"""
Query analysis module for the Scientific QA retrieval system.

This module provides comprehensive analysis of query characteristics,
including complexity scoring, domain classification, query type detection,
and scientific term extraction.

Note: This is a refactored version that uses the modular QueryAnalysisService
for better maintainability and separation of concerns.
"""

from typing import Dict, List, Any, Optional, Tuple
from omegaconf import DictConfig

from .query_analysis_service import QueryAnalysisService


class QueryAnalyzer:
    """
    Advanced query analyzer for Scientific QA queries.

    This class now serves as a facade that delegates to the modular
    QueryAnalysisService for better maintainability and testability.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the query analyzer.

        Args:
            config: Optional configuration for analysis parameters
        """
        self.service = QueryAnalysisService(config)

    def analyze_query(self, query: str):
        """
        Analyze a single query and extract comprehensive features.

        Args:
            query: The query string to analyze

        Returns:
            QueryFeatures: Detailed analysis results
        """
        return self.service.analyze_query(query)

    def analyze_batch(self, queries: List[str], max_workers: Optional[int] = None):
        """
        Analyze a batch of queries with optional parallel processing.

        Args:
            queries: List of query strings
            max_workers: Maximum number of worker threads. If None, uses min(32, len(queries))

        Returns:
            List[QueryFeatures]: Analysis results for each query
        """
        return self.service.analyze_batch(queries, max_workers)

    def measure_rewrite_effectiveness(
        self,
        original_query: str,
        rewritten_query: str
    ) -> Dict[str, Any]:
        """
        Measure the effectiveness of query rewriting.

        Args:
            original_query: The original query
            rewritten_query: The rewritten query

        Returns:
            Dict[str, Any]: Effectiveness metrics
        """
        return self.service.measure_rewrite_effectiveness(original_query, rewritten_query)

    def create_domain_validation_set(
        self,
        num_queries_per_domain: int = 10,
        use_llm: bool = True,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a validation set for domain classification using LLM-generated queries.

        Args:
            num_queries_per_domain: Number of queries to generate per domain
            use_llm: Whether to use LLM for query generation
            max_workers: Maximum number of worker threads for parallel domain processing

        Returns:
            List[Dict[str, Any]]: Validation set with queries and expected domains
        """
        return self.service.create_domain_validation_set(
            num_queries_per_domain, use_llm, max_workers
        )

    def evaluate_domain_classification(
        self,
        validation_set: List[Dict[str, Any]],
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate domain classification accuracy against a validation set.

        Args:
            validation_set: List of queries with expected domains
            max_workers: Maximum number of worker threads for parallel processing

        Returns:
            Dict[str, Any]: Evaluation metrics and detailed results
        """
        return self.service.evaluate_domain_classification(validation_set, max_workers)

    def perform_complete_analysis(
        self,
        queries: List[str],
        include_validation: bool = False,
        num_validation_queries: int = 10,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform complete analysis including feature extraction and optional validation.

        Args:
            queries: List of queries to analyze
            include_validation: Whether to include domain classification validation
            num_validation_queries: Number of validation queries per domain
            max_workers: Maximum number of worker threads

        Returns:
            Dict[str, Any]: Complete analysis results
        """
        return self.service.perform_complete_analysis(
            queries, include_validation, num_validation_queries, max_workers
        )

    # Legacy method aliases for backward compatibility
    def _evaluate_single_query(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Use DomainClassificationEvaluator directly for new code.
        """
        return self.service.classification_evaluator._evaluate_single_query(item)