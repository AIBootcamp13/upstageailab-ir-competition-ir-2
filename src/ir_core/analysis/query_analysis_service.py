# src/ir_core/analysis/query_analysis_service.py

"""
Query Analysis Service - Main orchestrator for query analysis operations.

This module provides a unified interface for all query analysis functionality,
coordinating between different analysis components.
"""

from typing import Dict, List, Any, Optional
from omegaconf import DictConfig

from .query_components import QueryFeatures, BatchQueryProcessor
from .query_rewrite_evaluator import QueryRewriteEvaluator
from .domain_validation_generator import DomainValidationGenerator
from .domain_classification_evaluator import DomainClassificationEvaluator


class QueryAnalysisService:
    """
    Main service for query analysis operations.

    This service orchestrates all query analysis functionality including
    feature extraction, batch processing, rewrite evaluation, validation
    set generation, and domain classification evaluation.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the query analysis service.

        Args:
            config: Optional configuration for analysis parameters
        """
        self.config = config or DictConfig({})

        # Initialize components
        self.batch_processor = BatchQueryProcessor(config)
        self.rewrite_evaluator = QueryRewriteEvaluator(config)
        self.validation_generator = DomainValidationGenerator(config)
        self.classification_evaluator = DomainClassificationEvaluator(config)

    def analyze_query(self, query: str) -> QueryFeatures:
        """
        Analyze a single query and extract comprehensive features.

        Args:
            query: The query string to analyze

        Returns:
            QueryFeatures: Detailed analysis results
        """
        return self.batch_processor.feature_extractor.extract_features(query)

    def analyze_batch(self, queries: List[str], max_workers: Optional[int] = None) -> List[QueryFeatures]:
        """
        Analyze a batch of queries with optional parallel processing.

        Args:
            queries: List of query strings
            max_workers: Maximum number of worker threads

        Returns:
            List[QueryFeatures]: Analysis results for each query
        """
        return self.batch_processor.process_batch(queries, max_workers)

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
        return self.rewrite_evaluator.measure_rewrite_effectiveness(
            original_query, rewritten_query
        )

    def compare_queries(
        self,
        original_query: str,
        rewritten_query: str
    ) -> Dict[str, Any]:
        """
        Compare two queries and provide detailed comparison metrics.

        Args:
            original_query: The original query
            rewritten_query: The rewritten query

        Returns:
            Dict[str, Any]: Detailed comparison results
        """
        return self.rewrite_evaluator.compare_queries(
            original_query, rewritten_query
        )

    def create_domain_validation_set(
        self,
        num_queries_per_domain: int = 10,
        use_llm: bool = True,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a validation set for domain classification.

        Args:
            num_queries_per_domain: Number of queries to generate per domain
            use_llm: Whether to use LLM for query generation
            max_workers: Maximum number of worker threads

        Returns:
            List[Dict[str, Any]]: Validation set with queries and expected domains
        """
        return self.validation_generator.create_domain_validation_set(
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
            max_workers: Maximum number of worker threads

        Returns:
            Dict[str, Any]: Evaluation metrics and detailed results
        """
        return self.classification_evaluator.evaluate_domain_classification(
            validation_set, max_workers
        )

    def get_evaluation_summary(self, evaluation_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of evaluation results.

        Args:
            evaluation_result: Evaluation result from evaluate_domain_classification

        Returns:
            str: Human-readable summary
        """
        return self.classification_evaluator.get_evaluation_summary(evaluation_result)

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
        results = {
            "query_analysis": self.analyze_batch(queries, max_workers),
            "summary": self._generate_analysis_summary(queries)
        }

        if include_validation:
            validation_set = self.create_domain_validation_set(
                num_validation_queries, max_workers=max_workers
            )
            evaluation = self.evaluate_domain_classification(
                validation_set, max_workers
            )
            results["validation"] = {
                "validation_set": validation_set,
                "evaluation": evaluation,
                "summary": self.get_evaluation_summary(evaluation)
            }

        return results

    def _generate_analysis_summary(self, queries: List[str]) -> Dict[str, Any]:
        """
        Generate a summary of query analysis results.

        Args:
            queries: List of analyzed queries

        Returns:
            Dict[str, Any]: Analysis summary
        """
        if not queries:
            return {"total_queries": 0}

        features = self.analyze_batch(queries)

        # Calculate aggregate statistics
        total_length = sum(f.length for f in features)
        avg_length = total_length / len(features)

        query_types = {}
        domains = {}

        for feature in features:
            # Count query types
            qtype = feature.query_type
            query_types[qtype] = query_types.get(qtype, 0) + 1

            # Count domains
            for domain in feature.domain:
                domains[domain] = domains.get(domain, 0) + 1

        return {
            "total_queries": len(queries),
            "average_length": avg_length,
            "query_type_distribution": query_types,
            "domain_distribution": domains,
            "features_extracted": len(features)
        }