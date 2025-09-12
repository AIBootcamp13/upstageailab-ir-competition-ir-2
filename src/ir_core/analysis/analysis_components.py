# src/ir_core/analysis/analysis_components.py

"""
Refactored analysis components for better modularity and maintainability.

This module contains smaller, focused classes extracted from the monolithic
RetrievalAnalyzer to improve code organization and reusability.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

from .metrics import RetrievalMetrics
from .query_analyzer import QueryAnalyzer
from .constants import (
    ANALYSIS_THRESHOLDS,
    PARALLEL_PROCESSING_DEFAULTS,
    DEFAULT_K_VALUES,
    ERROR_ANALYSIS_DOMAIN_CHECKS
)


@dataclass
class MetricCalculationResult:
    """Result of metric calculations for a batch."""
    map_score: float
    mean_ap: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ap_scores: List[float]


class MetricCalculator:
    """
    Handles metric calculations with parallel processing support.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the metric calculator.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})
        self.metrics_calculator = RetrievalMetrics()
        self.max_workers = self.config.get('analysis', {}).get('max_workers', None)
        self.enable_parallel = self.config.get('analysis', {}).get('enable_parallel', True)

    def calculate_batch_metrics(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str],
        max_workers: Optional[int] = None
    ) -> MetricCalculationResult:
        """
        Calculate comprehensive metrics for a batch of predictions.

        Args:
            predicted_docs_list: List of predicted documents for each query
            ground_truth_ids: List of ground truth document IDs
            max_workers: Maximum workers for parallel processing

        Returns:
            MetricCalculationResult: Calculated metrics
        """
        # Calculate basic metrics with optional parallel processing
        all_results_for_map = []
        ap_scores = []

        if len(predicted_docs_list) > PARALLEL_PROCESSING_DEFAULTS["batch_size_threshold"] and self.enable_parallel and max_workers != 0:
            # Use parallel processing
            if max_workers is None:
                max_workers = self.max_workers or min(PARALLEL_PROCESSING_DEFAULTS["max_workers_analysis"], len(predicted_docs_list))

            print(f"🔄 Calculating metrics for {len(predicted_docs_list)} queries using {max_workers} parallel workers...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {}
                for i, (pred_docs, gt_id) in enumerate(zip(predicted_docs_list, ground_truth_ids)):
                    future = executor.submit(self._calculate_single_query_metrics, pred_docs, gt_id, i)
                    future_to_index[future] = i

                results_by_index = {}
                for future in as_completed(future_to_index):
                    try:
                        index, pred_ids, ap_score = future.result()
                        results_by_index[index] = (pred_ids, ap_score)
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"Error calculating metrics for query {index}: {e}")
                        results_by_index[index] = ([], 0.0)

                for i in range(len(predicted_docs_list)):
                    if i in results_by_index:
                        pred_ids, ap_score = results_by_index[i]
                        all_results_for_map.append((pred_ids, [ground_truth_ids[i]]))
                        ap_scores.append(ap_score)
        else:
            # Sequential processing
            for i, (pred_docs, gt_id) in enumerate(zip(predicted_docs_list, ground_truth_ids)):
                pred_ids = [doc.get("id", "") for doc in pred_docs]
                relevant_ids = [gt_id]
                all_results_for_map.append((pred_ids, relevant_ids))

                ap_score = self.metrics_calculator.average_precision(pred_ids, relevant_ids)
                ap_scores.append(ap_score if ap_score is not None else 0.0)

        # Calculate overall metrics
        map_score = self.metrics_calculator.mean_average_precision(all_results_for_map)
        mean_ap = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

        precision_at_k = {}
        for k in DEFAULT_K_VALUES:
            precision_at_k[k] = self.metrics_calculator.precision_at_k(all_results_for_map, k)

        recall_at_k = {}
        for k in DEFAULT_K_VALUES:
            recall_at_k[k] = self.metrics_calculator.recall_at_k(all_results_for_map, k)

        return MetricCalculationResult(
            map_score=map_score,
            mean_ap=mean_ap,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ap_scores=ap_scores
        )

    def _calculate_single_query_metrics(
        self,
        pred_docs: List[Dict[str, Any]],
        gt_id: str,
        index: int
    ) -> Tuple[int, List[str], float]:
        """
        Calculate metrics for a single query (for parallel processing).

        Args:
            pred_docs: Predicted documents
            gt_id: Ground truth ID
            index: Query index

        Returns:
            Tuple of (index, predicted_ids, ap_score)
        """
        pred_ids = [doc.get("id", "") for doc in pred_docs]
        relevant_ids = [gt_id]

        ap_score = self.metrics_calculator.average_precision(pred_ids, relevant_ids)
        ap_score = ap_score if ap_score is not None else 0.0

        return index, pred_ids, ap_score


@dataclass
class QueryProcessingResult:
    """Result of query processing."""
    original_queries: List[str]
    rewritten_queries: List[str]
    query_features_list: List[Any]  # QueryFeatures
    avg_query_length: float
    avg_query_complexity: float
    rewrite_rate: float
    domain_distribution: Dict[str, int]


class QueryBatchProcessor:
    """
    Handles batch processing of queries with analysis.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the query batch processor.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})
        self.query_analyzer = QueryAnalyzer(config)

    def process_batch(
        self,
        queries: List[Dict[str, Any]],
        rewritten_queries: Optional[List[str]] = None,
        max_workers: Optional[int] = None
    ) -> QueryProcessingResult:
        """
        Process a batch of queries and extract features.

        Args:
            queries: List of query dictionaries
            rewritten_queries: Optional rewritten queries
            max_workers: Maximum workers for parallel processing

        Returns:
            QueryProcessingResult: Processed query data
        """
        original_queries = [q.get("msg", [{}])[0].get("content", "") for q in queries]

        if rewritten_queries is None:
            rewritten_queries = original_queries

        # Analyze queries
        query_features_list = self.query_analyzer.analyze_batch(original_queries, max_workers)

        total_queries = len(original_queries)

        # Calculate aggregate statistics
        avg_query_length = sum(f.length for f in query_features_list) / total_queries if total_queries > 0 else 0
        avg_query_complexity = sum(f.complexity_score for f in query_features_list) / total_queries if total_queries > 0 else 0

        # Rewrite analysis
        rewrite_changes = sum(1 for orig, rew in zip(original_queries, rewritten_queries) if orig != rew)
        rewrite_rate = rewrite_changes / total_queries if total_queries > 0 else 0

        # Domain distribution
        domain_distribution = {}
        for features in query_features_list:
            for domain in features.domain:
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

        return QueryProcessingResult(
            original_queries=original_queries,
            rewritten_queries=rewritten_queries,
            query_features_list=query_features_list,
            avg_query_length=avg_query_length,
            avg_query_complexity=avg_query_complexity,
            rewrite_rate=rewrite_rate,
            domain_distribution=domain_distribution
        )


@dataclass
class ErrorAnalysisResult:
    """Result of error analysis."""
    retrieval_success_rate: float
    error_categories: Dict[str, int]


class ErrorAnalyzer:
    """
    Analyzes retrieval errors and categorizes them.
    """

    def analyze_errors(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str],
        original_queries: List[str]
    ) -> ErrorAnalysisResult:
        """
        Analyze retrieval errors and categorize them.

        Args:
            predicted_docs_list: Predicted documents for each query
            ground_truth_ids: Ground truth IDs
            original_queries: Original query strings

        Returns:
            ErrorAnalysisResult: Error analysis results
        """
        error_categories = {
            "no_retrieval": 0,
            "wrong_domain": 0,
            "low_relevance": 0,
            "query_mismatch": 0,
            "successful": 0
        }

        success_count = 0
        for pred_docs, gt_id, query in zip(predicted_docs_list, ground_truth_ids, original_queries):
            if not pred_docs:
                error_categories["no_retrieval"] += 1
                continue

            top_10_ids = [doc.get("id", "") for doc in pred_docs[:10]]
            if gt_id in top_10_ids:
                error_categories["successful"] += 1
                success_count += 1
                continue

            # Analyze failure reasons
            query_lower = query.lower()

            # Domain mismatch check
            retrieved_domains = set()
            for doc in pred_docs[:5]:
                doc_content = doc.get("content", "").lower()[:200]
                for domain, terms in ERROR_ANALYSIS_DOMAIN_CHECKS.items():
                    if any(term in doc_content for term in terms):
                        retrieved_domains.add(domain)

            query_domain = "unknown"
            for domain, terms in ERROR_ANALYSIS_DOMAIN_CHECKS.items():
                if any(term in query_lower for term in terms):
                    query_domain = domain
                    break

            if query_domain != "unknown" and retrieved_domains and query_domain not in retrieved_domains:
                error_categories["wrong_domain"] += 1
            else:
                top_scores = [doc.get("score", 0.0) for doc in pred_docs[:3]]
                avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
                if avg_top_score < ANALYSIS_THRESHOLDS["low_relevance_threshold"]:
                    error_categories["low_relevance"] += 1
                else:
                    error_categories["query_mismatch"] += 1

        retrieval_success_rate = success_count / len(predicted_docs_list) if predicted_docs_list else 0

        # Remove zero categories
        error_categories = {k: v for k, v in error_categories.items() if v > 0}

        return ErrorAnalysisResult(
            retrieval_success_rate=retrieval_success_rate,
            error_categories=error_categories
        )


class ResultAggregator:
    """
    Aggregates results from different analysis components and generates recommendations.
    """

    @staticmethod
    def generate_recommendations(
        map_score: float,
        retrieval_success_rate: float,
        rewrite_rate: float
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis results.

        Args:
            map_score: Mean Average Precision
            retrieval_success_rate: Success rate
            rewrite_rate: Rewrite rate

        Returns:
            List[str]: Recommendations
        """
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
