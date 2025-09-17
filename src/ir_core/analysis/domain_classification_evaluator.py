# src/ir_core/analysis/domain_classification_evaluator.py

"""
Domain classification evaluation functionality.

This module provides functionality to evaluate domain classification
accuracy against validation sets.
"""

from typing import Dict, List, Any, Optional
from omegaconf import DictConfig

from .query_components import QueryFeatures, QueryFeatureExtractor
from .parallel_processor import ParallelProcessor
from .constants import PARALLEL_PROCESSING_DEFAULTS


class DomainClassificationEvaluator:
    """
    Evaluates domain classification accuracy against validation sets.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the domain classification evaluator.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})
        self.feature_extractor = QueryFeatureExtractor()
        self.parallel_processor = ParallelProcessor(
            max_workers=self.config.get('analysis', {}).get('max_workers', None),
            enable_parallel=self.config.get('analysis', {}).get('enable_parallel', True)
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
        if not validation_set:
            return self._get_empty_evaluation_result()

        print(f"🔍 Evaluating domain classification for {len(validation_set)} queries...")

        # Use parallel processing for larger validation sets
        if len(validation_set) > 5 and max_workers != 0:
            if max_workers is None:
                max_workers = PARALLEL_PROCESSING_DEFAULTS["max_workers_analysis"]

            error_result = {
                "query": "",
                "expected": ["unknown"],
                "predicted": ["unknown"],
                "correct": [],
                "false_positives": [],
                "false_negatives": ["unknown"],
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "exact_match": False
            }

            results = self.parallel_processor.process_batch_with_error_handling(
                validation_set,
                self._evaluate_single_query,
                error_result,
                batch_threshold=5,
                operation_name="validation queries"
            )
        else:
            # Sequential processing for small sets
            results = [self._evaluate_single_query(item) for item in validation_set]

        # Aggregate metrics
        return self._aggregate_evaluation_metrics(results, validation_set)

    def _evaluate_single_query(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query for domain classification.

        Args:
            item: Query item with expected domains

        Returns:
            Dict[str, Any]: Evaluation result for this query
        """
        query = item["query"]
        expected_domains = set(item["expected_domain"])

        # Get predicted domains
        features = self.feature_extractor.extract_features(query)
        predicted_domains = set(features.domain)

        # Calculate metrics
        correct_predictions = expected_domains.intersection(predicted_domains)
        false_positives = predicted_domains - expected_domains
        false_negatives = expected_domains - predicted_domains

        precision = len(correct_predictions) / len(predicted_domains) if predicted_domains else 0
        recall = len(correct_predictions) / len(expected_domains) if expected_domains else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "query": query,
            "expected": list(expected_domains),
            "predicted": list(predicted_domains),
            "correct": list(correct_predictions),
            "false_positives": list(false_positives),
            "false_negatives": list(false_negatives),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": expected_domains == predicted_domains
        }

    def _aggregate_evaluation_metrics(
        self,
        results: List[Dict[str, Any]],
        validation_set: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate evaluation metrics from individual query results.

        Args:
            results: List of individual evaluation results
            validation_set: Original validation set

        Returns:
            Dict[str, Any]: Aggregated evaluation metrics
        """
        total_queries = len(results)
        if total_queries == 0:
            return self._get_empty_evaluation_result()

        exact_matches = sum(1 for r in results if r["exact_match"])
        avg_precision = sum(r["precision"] for r in results) / total_queries
        avg_recall = sum(r["recall"] for r in results) / total_queries
        avg_f1 = sum(r["f1"] for r in results) / total_queries

        # Per-domain metrics
        domain_metrics = self._calculate_domain_metrics(results, validation_set)

        return {
            "total_queries": total_queries,
            "exact_match_accuracy": exact_matches / total_queries,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "domain_metrics": domain_metrics,
            "detailed_results": results
        }

    def _calculate_domain_metrics(
        self,
        results: List[Dict[str, Any]],
        validation_set: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate per-domain accuracy metrics.

        Args:
            results: List of individual evaluation results
            validation_set: Original validation set

        Returns:
            Dict[str, Any]: Per-domain metrics
        """
        domain_metrics = {}

        for item in validation_set:
            for domain in item["expected_domain"]:
                if domain not in domain_metrics:
                    domain_metrics[domain] = {"total": 0, "correct": 0}

                domain_metrics[domain]["total"] += 1

                # Check if this domain was correctly predicted for this query
                query_result = next(
                    (r for r in results if r["query"] == item["query"]),
                    None
                )
                if query_result and domain in query_result["predicted"]:
                    domain_metrics[domain]["correct"] += 1

        # Calculate accuracy for each domain
        for domain in domain_metrics:
            total = domain_metrics[domain]["total"]
            correct = domain_metrics[domain]["correct"]
            domain_metrics[domain]["accuracy"] = correct / total if total > 0 else 0

        return domain_metrics

    def _get_empty_evaluation_result(self) -> Dict[str, Any]:
        """
        Get empty evaluation result structure.

        Returns:
            Dict[str, Any]: Empty evaluation result
        """
        return {
            "total_queries": 0,
            "exact_match_accuracy": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "domain_metrics": {},
            "detailed_results": []
        }

    def get_evaluation_summary(self, evaluation_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of evaluation results.

        Args:
            evaluation_result: Evaluation result from evaluate_domain_classification

        Returns:
            str: Human-readable summary
        """
        total = evaluation_result["total_queries"]
        accuracy = evaluation_result["exact_match_accuracy"]
        precision = evaluation_result["avg_precision"]
        recall = evaluation_result["avg_recall"]
        f1 = evaluation_result["avg_f1"]

        summary = f"""
Domain Classification Evaluation Summary:
========================================
Total Queries: {total}
Exact Match Accuracy: {accuracy:.3f}
Average Precision: {precision:.3f}
Average Recall: {recall:.3f}
Average F1-Score: {f1:.3f}

Per-Domain Performance:
"""

        for domain, metrics in evaluation_result["domain_metrics"].items():
            accuracy = metrics["accuracy"]
            summary += f"- {domain}: {accuracy:.3f} ({metrics['correct']}/{metrics['total']})\n"

        return summary