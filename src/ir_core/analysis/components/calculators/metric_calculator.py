# src/ir_core/analysis/components/calculators/metric_calculator.py

"""
Metric Calculator Component

Handles metric calculations with parallel processing support for
comprehensive evaluation of retrieval system performance.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...metrics import RetrievalMetrics
from ...constants import (
    PARALLEL_PROCESSING_DEFAULTS,
    DEFAULT_K_VALUES
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
                pred_ids = []
                for doc in pred_docs:
                    if isinstance(doc, dict):
                        pred_ids.append(doc.get("id", ""))
                    elif isinstance(doc, str):
                        pred_ids.append(doc)
                    else:
                        pred_ids.append(str(doc) if doc else "")
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
        pred_ids = []
        for doc in pred_docs:
            if isinstance(doc, dict):
                pred_ids.append(doc.get("id", ""))
            elif isinstance(doc, str):
                pred_ids.append(doc)
            else:
                pred_ids.append(str(doc) if doc else "")
        relevant_ids = [gt_id]

        ap_score = self.metrics_calculator.average_precision(pred_ids, relevant_ids)
        ap_score = ap_score if ap_score is not None else 0.0

        return index, pred_ids, ap_score
