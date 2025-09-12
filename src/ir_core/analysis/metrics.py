# src/ir_core/analysis/metrics.py

"""
Comprehensive metrics calculation for Scientific QA retrieval evaluation.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from .constants import DEFAULT_K_VALUES


class RetrievalMetrics:
    """
    Collection of retrieval evaluation metrics for information retrieval tasks.
    """

    @staticmethod
    def average_precision(predicted_ids: List[str], relevant_ids: List[str]) -> Optional[float]:
        """
        Calculate Average Precision (AP) for a single query.

        Args:
            predicted_ids: List of predicted document IDs in ranked order
            relevant_ids: List of ground truth relevant document IDs

        Returns:
            float or None: Average Precision score, or None if no relevant documents
        """
        if not relevant_ids:
            return None

        relevant_set = set(relevant_ids)
        num_relevant = len(relevant_set)
        num_predicted = len(predicted_ids)

        if num_relevant == 0:
            return 0.0

        # Calculate precision at each relevant document position
        precision_sum = 0.0
        num_correct = 0

        for i, doc_id in enumerate(predicted_ids):
            if doc_id in relevant_set:
                num_correct += 1
                precision_at_i = num_correct / (i + 1)
                precision_sum += precision_at_i

        if num_correct == 0:
            return 0.0

        return precision_sum / num_relevant

    @staticmethod
    def mean_average_precision(results: List[Tuple[List[str], List[str]]]) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple queries.

        Args:
            results: List of (predicted_ids, relevant_ids) tuples

        Returns:
            float: Mean Average Precision score
        """
        if not results:
            return 0.0

        ap_scores = []
        for predicted_ids, relevant_ids in results:
            ap = RetrievalMetrics.average_precision(predicted_ids, relevant_ids)
            if ap is not None:
                ap_scores.append(ap)

        return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

    @staticmethod
    def precision_at_k(results: List[Tuple[List[str], List[str]]], k: int) -> float:
        """
        Calculate Precision@K across multiple queries.

        Args:
            results: List of (predicted_ids, relevant_ids) tuples
            k: Number of top results to consider

        Returns:
            float: Average Precision@K score
        """
        if not results:
            return 0.0

        precision_scores = []
        for predicted_ids, relevant_ids in results:
            pred_at_k = predicted_ids[:k]
            relevant_set = set(relevant_ids)

            if pred_at_k:
                correct_at_k = sum(1 for doc_id in pred_at_k if doc_id in relevant_set)
                precision_at_k = correct_at_k / len(pred_at_k)
                precision_scores.append(precision_at_k)
            else:
                precision_scores.append(0.0)

        return sum(precision_scores) / len(precision_scores)

    @staticmethod
    def recall_at_k(results: List[Tuple[List[str], List[str]]], k: int) -> float:
        """
        Calculate Recall@K across multiple queries.

        Args:
            results: List of (predicted_ids, relevant_ids) tuples
            k: Number of top results to consider

        Returns:
            float: Average Recall@K score
        """
        if not results:
            return 0.0

        recall_scores = []
        for predicted_ids, relevant_ids in results:
            pred_at_k = predicted_ids[:k]
            relevant_set = set(relevant_ids)

            if relevant_set:
                correct_at_k = sum(1 for doc_id in pred_at_k if doc_id in relevant_set)
                recall_at_k = correct_at_k / len(relevant_set)
                recall_scores.append(recall_at_k)
            else:
                recall_scores.append(0.0)

        return sum(recall_scores) / len(recall_scores)

    @staticmethod
    def ndcg_at_k(results: List[Tuple[List[str], List[str]]], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Args:
            results: List of (predicted_ids, relevant_ids) tuples
            k: Number of top results to consider

        Returns:
            float: Average NDCG@K score
        """
        if not results:
            return 0.0

        ndcg_scores = []
        for predicted_ids, relevant_ids in results:
            pred_at_k = predicted_ids[:k]
            relevant_set = set(relevant_ids)

            # Calculate DCG
            dcg = 0.0
            for i, doc_id in enumerate(pred_at_k):
                if doc_id in relevant_set:
                    dcg += 1.0 / np.log2(i + 2)  # i + 2 because positions start from 1

            # Calculate IDCG (ideal DCG)
            num_relevant = min(len(relevant_set), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0

            ndcg_scores.append(ndcg)

        return sum(ndcg_scores) / len(ndcg_scores)

    @staticmethod
    def retrieval_success_rate(results: List[Tuple[List[str], List[str]]], k: int = 10) -> float:
        """
        Calculate the rate at which ground truth documents appear in top-K results.

        Args:
            results: List of (predicted_ids, relevant_ids) tuples
            k: Number of top results to consider

        Returns:
            float: Retrieval success rate (0.0 to 1.0)
        """
        if not results:
            return 0.0

        success_count = 0
        for predicted_ids, relevant_ids in results:
            pred_at_k = set(predicted_ids[:k])
            relevant_set = set(relevant_ids)

            if pred_at_k & relevant_set:  # Intersection is non-empty
                success_count += 1

        return success_count / len(results)

    @staticmethod
    def mean_reciprocal_rank(results: List[Tuple[List[str], List[str]]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            results: List of (predicted_ids, relevant_ids) tuples

        Returns:
            float: Mean Reciprocal Rank score
        """
        if not results:
            return 0.0

        rr_scores = []
        for predicted_ids, relevant_ids in results:
            relevant_set = set(relevant_ids)

            rr = 0.0
            for i, doc_id in enumerate(predicted_ids):
                if doc_id in relevant_set:
                    rr = 1.0 / (i + 1)  # First relevant document position
                    break

            rr_scores.append(rr)

        return sum(rr_scores) / len(rr_scores)

    @staticmethod
    def query_level_metrics(predicted_ids: List[str], relevant_ids: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a single query.

        Args:
            predicted_ids: List of predicted document IDs in ranked order
            relevant_ids: List of ground truth relevant document IDs

        Returns:
            Dict[str, float]: Dictionary of metric names to values
        """
        metrics = {}

        # Average Precision
        ap = RetrievalMetrics.average_precision(predicted_ids, relevant_ids)
        metrics['AP'] = ap if ap is not None else 0.0

        # Precision@K
        for k in DEFAULT_K_VALUES:
            p_at_k = RetrievalMetrics.precision_at_k([(predicted_ids, relevant_ids)], k)
            metrics[f'P@{k}'] = p_at_k

        # Recall@K
        for k in DEFAULT_K_VALUES:
            r_at_k = RetrievalMetrics.recall_at_k([(predicted_ids, relevant_ids)], k)
            metrics[f'R@{k}'] = r_at_k

        # Reciprocal Rank
        rr = 0.0
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(predicted_ids):
            if doc_id in relevant_set:
                rr = 1.0 / (i + 1)
                break
        metrics['RR'] = rr

        # Rank of first relevant document
        rank = None
        for i, doc_id in enumerate(predicted_ids):
            if doc_id in relevant_set:
                rank = i + 1
                break
        metrics['first_relevant_rank'] = rank if rank is not None else 0

        return metrics
