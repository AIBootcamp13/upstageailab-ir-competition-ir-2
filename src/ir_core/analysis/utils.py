# src/ir_core/analysis/utils.py

"""
Utility functions for the Scientific QA retrieval analysis module.

This module contains common utility functions used across different analysis components.
"""

from typing import Dict, List, Optional


def find_rank_of_ground_truth(pred_ids: List[str], gt_id: str) -> Optional[int]:
    """
    Find the rank of ground truth document in predicted results.

    Args:
        pred_ids: List of predicted document IDs
        gt_id: Ground truth document ID

    Returns:
        Optional[int]: Rank of ground truth (1-based), None if not found
    """
    try:
        return pred_ids.index(gt_id) + 1
    except ValueError:
        return None


def calculate_top_k_precision(pred_ids: List[str], relevant_ids: List[str], k_values: List[int]) -> Dict[int, float]:
    """
    Calculate precision@K for different K values.

    Args:
        pred_ids: List of predicted document IDs
        relevant_ids: List of relevant document IDs
        k_values: List of K values to calculate precision for

    Returns:
        Dict[int, float]: Precision values for each K
    """
    precision_at_k = {}
    relevant_set = set(relevant_ids)

    for k in k_values:
        if len(pred_ids) >= k:
            top_k_preds = pred_ids[:k]
            correct = len([pid for pid in top_k_preds if pid in relevant_set])
            precision_at_k[k] = correct / k
        else:
            precision_at_k[k] = 0.0

    return precision_at_k
