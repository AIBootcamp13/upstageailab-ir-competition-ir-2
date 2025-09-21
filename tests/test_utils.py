# tests/test_utils.py

"""
Unit tests for utility functions.
"""

import pytest

from ir_core.analysis.utils import (
    find_rank_of_ground_truth,
    calculate_top_k_precision
)


class TestUtils:
    """Test cases for utility functions."""

    def test_find_rank_of_ground_truth_found(self):
        """Test finding rank when ground truth is present."""
        pred_ids = ["doc1", "doc2", "doc3"]
        gt_id = "doc2"

        rank = find_rank_of_ground_truth(pred_ids, gt_id)
        assert rank == 2

    def test_find_rank_of_ground_truth_not_found(self):
        """Test finding rank when ground truth is not present."""
        pred_ids = ["doc1", "doc2", "doc3"]
        gt_id = "doc4"

        rank = find_rank_of_ground_truth(pred_ids, gt_id)
        assert rank is None

    def test_calculate_top_k_precision(self):
        """Test precision@K calculation."""
        pred_ids = ["doc1", "doc2", "doc3", "doc4"]
        relevant_ids = ["doc1", "doc3"]
        k_values = [1, 3]

        precision = calculate_top_k_precision(pred_ids, relevant_ids, k_values)

        assert precision[1] == 1.0  # doc1 is relevant
        assert precision[3] == 2.0 / 3.0  # doc1 and doc3 are relevant out of top 3

    def test_calculate_top_k_precision_insufficient_results(self):
        """Test precision@K when fewer results than K."""
        pred_ids = ["doc1"]
        relevant_ids = ["doc1"]
        k_values = [1, 3]

        precision = calculate_top_k_precision(pred_ids, relevant_ids, k_values)

        assert precision[1] == 1.0
        assert precision[3] == 0.0  # Not enough results
