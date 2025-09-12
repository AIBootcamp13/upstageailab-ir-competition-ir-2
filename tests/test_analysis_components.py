# tests/test_analysis_components.py

"""
Unit tests for analysis components.
"""

import pytest
from unittest.mock import Mock, patch
from omegaconf import DictConfig

from src.ir_core.analysis.components import (
    MetricCalculator,
    QueryBatchProcessor,
    ErrorAnalyzer,
    ResultAggregator,
    MetricCalculationResult,
    QueryProcessingResult,
    ErrorAnalysisResult
)


class TestMetricCalculator:
    """Test cases for MetricCalculator."""

    def test_init(self):
        """Test initialization."""
        config = DictConfig({"analysis": {"max_workers": 4, "enable_parallel": True}})
        calculator = MetricCalculator(config)
        assert calculator.max_workers == 4
        assert calculator.enable_parallel is True

    @patch('src.ir_core.analysis.components.calculators.metric_calculator.RetrievalMetrics')
    def test_calculate_batch_metrics(self, mock_metrics):
        """Test batch metrics calculation."""
        # Mock the metrics calculator
        mock_instance = Mock()
        mock_instance.average_precision.return_value = 0.8
        mock_metrics.return_value = mock_instance

        config = DictConfig({})
        calculator = MetricCalculator(config)

        predicted_docs = [[{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.7}]]
        ground_truth_ids = ["doc1"]

        result = calculator.calculate_batch_metrics(predicted_docs, ground_truth_ids)

        assert isinstance(result, MetricCalculationResult)
        assert result.map_score >= 0.0
        assert isinstance(result.precision_at_k, dict)


class TestQueryBatchProcessor:
    """Test cases for QueryBatchProcessor."""

    @patch('src.ir_core.analysis.components.processors.query_processor.QueryAnalyzer')
    def test_process_batch(self, mock_query_analyzer):
        """Test batch query processing."""
        mock_instance = Mock()
        mock_instance.analyze_batch.return_value = [
            Mock(length=10, domain=["physics"], complexity_score=0.5)
        ]
        mock_query_analyzer.return_value = mock_instance

        config = DictConfig({})
        processor = QueryBatchProcessor(config)

        queries = [{"msg": [{"content": "What is gravity?"}]}]
        result = processor.process_batch(queries)

        assert isinstance(result, QueryProcessingResult)
        assert len(result.query_features_list) == 1


class TestErrorAnalyzer:
    """Test cases for ErrorAnalyzer."""

    def test_analyze_errors(self):
        """Test comprehensive error analysis."""
        analyzer = ErrorAnalyzer()

        predicted_docs = [[{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.7}]]
        ground_truth_ids = ["doc1"]
        queries = ["What is physics?"]
        query_domains = [["physics"]]

        result = analyzer.analyze_errors(predicted_docs, ground_truth_ids, queries, query_domains)

        assert isinstance(result, ErrorAnalysisResult)
        assert "successful" in result.error_categories
        assert result.retrieval_success_rate == 1.0  # Should be successful

        # Check new comprehensive fields
        assert isinstance(result.query_understanding_failures, dict)
        assert isinstance(result.retrieval_failures, dict)
        assert isinstance(result.system_failures, dict)
        assert isinstance(result.error_patterns, dict)
        assert isinstance(result.domain_error_rates, dict)
        assert isinstance(result.temporal_trends, dict)
        assert isinstance(result.error_recommendations, list)

    def test_error_categorization(self):
        """Test different error categorization scenarios."""
        analyzer = ErrorAnalyzer()

        # Test ambiguous query
        predicted_docs = [[{"id": "doc1", "score": 0.5}]]
        ground_truth_ids = ["doc_missing"]
        queries = ["What is this thing?"]  # Ambiguous
        query_domains = [["unknown"]]

        result = analyzer.analyze_errors(predicted_docs, ground_truth_ids, queries, query_domains)

        # Should detect out-of-domain error
        assert result.query_understanding_failures.get("out_of_domain", 0) == 1

        # Test false positive
        predicted_docs = [[{"id": "wrong_doc", "score": 0.95}]]
        ground_truth_ids = ["correct_doc"]
        queries = ["Physics question"]
        query_domains = [["physics"]]

        result = analyzer.analyze_errors(predicted_docs, ground_truth_ids, queries, query_domains)

        # Should detect false positive
        assert result.retrieval_failures.get("false_positive", 0) == 1


class TestResultAggregator:
    """Test cases for ResultAggregator."""

    def test_generate_recommendations(self):
        """Test recommendations generation."""
        recommendations = ResultAggregator.generate_recommendations(
            map_score=0.6,
            retrieval_success_rate=0.8,
            rewrite_rate=0.2
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_generate_recommendations_low_map(self):
        """Test recommendations for low MAP score."""
        recommendations = ResultAggregator.generate_recommendations(
            map_score=0.3,
            retrieval_success_rate=0.8,
            rewrite_rate=0.2
        )

        assert any("MAP score" in rec for rec in recommendations)
