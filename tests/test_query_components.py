# tests/test_query_components.py

"""
Unit tests for query components.
"""

import pytest
from omegaconf import DictConfig

from src.ir_core.analysis.query_components import (
    QueryFeatureExtractor,
    BatchQueryProcessor,
    QueryFeatures
)


class TestQueryFeatureExtractor:
    """Test cases for QueryFeatureExtractor."""

    def test_extract_features(self):
        """Test feature extraction."""
        extractor = QueryFeatureExtractor()

        query = "What is the speed of light in vacuum?"
        features = extractor.extract_features(query)

        assert isinstance(features, QueryFeatures)
        assert features.length == len(query)
        assert features.word_count > 0
        assert isinstance(features.domain, list)
        assert isinstance(features.complexity_score, float)

    def test_extract_domain(self):
        """Test domain extraction."""
        extractor = QueryFeatureExtractor()

        physics_query = "What is Newton's law?"
        biology_query = "How do cells divide?"

        physics_features = extractor.extract_features(physics_query)
        biology_features = extractor.extract_features(biology_query)

        assert "physics" in physics_features.domain
        assert "biology" in biology_features.domain


class TestBatchQueryProcessor:
    """Test cases for BatchQueryProcessor."""

    def test_process_batch(self):
        """Test batch processing."""
        processor = BatchQueryProcessor()

        queries = ["What is gravity?", "How do magnets work?"]
        results = processor.process_batch(queries)

        assert len(results) == 2
        assert all(isinstance(r, QueryFeatures) for r in results)

    def test_process_batch_with_workers(self):
        """Test batch processing with parallel workers."""
        processor = BatchQueryProcessor()

        queries = ["Query 1", "Query 2", "Query 3"]
        results = processor.process_batch(queries, max_workers=2)

        assert len(results) == 3
        assert all(isinstance(r, QueryFeatures) for r in results)
