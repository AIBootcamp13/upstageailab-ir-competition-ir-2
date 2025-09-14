# tests/test_query_enhancement.py

"""
Unit tests for query enhancement techniques.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.ir_core.query_enhancement.rewriter import QueryRewriter
from src.ir_core.query_enhancement.step_back import StepBackPrompting
from src.ir_core.query_enhancement.decomposer import QueryDecomposer
from src.ir_core.query_enhancement.hyde import HyDE
from src.ir_core.query_enhancement.translator import QueryTranslator
from src.ir_core.query_enhancement.manager import QueryEnhancementManager


class TestQueryRewriter:
    """Test cases for QueryRewriter."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Enhanced query with synonyms and related terms"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        client.chat.completions.create.return_value = mock_response
        return client

    def test_rewrite_query_success(self, mock_openai_client):
        """Test successful query rewriting."""
        rewriter = QueryRewriter(openai_client=mock_openai_client)
        result = rewriter.rewrite_query("What is photosynthesis?")

        assert isinstance(result, str)
        assert len(result) > 0
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_rewrite_query_fallback_on_error(self, mock_openai_client):
        """Test fallback to original query on API error."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        rewriter = QueryRewriter(openai_client=mock_openai_client)
        original_query = "What is photosynthesis?"
        result = rewriter.rewrite_query(original_query)

        assert result == original_query

    def test_expand_query(self, mock_openai_client):
        """Test query expansion."""
        rewriter = QueryRewriter(openai_client=mock_openai_client)
        result = rewriter.expand_query("photosynthesis", expansion_factor=2)

        assert isinstance(result, str)
        assert len(result) > 0


class TestStepBackPrompting:
    """Test cases for StepBackPrompting."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Mock responses for different calls
        responses = [
            "The fundamental concept of energy conversion in plants",  # Abstract concept
            "plant energy conversion chlorophyll light carbon dioxide glucose"  # Search keywords
        ]
        response_iter = iter(responses)

        def mock_create(**kwargs):
            mock_message.content = next(response_iter)
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        client.chat.completions.create.side_effect = mock_create
        return client

    def test_step_back_success(self, mock_openai_client):
        """Test successful step-back prompting."""
        step_back = StepBackPrompting(openai_client=mock_openai_client)
        result = step_back.step_back("How does photosynthesis work?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_enhance_ambiguous_query(self, mock_openai_client):
        """Test enhancement of ambiguous queries."""
        step_back = StepBackPrompting(openai_client=mock_openai_client)

        # Test ambiguous query
        result = step_back.enhance_ambiguous_query("What is the meaning of life?")
        assert isinstance(result, str)

        # Test non-ambiguous query (should return original)
        clear_query = "Calculate 2 + 2"
        result = step_back.enhance_ambiguous_query(clear_query)
        assert result == clear_query

    def test_is_ambiguous_detection(self):
        """Test ambiguity detection."""
        step_back = StepBackPrompting()

        ambiguous_queries = [
            "What is the meaning of life?",
            "Why do we exist?",
            "How does evolution work?"
        ]

        clear_queries = [
            "Calculate 5 * 3",
            "What is 2 + 2?",
            "Show me the weather"
        ]

        for query in ambiguous_queries:
            assert step_back._is_ambiguous(query)

        for query in clear_queries:
            assert not step_back._is_ambiguous(query)


class TestQueryDecomposer:
    """Test cases for QueryDecomposer."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "1. What are the main components of photosynthesis?\n2. How do plants convert light energy?\n3. What role does chlorophyll play?"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        client.chat.completions.create.return_value = mock_response
        return client

    def test_decompose_query_success(self, mock_openai_client):
        """Test successful query decomposition."""
        decomposer = QueryDecomposer(openai_client=mock_openai_client)
        sub_queries = decomposer.decompose_query("Explain photosynthesis in detail")

        assert isinstance(sub_queries, list)
        assert len(sub_queries) > 1
        assert all(isinstance(q, str) for q in sub_queries)

    def test_should_decompose_complex_query(self):
        """Test detection of queries that should be decomposed."""
        decomposer = QueryDecomposer()

        complex_queries = [
            "Compare photosynthesis and cellular respiration",
            "Explain the water cycle and its importance",
            "Describe the differences between mitosis and meiosis"
        ]

        simple_queries = [
            "What is water?",
            "Define photosynthesis",
            "Calculate area"
        ]

        for query in complex_queries:
            assert decomposer.should_decompose(query)

        for query in simple_queries:
            assert not decomposer.should_decompose(query)


class TestHyDE:
    """Test cases for HyDE (Hypothetical Document Embeddings)."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Photosynthesis is the process by which plants convert light energy into chemical energy..."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        client.chat.completions.create.return_value = mock_response
        return client

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model."""
        model = Mock()
        model.encode.return_value = np.random.rand(768).astype(np.float32)
        return model

    @patch('src.ir_core.query_enhancement.hyde.encode_query')
    @patch('src.ir_core.query_enhancement.hyde.dense_retrieve')
    def test_retrieve_with_hyde(self, mock_dense_retrieve, mock_encode_query, mock_openai_client):
        """Test HyDE retrieval."""
        # Setup mocks
        mock_encode_query.return_value = np.random.rand(768).astype(np.float32)
        mock_dense_retrieve.return_value = [
            {"_source": {"content": "Test content", "docid": "1", "title": "Test"}, "_score": 0.8}
        ]

        hyde = HyDE(openai_client=mock_openai_client)
        results = hyde.retrieve_with_hyde("What is photosynthesis?", top_k=5)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]
        assert "score" in results[0]

    def test_should_use_hyde(self):
        """Test HyDE usage detection."""
        hyde = HyDE()

        # Should use HyDE for short queries
        assert hyde.should_use_hyde("What is water?")
        assert hyde.should_use_hyde("How does it work?")

        # Should not use HyDE for long queries
        long_query = "Explain in detail the process of photosynthesis including all the chemical reactions and biological mechanisms involved"
        assert not hyde.should_use_hyde(long_query)


class TestQueryTranslator:
    """Test cases for QueryTranslator."""

    def test_translate_without_googletrans(self):
        """Test translator behavior when googletrans is not available."""
        translator = QueryTranslator()

        # Should return original query when translator not available
        result = translator.translate_to_english("테스트 쿼리")
        assert result == "테스트 쿼리"

    def test_detect_language_without_googletrans(self):
        """Test language detection fallback."""
        translator = QueryTranslator()

        result = translator.detect_language("Hello world")
        assert result == "unknown"

    def test_should_translate_without_googletrans(self):
        """Test translation detection fallback."""
        translator = QueryTranslator()

        # Should not attempt translation when translator not available
        assert not translator.should_translate("테스트 쿼리")


class TestQueryEnhancementManager:
    """Test cases for QueryEnhancementManager."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'enabled': True,
            'openai_model': 'gpt-3.5-turbo',
            'max_tokens': 500,
            'temperature': 0.3,
            'techniques': {
                'rewriting': {'enabled': True, 'priority': 1},
                'step_back': {'enabled': True, 'priority': 2},
                'decomposition': {'enabled': True, 'priority': 3},
                'hyde': {'enabled': True, 'priority': 4},
                'translation': {'enabled': True, 'priority': 5}
            }
        }

    @patch('src.ir_core.query_enhancement.manager.QueryRewriter')
    def test_enhance_query_rewriting(self, mock_rewriter_class, mock_config):
        """Test query enhancement with rewriting technique."""
        # Setup mock rewriter
        mock_rewriter = Mock()
        mock_rewriter.rewrite_query.return_value = "Enhanced query"
        mock_rewriter_class.return_value = mock_rewriter

        manager = QueryEnhancementManager(config=mock_config)
        result = manager.enhance_query("Test query", technique="rewriting")

        assert result['enhanced'] is True
        assert result['enhanced_query'] == "Enhanced query"
        assert result['technique_used'] == "rewriting"

    def test_auto_select_technique(self, mock_config):
        """Test automatic technique selection."""
        manager = QueryEnhancementManager(config=mock_config)

        # Test with short query (should prefer HyDE)
        short_query = "What is water?"
        analysis = manager._analyze_query(short_query)
        selected = manager._select_technique(analysis)

        # Should select a technique (likely hyde for short queries)
        assert selected is not None
        assert selected in ['rewriting', 'step_back', 'decomposition', 'hyde', 'translation']

    def test_get_available_techniques(self, mock_config):
        """Test getting available techniques."""
        manager = QueryEnhancementManager(config=mock_config)
        techniques = manager.get_available_techniques()

        assert isinstance(techniques, list)
        assert len(techniques) > 0
        assert all(isinstance(t, str) for t in techniques)

    def test_get_technique_info(self, mock_config):
        """Test getting technique information."""
        manager = QueryEnhancementManager(config=mock_config)
        info = manager.get_technique_info('rewriting')

        assert isinstance(info, dict)
        assert 'description' in info
        assert 'best_for' in info
        assert 'priority' in info

    def test_disabled_enhancement(self):
        """Test behavior when enhancement is disabled."""
        config = {'enabled': False}
        manager = QueryEnhancementManager(config=config)

        result = manager.enhance_query("Test query")
        assert result['enhanced'] is False
        assert result['technique_used'] == 'none'