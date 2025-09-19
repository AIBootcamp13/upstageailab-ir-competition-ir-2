# tests/integration/test_enhanced_pipeline.py

"""
Integration tests for the enhanced RAG pipeline with query enhancement techniques.
"""

import pytest
import os
import json
from unittest.mock import patch, Mock

from ir_core.orchestration.pipeline import RAGPipeline
from ir_core.generation.openai import OpenAIGenerator
from ir_core.query_enhancement.manager import QueryEnhancementManager

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


class TestEnhancedPipelineIntegration:
    """Integration tests for enhanced pipeline functionality."""

    @pytest.fixture
    def mock_generator(self):
        """Mock generator for testing."""
        generator = Mock()
        generator.generate.return_value = "This is a mock generated answer based on the provided context."
        return generator

    @pytest.fixture
    def enhanced_pipeline(self, mock_generator):
        """Create enhanced pipeline with mocked components."""
        pipeline = RAGPipeline(
            generator=mock_generator,
            tool_prompt_description="Search for scientific information",
            tool_calling_model="gpt-3.5-turbo",
            use_query_enhancement=True
        )
        return pipeline

    @patch('src.ir_core.query_enhancement.manager.QueryEnhancementManager.enhance_query')
    @patch('src.ir_core.orchestration.pipeline.default_dispatcher.execute_tool')
    def test_enhanced_pipeline_with_query_enhancement(self, mock_execute_tool, mock_enhance_query, mock_generator, enhanced_pipeline):
        """Test enhanced pipeline with query enhancement enabled."""
        # Setup mocks
        mock_enhance_query.return_value = {
            'enhanced': True,
            'original_query': 'What is photosynthesis?',
            'enhanced_query': 'photosynthesis process plant energy conversion light',
            'technique_used': 'rewriting',
            'confidence': 0.8
        }

        mock_execute_tool.return_value = [
            {
                'id': 'doc1',
                'content': 'Photosynthesis is the process by which plants convert light energy into chemical energy.',
                'score': 0.9
            }
        ]

        # Test the pipeline
        result = enhanced_pipeline.run("What is photosynthesis?")

        # Verify query enhancement was called
        mock_enhance_query.assert_called_once_with('What is photosynthesis?')

        # Verify tool execution was called with enhanced query
        mock_execute_tool.assert_called_once()

        # Verify generator was called
        mock_generator.generate.assert_called_once()

        assert isinstance(result, str)
        assert len(result) > 0

    @patch('src.ir_core.orchestration.pipeline.default_dispatcher.execute_tool')
    def test_pipeline_without_enhancement(self, mock_execute_tool, mock_generator):
        """Test pipeline with query enhancement disabled."""
        # Create pipeline without enhancement
        pipeline = RAGPipeline(
            generator=mock_generator,
            tool_prompt_description="Search for scientific information",
            tool_calling_model="gpt-3.5-turbo",
            use_query_enhancement=False
        )

        mock_execute_tool.return_value = [
            {
                'id': 'doc1',
                'content': 'Test content about the topic.',
                'score': 0.8
            }
        ]

        result = pipeline.run("What is photosynthesis?")

        # Verify tool execution was called with original query
        mock_execute_tool.assert_called_once()

        # Verify generator was called
        pipeline.generator.generate.assert_called_once()

        assert isinstance(result, str)

    def test_enhancement_manager_creation(self):
        """Test that enhancement manager is properly created when enabled."""
        pipeline = RAGPipeline(
            generator=Mock(),
            tool_prompt_description="Test",
            tool_calling_model="gpt-3.5-turbo",
            use_query_enhancement=True
        )

        assert pipeline.enhancement_manager is not None
        assert isinstance(pipeline.enhancement_manager, QueryEnhancementManager)

    def test_enhancement_manager_disabled(self):
        """Test that enhancement manager is None when disabled."""
        pipeline = RAGPipeline(
            generator=Mock(),
            tool_prompt_description="Test",
            tool_calling_model="gpt-3.5-turbo",
            use_query_enhancement=False
        )

        assert pipeline.enhancement_manager is None


class TestQueryEnhancementTechniquesIntegration:
    """Integration tests for individual query enhancement techniques."""

    @pytest.fixture
    def enhancement_manager(self):
        """Create query enhancement manager for testing."""
        return QueryEnhancementManager()

    @patch('src.ir_core.query_enhancement.rewriter.QueryRewriter.rewrite_query')
    def test_rewriting_technique_integration(self, mock_rewrite, enhancement_manager):
        """Test rewriting technique in manager context."""
        mock_rewrite.return_value = "enhanced scientific query with synonyms"

        result = enhancement_manager.enhance_query("What is DNA?", technique="rewriting")

        assert result['enhanced'] is True
        assert result['technique_used'] == 'rewriting'
        assert 'enhanced scientific query' in result['enhanced_query']

    @patch('src.ir_core.query_enhancement.step_back.StepBackPrompting.step_back')
    def test_step_back_technique_integration(self, mock_step_back, enhancement_manager):
        """Test step-back technique in manager context."""
        mock_step_back.return_value = "genetic information storage molecular biology"

        result = enhancement_manager.enhance_query("What does DNA do?", technique="step_back")

        assert result['enhanced'] is True
        assert result['technique_used'] == 'step_back'

    @patch('src.ir_core.query_enhancement.hyde.HyDE.retrieve_with_hyde')
    def test_hyde_technique_integration(self, mock_hyde_retrieve, enhancement_manager):
        """Test HyDE technique in manager context."""
        mock_hyde_retrieve.return_value = [
            {'id': 'doc1', 'content': 'DNA content', 'score': 0.9}
        ]

        result = enhancement_manager.enhance_query("DNA structure", technique="hyde")

        assert result['enhanced'] is True
        assert result['technique_used'] == 'hyde'
        assert 'hyde_info' in result

    def test_auto_selection_logic(self, enhancement_manager):
        """Test automatic technique selection logic."""
        # Test with short query (should be suitable for multiple techniques)
        short_query = "What is water?"
        analysis = enhancement_manager._analyze_query(short_query)

        assert analysis['length'] <= 5
        assert analysis['has_question_words'] is True
        assert analysis['should_use_hyde'] is True

        # Should be able to select a technique
        selected = enhancement_manager._select_technique(analysis)
        assert selected is not None

    def test_technique_matching(self, enhancement_manager):
        """Test technique matching logic."""
        # Test ambiguous query
        ambiguous_analysis = {
            'is_ambiguous': True,
            'has_question_words': True,
            'length': 8
        }

        # Should match step_back for ambiguous queries
        matches_step_back = enhancement_manager._technique_matches_query('step_back', ambiguous_analysis)
        assert matches_step_back is True

        # Test short query
        short_analysis = {
            'length': 3,
            'should_use_hyde': True,
            'has_question_words': True
        }

        # Should match hyde for short queries
        matches_hyde = enhancement_manager._technique_matches_query('hyde', short_analysis)
        assert matches_hyde is True


class TestPipelinePerformanceComparison:
    """Tests to compare performance of enhanced vs non-enhanced pipeline."""

    @pytest.fixture
    def sample_queries(self):
        """Sample queries for performance testing."""
        return [
            "What is photosynthesis?",
            "How does evolution work?",
            "Explain quantum mechanics",
            "What are black holes?",
            "How do vaccines work?"
        ]

    @patch('src.ir_core.orchestration.pipeline.default_dispatcher.execute_tool')
    def test_enhanced_vs_basic_pipeline(self, mock_execute_tool, sample_queries, mock_generator):
        """Compare enhanced pipeline vs basic pipeline performance."""
        # Setup mock to return different results for different queries
        def mock_tool_execution(*args, **kwargs):
            return [
                {
                    'id': f'doc_{hash(str(args)) % 1000}',
                    'content': f'Mock content for query: {str(args)}',
                    'score': 0.8
                }
            ]

        mock_execute_tool.side_effect = mock_tool_execution

        # Create both pipelines
        enhanced_pipeline = RAGPipeline(
            generator=mock_generator,
            tool_prompt_description="Search for scientific information",
            tool_calling_model="gpt-3.5-turbo",
            use_query_enhancement=True
        )

        basic_pipeline = RAGPipeline(
            generator=mock_generator,
            tool_prompt_description="Search for scientific information",
            tool_calling_model="gpt-3.5-turbo",
            use_query_enhancement=False
        )

        # Test with sample queries
        for query in sample_queries[:2]:  # Test subset for speed
            enhanced_result = enhanced_pipeline.run(query)
            basic_result = basic_pipeline.run(query)

            # Both should return results
            assert isinstance(enhanced_result, str)
            assert isinstance(basic_result, str)
            assert len(enhanced_result) > 0
            assert len(basic_result) > 0

    def test_configuration_persistence(self):
        """Test that configuration settings are properly applied."""
        # Test with custom configuration
        custom_config = {
            'enabled': True,
            'techniques': {
                'rewriting': {'enabled': True, 'priority': 1},
                'hyde': {'enabled': False, 'priority': 4}  # Disable HyDE
            }
        }

        manager = QueryEnhancementManager(config=custom_config)

        # Should have rewriting enabled
        available = manager.get_available_techniques()
        assert 'rewriting' in available

        # Should not have hyde available
        assert 'hyde' not in available