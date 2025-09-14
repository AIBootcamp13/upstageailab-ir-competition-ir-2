# src/ir_core/query_enhancement/manager.py

from typing import List, Dict, Any, Optional, Union
import openai
from ..config import settings
from .llm_client import LLMClient, create_llm_client, detect_client_type
from .rewriter import QueryRewriter
from .step_back import StepBackPrompting
from .decomposer import QueryDecomposer
from .hyde import HyDE
from .translator import QueryTranslator


class QueryEnhancementManager:
    """
    Manager class that coordinates all query enhancement techniques.

    This class analyzes queries and applies the most appropriate enhancement
    technique(s) based on query characteristics and configuration settings.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        openai_client: Optional[openai.OpenAI] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Query Enhancement Manager.

        Args:
            model_name: Name of the model to use (e.g., "gpt-3.5-turbo" or "qwen2:7b").
                       If None, uses settings value.
            openai_client: Pre-configured OpenAI client. If None, creates a new one.
            config: Configuration dictionary. If None, uses settings.
        """
        self.config = config or getattr(settings, 'query_enhancement', {})

        # Determine model name
        if model_name is None:
            model_name = self.config.get('openai_model', 'gpt-3.5-turbo')

        self.model_name = model_name

        # Create appropriate LLM client
        if openai_client:
            # If OpenAI client is provided, use it
            from .llm_client import OpenAIClient
            self.llm_client = OpenAIClient(openai_client)
        else:
            # Auto-detect client type based on model name
            client_type = detect_client_type(self.model_name)
            if client_type == "openai":
                self.llm_client = create_llm_client("openai")
            elif client_type == "ollama":
                self.llm_client = create_llm_client("ollama", model_name=self.model_name)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

        # Keep backward compatibility
        self.client = openai_client or openai.OpenAI()

        # Initialize all enhancement techniques with LLM client
        self.rewriter = QueryRewriter(
            llm_client=self.llm_client,
            model_name=self.model_name,
            max_tokens=self.config.get('max_tokens'),
            temperature=self.config.get('temperature')
        )

        self.step_back = StepBackPrompting(
            llm_client=self.llm_client,
            model_name=self.model_name,
            max_tokens=self.config.get('max_tokens'),
            temperature=self.config.get('temperature')
        )

        self.decomposer = QueryDecomposer(
            llm_client=self.llm_client,
            model_name=self.model_name,
            max_tokens=self.config.get('max_tokens'),
            temperature=self.config.get('temperature')
        )

        self.hyde = HyDE(
            llm_client=self.llm_client,
            model_name=self.model_name,
            max_tokens=self.config.get('max_tokens'),
            temperature=self.config.get('temperature')
        )

        self.translator = QueryTranslator(
            llm_client=self.llm_client,
            model_name=self.model_name,
            max_tokens=self.config.get('max_tokens'),
            temperature=self.config.get('temperature')
        )

        # Get technique priorities and enabled status
        self.techniques_config = self.config.get('techniques', {})

    def enhance_query(self, query: str, technique: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhance a query using the most appropriate technique(s).

        Args:
            query: Original query to enhance
            technique: Specific technique to use. If None, auto-select.

        Returns:
            Dictionary with enhancement results and metadata
        """
        if not self.config.get('enabled', True):
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'none',
                'reason': 'Query enhancement disabled in configuration'
            }

        if technique:
            return self._apply_specific_technique(query, technique)
        else:
            return self._auto_select_and_apply(query)

    def _apply_specific_technique(self, query: str, technique: str) -> Dict[str, Any]:
        """
        Apply a specific enhancement technique.

        Args:
            query: Original query
            technique: Technique to apply

        Returns:
            Enhancement results
        """
        technique_map = {
            'rewriting': self._apply_rewriting,
            'step_back': self._apply_step_back,
            'decomposition': self._apply_decomposition,
            'hyde': self._apply_hyde,
            'translation': self._apply_translation
        }

        if technique not in technique_map:
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': technique,
                'error': f'Unknown technique: {technique}'
            }

        try:
            return technique_map[technique](query)
        except Exception as e:
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': technique,
                'error': str(e)
            }

    def _auto_select_and_apply(self, query: str) -> Dict[str, Any]:
        """
        Automatically select and apply the best enhancement technique.

        Args:
            query: Original query

        Returns:
            Enhancement results
        """
        # Analyze query characteristics
        query_analysis = self._analyze_query(query)

        # Select technique based on analysis and priorities
        selected_technique = self._select_technique(query_analysis)

        if not selected_technique:
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'none',
                'reason': 'No suitable technique found'
            }

        return self._apply_specific_technique(query, selected_technique)

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query characteristics to inform technique selection.

        Args:
            query: Query to analyze

        Returns:
            Analysis results
        """
        analysis = {
            'length': len(query.split()),
            'has_question_words': self._has_question_words(query),
            'is_complex': self.decomposer.should_decompose(query),
            'detected_language': self.translator.detect_language(query),
            'needs_translation': self.translator.should_translate(query),
            'is_ambiguous': self.step_back._is_ambiguous(query),
            'should_use_hyde': self.hyde.should_use_hyde(query)
        }

        return analysis

    def _has_question_words(self, query: str) -> bool:
        """
        Check if query contains question words.

        Args:
            query: Query to check

        Returns:
            True if question words are present
        """
        question_words = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose',
            '무엇', '어떻게', '왜', '언제', '어디', '누구', '어느', '누구의'
        ]
        query_lower = query.lower()
        return any(word in query_lower for word in question_words)

    def _select_technique(self, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Select the most appropriate technique based on analysis.

        Args:
            analysis: Query analysis results

        Returns:
            Selected technique name or None
        """
        techniques = self.techniques_config

        # Priority-based selection
        priority_order = ['rewriting', 'step_back', 'decomposition', 'hyde', 'translation']

        for technique in priority_order:
            if not techniques.get(technique, {}).get('enabled', False):
                continue

            if self._technique_matches_query(technique, analysis):
                return technique

        return None

    def _technique_matches_query(self, technique: str, analysis: Dict[str, Any]) -> bool:
        """
        Check if a technique is suitable for the analyzed query.

        Args:
            technique: Technique name
            analysis: Query analysis

        Returns:
            True if technique matches query characteristics
        """
        if technique == 'rewriting':
            # Default technique for most queries
            return True

        elif technique == 'step_back':
            # Good for ambiguous queries
            return analysis.get('is_ambiguous', False)

        elif technique == 'decomposition':
            # Good for complex queries
            return analysis.get('is_complex', False)

        elif technique == 'hyde':
            # Good for short queries or questions
            return (analysis.get('should_use_hyde', False) or
                   analysis.get('has_question_words', False) or
                   analysis.get('length', 0) <= 5)

        elif technique == 'translation':
            # Good for non-English queries
            return analysis.get('needs_translation', False)

        return False

    def _apply_rewriting(self, query: str) -> Dict[str, Any]:
        """Apply query rewriting technique."""
        enhanced_query = self.rewriter.rewrite_query(query)
        return {
            'enhanced': True,
            'original_query': query,
            'enhanced_query': enhanced_query,
            'technique_used': 'rewriting',
            'confidence': 0.8
        }

    def _apply_step_back(self, query: str) -> Dict[str, Any]:
        """Apply step-back prompting technique."""
        enhanced_query = self.step_back.step_back(query)
        return {
            'enhanced': True,
            'original_query': query,
            'enhanced_query': enhanced_query,
            'technique_used': 'step_back',
            'confidence': 0.7
        }

    def _apply_decomposition(self, query: str) -> Dict[str, Any]:
        """Apply query decomposition technique."""
        decomposition_info = self.decomposer.enhance_complex_query(query)
        return {
            'enhanced': decomposition_info['should_decompose'],
            'original_query': query,
            'enhanced_query': query,  # Decomposition doesn't produce a single query
            'technique_used': 'decomposition',
            'decomposition_info': decomposition_info,
            'confidence': 0.9 if decomposition_info['should_decompose'] else 0.0
        }

    def _apply_hyde(self, query: str) -> Dict[str, Any]:
        """Apply HyDE technique."""
        hyde_info = self.hyde.enhance_with_hyde(query)
        return {
            'enhanced': hyde_info['used_hyde'],
            'original_query': query,
            'enhanced_query': query,  # HyDE works with embeddings, not text
            'technique_used': 'hyde',
            'hyde_info': hyde_info,
            'confidence': 0.8 if hyde_info['used_hyde'] else 0.0
        }

    def _apply_translation(self, query: str) -> Dict[str, Any]:
        """Apply query translation technique."""
        translation_info = self.translator.enhance_with_translation(query)
        enhanced_query = translation_info.get('translated_query', query)
        return {
            'enhanced': translation_info['translated'],
            'original_query': query,
            'enhanced_query': enhanced_query,
            'technique_used': 'translation',
            'translation_info': translation_info,
            'confidence': 0.6
        }

    def get_available_techniques(self) -> List[str]:
        """
        Get list of available and enabled techniques.

        Returns:
            List of technique names
        """
        techniques = []
        for technique_name, config in self.techniques_config.items():
            if config.get('enabled', False):
                techniques.append(technique_name)
        return techniques

    def get_technique_info(self, technique: str) -> Dict[str, Any]:
        """
        Get information about a specific technique.

        Args:
            technique: Technique name

        Returns:
            Technique information
        """
        technique_info = {
            'rewriting': {
                'description': 'Expands queries with synonyms and related terms',
                'best_for': 'General queries that need more context',
                'priority': self.techniques_config.get('rewriting', {}).get('priority', 1)
            },
            'step_back': {
                'description': 'Identifies underlying concepts for ambiguous queries',
                'best_for': 'Vague or ambiguous questions',
                'priority': self.techniques_config.get('step_back', {}).get('priority', 2)
            },
            'decomposition': {
                'description': 'Breaks complex queries into simpler sub-queries',
                'best_for': 'Multi-part or complex questions',
                'priority': self.techniques_config.get('decomposition', {}).get('priority', 3)
            },
            'hyde': {
                'description': 'Uses hypothetical answer embeddings for retrieval',
                'best_for': 'Short queries or questions',
                'priority': self.techniques_config.get('hyde', {}).get('priority', 4)
            },
            'translation': {
                'description': 'Translates queries for better multilingual retrieval',
                'best_for': 'Non-English queries',
                'priority': self.techniques_config.get('translation', {}).get('priority', 5)
            }
        }

        return technique_info.get(technique, {})