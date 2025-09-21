# src/ir_core/query_enhancement/manager.py

from typing import List, Dict, Any, Optional, Union, Set
import os
import openai
from functools import lru_cache
from ..config import settings
from .llm_client import LLMClient, create_llm_client, detect_client_type
from .rewriter import QueryRewriter
from .step_back import StepBackPrompting
from .decomposer import QueryDecomposer
from .hyde import HyDE
from .translator import QueryTranslator
from .strategic_classifier import StrategicQueryClassifier, QueryType
from ..analysis.query_analyzer import QueryAnalyzer
from .constants import QUERY_TYPE_MAPPING, DEFAULT_CONFIDENCE_SCORES, TECHNIQUE_PRIORITY_ORDER
from .utils import has_question_words, has_conversational_indicators
from .prompts import format_query_intent_prompt
from .confidence_logger import log_confidence_score, log_error_confidence, log_fallback_triggered


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
            client_type = detect_client_type(str(self.model_name))
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

        # Initialize strategic classifier
        self.strategic_classifier = StrategicQueryClassifier()

        # Initialize centralized query analyzer for fallback classification
        self.query_analyzer = QueryAnalyzer()

        # Get technique priorities and enabled status
        self.techniques_config = self.config.get('techniques', {})

        # Strategic classifier settings
        self.use_strategic_classifier = self.config.get('use_strategic_classifier', True)
        self.strategic_config = self.config.get('strategic_classifier', {})

    def enhance_query(self, query: str, technique: Optional[str] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Enhance a query using the most appropriate technique(s) based on strategic classification.

        Args:
            query: Original query to enhance
            technique: Specific technique to use. If None, auto-select based on classification.
            conversation_history: Previous conversation turns for context-aware classification

        Returns:
            Dictionary with enhancement results and metadata
        """
        if not self.config.get('enabled', True):
            confidence = DEFAULT_CONFIDENCE_SCORES.get('none', 0.5)

            # Log confidence score
            log_confidence_score(
                technique='none',
                confidence=confidence,
                query=query,
                reasoning="Query enhancement disabled in configuration",
                context={}
            )

            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'none',
                'confidence': confidence,
                'reason': 'Query enhancement disabled in configuration'
            }

        # --- REFACTOR: Addresses [R-02] ---
        # Determine disabled techniques based on environment/config and pass as a parameter
        # This avoids modifying the instance state (self.techniques_config)
        disabled_techniques = set()
        is_evaluation_mode = os.getenv('RAG_EVALUATION_MODE', 'false').lower() == 'true'
        if is_evaluation_mode:
            hyde_config = self.techniques_config.get('hyde', {})
            if hyde_config.get('disable_for_evaluation', False):
                disabled_techniques.add('hyde')

        # If specific technique requested, apply it directly
        if technique:
            if technique in disabled_techniques:
                confidence = DEFAULT_CONFIDENCE_SCORES.get('none', 0.5)

                # Log confidence score
                log_confidence_score(
                    technique='none',
                    confidence=confidence,
                    query=query,
                    reasoning=f"Technique '{technique}' is disabled in the current mode",
                    context={'disabled_technique': technique}
                )

                return {
                    'enhanced': False,
                    'original_query': query,
                    'enhanced_query': query,
                    'technique_used': 'none',
                    'confidence': confidence,
                    'reason': f"Technique '{technique}' is disabled in the current mode."
                }
            return self._apply_specific_technique(query, technique)

        # Main logic for auto-selection
        return self._auto_select_and_enhance(query, conversation_history, disabled_techniques)

    def _auto_select_and_enhance(self, query: str, conversation_history: Optional[List[Dict[str, str]]], disabled_techniques: Set[str]) -> Dict[str, Any]:
        """
        Internal implementation of query enhancement logic.
        """
        # Convert conversation_history to a hashable type for caching
        history_tuple = tuple(tuple(d.items()) for d in conversation_history) if conversation_history else None

        # Classify query intent using LLM for better accuracy
        intent = self._classify_query_intent(query, history_tuple)

        # SHORT-CIRCUIT: If conversational, bypass entire RAG pipeline
        if intent in ["conversational", "conversational_follow_up"]:
            import logging
            logging.info(f"Intent classified as '{intent}'. Bypassing RAG pipeline.")
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': "CONVERSATIONAL_SKIP",  # Special flag for pipeline to detect
                'technique_used': 'CONVERSATIONAL_SKIP',
                'reason': f'Conversational query ({intent}) - bypassing entire RAG pipeline',
                'intent': intent
            }

        # Use strategic classifier for technique selection
        if self.use_strategic_classifier:
            classification = self.strategic_classifier.classify_query(query)
        else:
            classification = self._fallback_classification(query)

        # Check if retrieval should be bypassed
        if self.strategic_classifier.should_bypass_retrieval(classification):
            confidence = DEFAULT_CONFIDENCE_SCORES.get('bypass', 0.9)

            # Log confidence score
            log_confidence_score(
                technique='bypass',
                confidence=confidence,
                query=query,
                reasoning="Strategic classifier recommends bypassing enhancement",
                context={'classification': classification}
            )

            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'bypass',
                'confidence': confidence,
                'reason': 'Strategic classifier recommends bypassing enhancement',
                'classification': classification,
                'intent': intent
            }

        return self._auto_select_and_apply_with_classification(query, classification, disabled_techniques)

    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """
        Fallback classification when strategic classifier is disabled.

        Args:
            query: Query to classify

        Returns:
            Simple classification result
        """
        # Simple analysis without strategic classifier
        analysis = self._analyze_query(query)

        # Determine technique using old priority-based approach
        selected_technique = self._select_technique(analysis)

        return {
            'primary_type': 'unknown',
            'scores': {},
            'recommended_techniques': [{
                'technique': selected_technique or 'rewriting',
                'priority': 1,
                'reason': 'Fallback classification'
            }],
            'confidence': 0.5,
            'query_length': len(query.split()),
            'analysis': 'Fallback classification used'
        }

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
            # Apply primary technique
            result = technique_map[technique](query)

            # If HyDE produced no effective enhancement, fall back to rewriting
            if technique == 'hyde':
                try:
                    enhanced_query = result.get('enhanced_query', query)
                    retrieval_results = result.get('retrieval_results', [])
                    original_confidence = result.get('confidence', 0.0)

                    # Consider HyDE ineffective if it returned the original query or no results
                    ineffective = (enhanced_query.strip() == query.strip()) or (not retrieval_results)
                    if ineffective:
                        # Log fallback triggered
                        log_fallback_triggered(
                            original_technique='hyde',
                            fallback_technique='rewriting',
                            original_confidence=original_confidence,
                            query=query,
                            reason="HyDE ineffective - no retrieval results or unchanged query"
                        )

                        fallback = self._apply_rewriting(query)
                        # Mark fallback metadata
                        fallback['technique_used'] = 'rewriting'
                        fallback['fallback_from'] = 'hyde'
                        return fallback
                except Exception:
                    # On any issue, degrade gracefully to rewriting
                    log_fallback_triggered(
                        original_technique='hyde',
                        fallback_technique='rewriting',
                        original_confidence=result.get('confidence', 0.0),
                        query=query,
                        reason="HyDE application failed with exception"
                    )

                    fallback = self._apply_rewriting(query)
                    fallback['technique_used'] = 'rewriting'
                    fallback['fallback_from'] = 'hyde'
                    return fallback

            return result
        except Exception as e:
            # Log error confidence
            log_error_confidence(
                technique=technique,
                error=str(e),
                query=query
            )

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

    def _auto_select_and_apply_with_classification(self, query: str, classification: Dict[str, Any], disabled_techniques: Set[str]) -> Dict[str, Any]:
        """
        Automatically select and apply the best enhancement technique based on strategic classification.

        Args:
            query: Original query
            classification: Strategic classification results

        Returns:
            Enhancement results
        """
        recommended_techniques = classification['recommended_techniques']

        if not recommended_techniques:
            confidence = DEFAULT_CONFIDENCE_SCORES.get('none', 0.5)

            # Log confidence score
            log_confidence_score(
                technique='none',
                confidence=confidence,
                query=query,
                reasoning="No suitable technique found",
                context={}
            )

            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'none',
                'confidence': confidence,
                'reason': 'No suitable technique found'
            }

        # Apply the highest priority technique that is enabled
        for technique_info in recommended_techniques:
            technique = technique_info['technique']
            if technique == 'bypass':
                confidence = DEFAULT_CONFIDENCE_SCORES.get('bypass', 0.9)

                # Log confidence score
                log_confidence_score(
                    technique='bypass',
                    confidence=confidence,
                    query=query,
                    reasoning="Strategic classifier recommends bypassing enhancement",
                    context={'classification': classification}
                )

                return {
                    'enhanced': False,
                    'original_query': query,
                    'enhanced_query': query,
                    'technique_used': 'bypass',
                    'confidence': confidence,
                    'reason': 'Strategic classifier recommends bypassing enhancement'
                }

            # --- REFACTOR: Addresses [R-02] ---
            # Check against the passed `disabled_techniques` set
            if technique not in disabled_techniques and self.techniques_config.get(technique, {}).get('enabled', False):
                primary_technique = technique
                break
        else:
            # No enabled techniques found, fall back to no enhancement
            confidence = DEFAULT_CONFIDENCE_SCORES.get('none', 0.5)

            # Log confidence score
            log_confidence_score(
                technique='none',
                confidence=confidence,
                query=query,
                reasoning="No enabled techniques available from classifier recommendations",
                context={'recommended_techniques': recommended_techniques}
            )

            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'none',
                'confidence': confidence,
                'reason': 'No enabled techniques available from classifier recommendations'
            }

        # Check if technique supports sequential application
        if len(recommended_techniques) > 1 and primary_technique == 'decomposition':
            return self._apply_sequential_techniques(query, recommended_techniques, disabled_techniques)

        # Apply single technique
        return self._apply_specific_technique(query, primary_technique)

    def _apply_sequential_techniques(self, query: str, techniques: List[Dict[str, Any]], disabled_techniques: Set[str]) -> Dict[str, Any]:
        """
        Apply multiple techniques in sequence (e.g., decomposition then HyDE).

        Args:
            query: Original query
            techniques: List of techniques to apply in order

        Returns:
            Combined enhancement results
        """
        current_query = query
        applied_techniques = []
        final_result = None

        for technique_info in techniques:
            technique = technique_info['technique']

            if technique == 'decomposition':
                # Check if technique is disabled
                if technique in disabled_techniques:
                    continue
                # Apply decomposition first
                decomp_result = self._apply_decomposition(current_query)
                if decomp_result['enhanced'] and decomp_result.get('decomposition_info', {}).get('should_decompose'):
                    applied_techniques.append('decomposition')
                    # For decomposed queries, we don't change current_query as decomposition
                    # produces sub-queries that are handled separately
                    final_result = decomp_result
                    break  # Decomposition is typically the final step for complex queries

            elif technique == 'hyde':
                # Check if technique is disabled
                if technique in disabled_techniques:
                    continue
                # Apply HyDE
                hyde_result = self._apply_hyde(current_query)
                if hyde_result['enhanced']:
                    applied_techniques.append('hyde')
                    current_query = hyde_result['enhanced_query']
                    final_result = hyde_result

            elif technique == 'rewriting':
                # Check if technique is disabled
                if technique in disabled_techniques:
                    continue
                # Apply rewriting
                rewrite_result = self._apply_rewriting(current_query)
                if rewrite_result['enhanced']:
                    applied_techniques.append('rewriting')
                    current_query = rewrite_result['enhanced_query']
                    final_result = rewrite_result

            elif technique == 'step_back':
                # Check if technique is disabled
                if technique in disabled_techniques:
                    continue
                # Apply step-back
                stepback_result = self._apply_step_back(current_query)
                if stepback_result['enhanced']:
                    applied_techniques.append('step_back')
                    current_query = stepback_result['enhanced_query']
                    final_result = stepback_result

        if final_result:
            final_result['applied_techniques'] = applied_techniques
            final_result['sequential_application'] = True
            return final_result

        # Fallback to basic rewriting if no techniques succeeded
        return self._apply_rewriting(query)

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
            'has_question_words': has_question_words(query),
            'is_complex': self.decomposer.should_decompose(query),
            'detected_language': self.translator.detect_language(query),
            'needs_translation': self.translator.should_translate(query),
            'is_ambiguous': self.step_back._is_ambiguous(query),
            'should_use_hyde': self.hyde.should_use_hyde(query)
        }

        return analysis

    def _select_technique(self, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Select the most appropriate technique based on analysis.

        Args:
            analysis: Query analysis results

        Returns:
            Selected technique name or None
        """
        techniques = self.techniques_config

        # First, try the default technique if specified and enabled
        default_technique = self.config.get('default_technique')
        if default_technique and techniques.get(default_technique, {}).get('enabled', False):
            if self._technique_matches_query(default_technique, analysis):
                return default_technique

        # Fallback to priority-based selection
        for technique in TECHNIQUE_PRIORITY_ORDER:
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
        confidence = DEFAULT_CONFIDENCE_SCORES.get('rewriting', 0.8)

        # Log confidence score
        log_confidence_score(
            technique='rewriting',
            confidence=confidence,
            query=query,
            reasoning="Standard query rewriting applied",
            context={'enhanced_query': enhanced_query}
        )

        return {
            'enhanced': True,
            'original_query': query,
            'enhanced_query': enhanced_query,
            'technique_used': 'rewriting',
            'confidence': confidence
        }

    def _apply_step_back(self, query: str) -> Dict[str, Any]:
        """Apply step-back prompting technique."""
        enhanced_query = self.step_back.step_back(query)
        confidence = DEFAULT_CONFIDENCE_SCORES.get('step_back', 0.7)

        # Log confidence score
        log_confidence_score(
            technique='step_back',
            confidence=confidence,
            query=query,
            reasoning="Step-back prompting applied to generalize query",
            context={'enhanced_query': enhanced_query}
        )

        return {
            'enhanced': True,
            'original_query': query,
            'enhanced_query': enhanced_query,
            'technique_used': 'step_back',
            'confidence': confidence
        }

    def _apply_decomposition(self, query: str) -> Dict[str, Any]:
        """Apply query decomposition technique.

        Confidence scoring:
        - 0.9: Decomposition recommended and applied
        - 0.0: Decomposition not needed
        """
        decomposition_info = self.decomposer.enhance_complex_query(query)
        should_decompose = decomposition_info['should_decompose']
        confidence = DEFAULT_CONFIDENCE_SCORES.get('decomposition', 0.9) if should_decompose else 0.0

        # Log confidence score
        reasoning = "Query decomposition applied - complex query detected" if should_decompose else "Query decomposition not needed - simple query"
        log_confidence_score(
            technique='decomposition',
            confidence=confidence,
            query=query,
            reasoning=reasoning,
            context={
                'should_decompose': should_decompose,
                'decomposition_reason': decomposition_info.get('reason', 'N/A')
            }
        )

        return {
            'enhanced': should_decompose,
            'original_query': query,
            'enhanced_query': query,  # Decomposition doesn't produce a single query
            'technique_used': 'decomposition',
            'decomposition_info': decomposition_info,
            'confidence': confidence
        }

    def _apply_hyde(self, query: str) -> Dict[str, Any]:
        """Apply HyDE technique with proper retrieval integration.

        Confidence scoring:
        - 0.9: Successful retrieval with results
        - 0.6: Generated answer but no retrieval results
        - 0.0: Complete failure or error
        """
        try:
            # Generate hypothetical answer and retrieve documents using its embedding
            hyde_results = self.hyde.retrieve_with_hyde(query, top_k=5)

            if hyde_results:
                # Use the hypothetical answer as enhanced query for downstream processing
                hypothetical_answer = self.hyde.generate_hypothetical_answer(query)
                confidence = DEFAULT_CONFIDENCE_SCORES.get('hyde', 0.9)

                # Log confidence score
                log_confidence_score(
                    technique='hyde',
                    confidence=confidence,
                    query=query,
                    reasoning="HyDE applied successfully with retrieval results",
                    context={
                        'result_count': len(hyde_results),
                        'hypothetical_answer': hypothetical_answer[:50] + '...'
                    }
                )

                return {
                    'enhanced': True,
                    'original_query': query,
                    'enhanced_query': hypothetical_answer,  # Use hypothetical answer as query
                    'technique_used': 'hyde',
                    'retrieval_results': hyde_results,
                    'result_count': len(hyde_results),
                    'confidence': confidence
                }
            else:
                # Fallback to just generating hypothetical answer
                hypothetical_answer = self.hyde.generate_hypothetical_answer(query)
                confidence = 0.6  # Lower confidence without retrieval results

                # Log confidence score
                log_confidence_score(
                    technique='hyde',
                    confidence=confidence,
                    query=query,
                    reasoning="HyDE applied but no retrieval results found",
                    context={
                        'result_count': 0,
                        'hypothetical_answer': hypothetical_answer[:50] + '...'
                    }
                )

                return {
                    'enhanced': True,
                    'original_query': query,
                    'enhanced_query': hypothetical_answer,
                    'technique_used': 'hyde',
                    'retrieval_results': [],
                    'result_count': 0,
                    'confidence': confidence
                }

        except Exception as e:
            print(f"HyDE application failed: {e}")
            confidence = 0.0  # Zero confidence on error

            # Log error confidence
            log_error_confidence(
                technique='hyde',
                error=str(e),
                query=query
            )

            # Fallback to original query
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'hyde',
                'error': str(e),
                'confidence': confidence
            }

    def _apply_translation(self, query: str) -> Dict[str, Any]:
        """Apply query translation technique."""
        translation_info = self.translator.enhance_with_translation(query)
        enhanced_query = translation_info.get('translated_query', query)
        translated = translation_info['translated']
        confidence = DEFAULT_CONFIDENCE_SCORES.get('translation', 0.6)

        # Log confidence score
        reasoning = "Query translation applied" if translated else "Query translation not needed"
        log_confidence_score(
            technique='translation',
            confidence=confidence,
            query=query,
            reasoning=reasoning,
            context={
                'translated': translated,
                'target_language': translation_info.get('target_language', 'N/A'),
                'enhanced_query': enhanced_query[:50] + '...' if len(enhanced_query) > 50 else enhanced_query
            }
        )

        return {
            'enhanced': translated,
            'original_query': query,
            'enhanced_query': enhanced_query,
            'technique_used': 'translation',
            'translation_info': translation_info,
            'confidence': confidence
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

    # --- REFACTOR: Addresses [R-03] ---
    @lru_cache(maxsize=1024)
    def _classify_query_intent(self, query: str, conversation_history: Optional[tuple] = None) -> str:
        """
        Classify query intent using LLM with detailed prompt for better accuracy.

        Args:
            query: The query to classify
            conversation_history: Previous conversation turns for context

        Returns:
            Classification category as string
        """
        # Enhanced classification prompt
        # Format conversation history
        if conversation_history:
            # Reconstruct history from the hashable tuple
            history_list = [dict(turn) for turn in conversation_history]
            history_text = "\n".join([f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
                                    for turn in history_list[-3:]])  # Last 3 turns
        else:
            history_text = "(이전 대화 없음)"

        formatted_prompt = format_query_intent_prompt(query, history_text)

        try:
            # Use LLM client to classify
            messages = [{"role": "user", "content": formatted_prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=50,
                temperature=0.1  # Low temperature for consistent classification
            )

            if response['success']:
                response_text = response['content'].strip().lower()

                # Map response to valid categories
                valid_categories = ['conversational', 'conversational_follow_up', 'simple_keyword', 'conceptual_vague']

                for category in valid_categories:
                    if category in response_text:
                        return category

                # Fallback to conceptual_vague if no match
                return 'conceptual_vague'
            else:
                # Fallback on API error
                import logging
                logging.warning(f"LLM classification failed: {response.get('error', 'Unknown error')}, falling back to regex classification")
                return self._fallback_llm_classification(query)

        except Exception as e:
            # Fallback to regex-based classification on error
            import logging
            logging.warning(f"LLM classification failed: {e}, falling back to regex classification")
            return self._fallback_llm_classification(query)

    def _fallback_llm_classification(self, query: str) -> str:
        """
        Fallback classification when LLM fails, using centralized analysis module.

        Args:
            query: Query to classify

        Returns:
            Classification category
        """
        try:
            # Use centralized QueryAnalyzer for classification
            features = self.query_analyzer.analyze_query(query)

            # Map query types to enhancement categories
            # Check for conversational patterns using query features
            query_lower = query.lower()
            if has_conversational_indicators(query_lower):
                return 'conversational'

            # Return mapped classification
            return QUERY_TYPE_MAPPING.get(features.query_type, "conceptual_vague")

        except Exception as e:
            # Final fallback to conceptual_vague
            import logging
            logging.warning(f"Centralized classification failed: {e}, using default")
            return 'conceptual_vague'