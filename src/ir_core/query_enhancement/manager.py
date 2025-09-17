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
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'none',
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
                 return {
                    'enhanced': False,
                    'original_query': query,
                    'enhanced_query': query,
                    'technique_used': 'none',
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
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'bypass',
                'reason': f'Query classified as {classification["primary_type"]} - bypassing retrieval',
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
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'none',
                'reason': 'No techniques recommended by classifier'
            }

        # Apply the highest priority technique that is enabled
        for technique_info in recommended_techniques:
            technique = technique_info['technique']
            if technique == 'bypass':
                return {
                    'enhanced': False,
                    'original_query': query,
                    'enhanced_query': query,
                    'technique_used': 'bypass',
                    'reason': 'Strategic classifier recommends bypassing enhancement'
                }

            # --- REFACTOR: Addresses [R-02] ---
            # Check against the passed `disabled_techniques` set
            if technique not in disabled_techniques and self.techniques_config.get(technique, {}).get('enabled', False):
                primary_technique = technique
                break
        else:
            # No enabled techniques found, fall back to no enhancement
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'none',
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

        # First, try the default technique if specified and enabled
        default_technique = self.config.get('default_technique')
        if default_technique and techniques.get(default_technique, {}).get('enabled', False):
            if self._technique_matches_query(default_technique, analysis):
                return default_technique

        # Fallback to priority-based selection
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
        """Apply HyDE technique with proper retrieval integration."""
        try:
            # Generate hypothetical answer and retrieve documents using its embedding
            hyde_results = self.hyde.retrieve_with_hyde(query, top_k=5)

            if hyde_results:
                # Use the hypothetical answer as enhanced query for downstream processing
                hypothetical_answer = self.hyde.generate_hypothetical_answer(query)

                return {
                    'enhanced': True,
                    'original_query': query,
                    'enhanced_query': hypothetical_answer,  # Use hypothetical answer as query
                    'technique_used': 'hyde',
                    'retrieval_results': hyde_results,
                    'result_count': len(hyde_results),
                    'confidence': 0.9
                }
            else:
                # Fallback to just generating hypothetical answer
                hypothetical_answer = self.hyde.generate_hypothetical_answer(query)
                return {
                    'enhanced': True,
                    'original_query': query,
                    'enhanced_query': hypothetical_answer,
                    'technique_used': 'hyde',
                    'retrieval_results': [],
                    'result_count': 0,
                    'confidence': 0.6
                }

        except Exception as e:
            print(f"HyDE application failed: {e}")
            # Fallback to original query
            return {
                'enhanced': False,
                'original_query': query,
                'enhanced_query': query,
                'technique_used': 'hyde',
                'error': str(e),
                'confidence': 0.0
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
        prompt = """당신은 사용자 질문의 의도를 분석하여 가장 적절한 처리 방식을 결정하는 AI 분류 전문가입니다.
다음 카테고리 중 하나로만 분류하여 그 키워드를 출력해야 합니다.

[카테고리]
- conversational: 사용자가 감정을 표현하거나, 인사하거나, AI 자체에 대해 묻는 등 정보 검색 목적이 아닌 일반 대화. (예: "고마워", "너는 누구니?", "오늘 기분 어때?", "요새 너무 힘들다")
- conversational_follow_up: 이전 대화의 맥락 없이는 의미가 불분명한 후속 질문. (예: "그럼 그건 왜 그런데?", "다른 예시는 없어?")
- simple_keyword: 명확한 키워드(고유명사, 전문 용어)를 포함하고 있어 추가적인 향상 없이도 검색이 가능한 직접적인 질문. (예: "엽록체의 역할은?", "라마 3.1 모델에 대해 알려줘.")
- conceptual_vague: "방법", "원리", "영향", "이유", "차이점" 등 키워드가 아닌 개념에 대해 묻는 광범위하거나 추상적인 질문. (예: "공생과 기생의 차이점은?", "지구 온난화가 해양 생태계에 미치는 영향은?", "나무가 생태계에서 하는 역할에 대해 설명해줘.")

[대화 기록]
{conversation_history}

[사용자 최근 질문]
{query}

[분류]
출력은 카테고리 키워드만 출력하세요. 설명을 추가하지 마세요."""

        # Format conversation history
        if conversation_history:
            # Reconstruct history from the hashable tuple
            history_list = [dict(turn) for turn in conversation_history]
            history_text = "\n".join([f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
                                    for turn in history_list[-3:]])  # Last 3 turns
        else:
            history_text = "(이전 대화 없음)"

        formatted_prompt = prompt.format(
            conversation_history=history_text,
            query=query
        )

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
        Fallback classification when LLM fails, using regex patterns.

        Args:
            query: Query to classify

        Returns:
            Classification category
        """
        query_lower = query.lower()

        # Check for conversational patterns
        conversational_keywords = ['너', '너는', '너의', '안녕', '반가워', '고마워', '미안', '기분', '힘들', '좋아', '싫어']
        if any(keyword in query_lower for keyword in conversational_keywords):
            return 'conversational'

        # Check for simple keyword patterns
        scientific_terms = ['dna', 'rna', '세포', '유전자', '단백질', '효소', '원자', '분자', '화학', '물리', '생물']
        if any(term in query_lower for term in scientific_terms):
            return 'simple_keyword'

        # Check for conceptual patterns
        conceptual_keywords = ['왜', '어떻게', '무엇을', '영향', '차이', '역할', '기능', '원리', '이유']
        if any(keyword in query_lower for keyword in conceptual_keywords):
            return 'conceptual_vague'

        # Default to conceptual_vague
        return 'conceptual_vague'