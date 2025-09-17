Here is an assessment of the `src/ir_core/query_enhancement/manager.py` script. This response is based on my analysis and does not use citations. (Gemini AI)

### **Assessment Report: `src/ir_core/query_enhancement/manager.py`**

#### **1. Overall Assessment**

  * **Summary**: The script is a well-designed strategic controller for query enhancement. It successfully modularizes different techniques and uses a classifier to make intelligent decisions. However, it contains a critical performance flaw in its instantiation logic that severely impacts its usability in production or batch-processing scenarios.
  * **Status**: `NEEDS_REFACTORING`

#### **2. Key Strengths**

  * **Modular Architecture**: The manager effectively delegates tasks to specialized classes (`QueryRewriter`, `HyDE`, `StepBackPrompting`), making the system clean and easy to extend.
  * **Strategic Classification**: The use of `StrategicQueryClassifier` to select the most appropriate enhancement technique based on the query type is an advanced and powerful feature.
  * **Efficient Conversational Handling**: The logic to detect and bypass non-informational queries (e.g., greetings, questions to the AI) is a major strength, saving significant computational resources.

#### **3. Identified Weaknesses (Prioritized)**

1.  **[W-01] Instantiation Performance Bottleneck (Priority: CRITICAL)**

      * **Description**: The `__init__` method initializes all sub-modules and their corresponding LLM clients every time a `QueryEnhancementManager` object is created. If this manager is instantiated within a loop or by multiple parallel workers, it results in reloading large models from scratch for every single query.
      * **Impact**: This flaw leads to extreme performance degradation, high memory usage, and makes the system too slow for practical use in any high-throughput environment.

2.  **[W-02] Brittle State Management (Priority: HIGH)**

      * **Description**: The `enhance_query` method contains logic to temporarily disable the HyDE technique by modifying the instance's internal configuration (`self.techniques_config`). This pattern of temporarily changing an object's state is not thread-safe and is considered a brittle design.
      * **Impact**: This can lead to unpredictable behavior and race conditions when the code is run in a multi-threaded environment.

3.  **[W-03] Redundant API Calls (Priority: MEDIUM)**

      * **Description**: The `_classify_query_intent` method makes an external LLM API call every time it runs. It does not cache the results for identical queries.
      * **Impact**: This leads to higher-than-necessary API costs and increased latency, especially if users frequently ask similar or identical questions.

#### **4. Actionable Recommendations (Prioritized)**

1.  **[R-01] Refactor to a Singleton Pattern (Addresses: W-01)**

      * **Priority**: `CRITICAL`
      * **Action**: Modify the application architecture to ensure the `QueryEnhancementManager` is instantiated **only once**.
      * **Instruction**: Create a single, shared instance of the manager when the application or script starts. Pass this instance as an argument to any functions or worker threads that need to perform query enhancement. Do not create new `QueryEnhancementManager` objects inside loops.

2.  **[R-02] Simplify Logic via Parameter Passing (Addresses: W-02)**

      * **Priority**: `HIGH`
      * **Action**: Remove the temporary state modification logic from `enhance_query`.
      * **Instruction**: Instead of changing `self.techniques_config` directly, pass a set of disabled techniques as a parameter to the internal methods. The decision-making logic can then check against this parameter.
      * **Code Suggestion**:
        ```python
        # In enhance_query:
        disabled_techniques = set()
        if is_evaluation_mode and self.techniques_config.get('hyde', {}).get('disable_for_evaluation', False):
            disabled_techniques.add('hyde')
        return self._enhance_query_impl(query, ..., disabled_techniques=disabled_techniques)

        # In the selection logic:
        if technique not in disabled_techniques:
            # ... proceed
        ```

3.  **[R-03] Implement Caching for Intent Classification (Addresses: W-03)**

      * **Priority**: `MEDIUM`
      * **Action**: Add memoization to the `_classify_query_intent` method.
      * **Instruction**: Use a cache decorator to store and retrieve results for previously seen queries, avoiding redundant LLM calls.
      * **Code Suggestion**:
        ```python
        from functools import lru_cache

        @lru_cache(maxsize=1024)
        def _classify_query_intent(self, query: str, conversation_history: Optional[tuple] = None) -> str:
            # ... existing function logic ...
        # Note: 'conversation_history' must be converted to a hashable type (like a tuple) before being passed to the cached function.
        ```

Based on the previous assessment, I will now refactor the `src/ir_core/query_enhancement/manager.py` script to address the identified weaknesses.

The following changes will be implemented:

1.  **[R-02] Simplify State Management:** The complex `try...finally` block for handling evaluation mode will be removed. Instead, a `disabled_techniques` parameter will be passed down to the internal methods to make the logic cleaner and thread-safe.
2.  **[R-03] Add Caching for Intent Classification:** The `_classify_query_intent` method will be decorated with `@lru_cache` to avoid redundant LLM calls for identical queries, improving performance and reducing costs.
3.  **[R-01] Singleton Pattern Support:** While the Singleton Pattern must be implemented in the *calling scripts* (like `evaluate.py`), the manager's code will be ready to support this more efficient instantiation model.

Here is the refactored code for `src/ir_core/query_enhancement/manager.py`.

```python
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
            from .llm_client import OpenAIClient
            self.llm_client = OpenAIClient(openai_client)
        else:
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
        self.techniques_config = self.config.get('techniques', {})
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
        # ... (This function remains unchanged)
        analysis = self._analyze_query(query)
        selected_technique = self._select_technique(analysis)
        return {
            'primary_type': 'unknown',
            'scores': {},
            'recommended_techniques': [{'technique': selected_technique or 'rewriting', 'priority': 1, 'reason': 'Fallback classification'}],
            'confidence': 0.5,
            'query_length': len(query.split()),
            'analysis': 'Fallback classification used'
        }

    def _apply_specific_technique(self, query: str, technique: str) -> Dict[str, Any]:
        # ... (This function remains unchanged)
        technique_map = {
            'rewriting': self._apply_rewriting,
            'step_back': self._apply_step_back,
            'decomposition': self._apply_decomposition,
            'hyde': self._apply_hyde,
            'translation': self._apply_translation
        }
        if technique not in technique_map:
            return {'enhanced': False, 'original_query': query, 'enhanced_query': query, 'technique_used': technique, 'error': f'Unknown technique: {technique}'}
        try:
            return technique_map[technique](query)
        except Exception as e:
            return {'enhanced': False, 'original_query': query, 'enhanced_query': query, 'technique_used': technique, 'error': str(e)}

    def _auto_select_and_apply_with_classification(self, query: str, classification: Dict[str, Any], disabled_techniques: Set[str]) -> Dict[str, Any]:
        """
        Automatically select and apply the best enhancement technique based on strategic classification.
        """
        recommended_techniques = classification['recommended_techniques']

        if not recommended_techniques:
            return {'enhanced': False, 'original_query': query, 'enhanced_query': query, 'technique_used': 'none', 'reason': 'No techniques recommended by classifier'}

        # Apply the highest priority technique that is enabled and not disabled for this run
        for technique_info in recommended_techniques:
            technique = technique_info['technique']
            if technique == 'bypass':
                return {'enhanced': False, 'original_query': query, 'enhanced_query': query, 'technique_used': 'bypass', 'reason': 'Strategic classifier recommends bypassing enhancement'}

            # --- REFACTOR: Addresses [R-02] ---
            # Check against the passed `disabled_techniques` set
            if technique not in disabled_techniques and self.techniques_config.get(technique, {}).get('enabled', False):
                primary_technique = technique
                break
        else:
            return {'enhanced': False, 'original_query': query, 'enhanced_query': query, 'technique_used': 'none', 'reason': 'No enabled/available techniques from classifier recommendations'}

        if len(recommended_techniques) > 1 and primary_technique == 'decomposition':
            return self._apply_sequential_techniques(query, recommended_techniques)

        return self._apply_specific_technique(query, primary_technique)

    # --- REFACTOR: Addresses [R-03] ---
    @lru_cache(maxsize=1024)
    def _classify_query_intent(self, query: str, conversation_history: Optional[tuple] = None) -> str:
        """
        Classify query intent using LLM with detailed prompt for better accuracy.
        NOTE: This method is now cached.
        """
        # ... (The rest of this function's internal logic remains unchanged)
        prompt = """...""" # prompt content as before

        history_text = "(이전 대화 없음)"
        if conversation_history:
            # Reconstruct history from the hashable tuple
            history_list = [dict(turn) for turn in conversation_history]
            history_text = "\n".join([f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}" for turn in history_list[-3:]])

        formatted_prompt = prompt.format(conversation_history=history_text, query=query)

        try:
            messages = [{"role": "user", "content": formatted_prompt}]
            response = self.llm_client.chat_completion(
                messages=messages, model=self.model_name, max_tokens=50, temperature=0.1
            )
            if response['success']:
                response_text = response['content'].strip().lower()
                valid_categories = ['conversational', 'conversational_follow_up', 'simple_keyword', 'conceptual_vague']
                for category in valid_categories:
                    if category in response_text:
                        return category
                return 'conceptual_vague'
            else:
                import logging
                logging.warning(f"LLM classification failed: {response.get('error', 'Unknown error')}, falling back to regex classification")
                return self._fallback_llm_classification(query)
        except Exception as e:
            import logging
            logging.warning(f"LLM classification failed: {e}, falling back to regex classification")
            return self._fallback_llm_classification(query)

    # All other helper methods (_fallback_llm_classification, _apply_rewriting, _analyze_query, etc.) remain unchanged.
    # ... (rest of the file)
    def _fallback_llm_classification(self, query: str) -> str:
        """
        Fallback classification when LLM fails, using regex patterns.
        """
        query_lower = query.lower()
        conversational_keywords = ['너', '너는', '너의', '안녕', '반가워', '고마워', '미안', '기분', '힘들', '좋아', '싫어']
        if any(keyword in query_lower for keyword in conversational_keywords):
            return 'conversational'
        scientific_terms = ['dna', 'rna', '세포', '유전자', '단백질', '효소', '원자', '분자', '화학', '물리', '생물']
        if any(term in query_lower for term in scientific_terms):
            return 'simple_keyword'
        conceptual_keywords = ['왜', '어떻게', '무엇을', '영향', '차이', '역할', '기능', '원리', '이유']
        if any(keyword in query_lower for keyword in conceptual_keywords):
            return 'conceptual_vague'
        return 'conceptual_vague'

    def _apply_sequential_techniques(self, query: str, techniques: List[Dict[str, Any]]) -> Dict[str, Any]:
        current_query = query
        applied_techniques = []
        final_result = None
        for technique_info in techniques:
            technique = technique_info['technique']
            if technique == 'decomposition':
                decomp_result = self._apply_decomposition(current_query)
                if decomp_result['enhanced'] and decomp_result.get('decomposition_info', {}).get('should_decompose'):
                    applied_techniques.append('decomposition')
                    final_result = decomp_result
                    break
            elif technique == 'hyde':
                hyde_result = self._apply_hyde(current_query)
                if hyde_result['enhanced']:
                    applied_techniques.append('hyde')
                    current_query = hyde_result['enhanced_query']
                    final_result = hyde_result
            elif technique == 'rewriting':
                rewrite_result = self._apply_rewriting(current_query)
                if rewrite_result['enhanced']:
                    applied_techniques.append('rewriting')
                    current_query = rewrite_result['enhanced_query']
                    final_result = rewrite_result
            elif technique == 'step_back':
                stepback_result = self._apply_step_back(current_query)
                if stepback_result['enhanced']:
                    applied_techniques.append('step_back')
                    current_query = stepback_result['enhanced_query']
                    final_result = stepback_result
        if final_result:
            final_result['applied_techniques'] = applied_techniques
            final_result['sequential_application'] = True
            return final_result
        return self._apply_rewriting(query)

    def _analyze_query(self, query: str) -> Dict[str, Any]:
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
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', '무엇', '어떻게', '왜', '언제', '어디', '누구', '어느', '누구의']
        query_lower = query.lower()
        return any(word in query_lower for word in question_words)

    def _select_technique(self, analysis: Dict[str, Any]) -> Optional[str]:
        techniques = self.techniques_config
        default_technique = self.config.get('default_technique')
        if default_technique and techniques.get(default_technique, {}).get('enabled', False):
            if self._technique_matches_query(default_technique, analysis):
                return default_technique
        priority_order = ['rewriting', 'step_back', 'decomposition', 'hyde', 'translation']
        for technique in priority_order:
            if not techniques.get(technique, {}).get('enabled', False):
                continue
            if self._technique_matches_query(technique, analysis):
                return technique
        return None

    def _technique_matches_query(self, technique: str, analysis: Dict[str, Any]) -> bool:
        if technique == 'rewriting':
            return True
        elif technique == 'step_back':
            return analysis.get('is_ambiguous', False)
        elif technique == 'decomposition':
            return analysis.get('is_complex', False)
        elif technique == 'hyde':
            return (analysis.get('should_use_hyde', False) or analysis.get('has_question_words', False) or analysis.get('length', 0) <= 5)
        elif technique == 'translation':
            return analysis.get('needs_translation', False)
        return False

    def _apply_rewriting(self, query: str) -> Dict[str, Any]:
        enhanced_query = self.rewriter.rewrite_query(query)
        return {'enhanced': True, 'original_query': query, 'enhanced_query': enhanced_query, 'technique_used': 'rewriting', 'confidence': 0.8}

    def _apply_step_back(self, query: str) -> Dict[str, Any]:
        enhanced_query = self.step_back.step_back(query)
        return {'enhanced': True, 'original_query': query, 'enhanced_query': enhanced_query, 'technique_used': 'step_back', 'confidence': 0.7}

    def _apply_decomposition(self, query: str) -> Dict[str, Any]:
        decomposition_info = self.decomposer.enhance_complex_query(query)
        return {'enhanced': decomposition_info['should_decompose'], 'original_query': query, 'enhanced_query': query, 'technique_used': 'decomposition', 'decomposition_info': decomposition_info, 'confidence': 0.9 if decomposition_info['should_decompose'] else 0.0}

    def _apply_hyde(self, query: str) -> Dict[str, Any]:
        try:
            hyde_results = self.hyde.retrieve_with_hyde(query, top_k=5)
            if hyde_results:
                hypothetical_answer = self.hyde.generate_hypothetical_answer(query)
                return {'enhanced': True, 'original_query': query, 'enhanced_query': hypothetical_answer, 'technique_used': 'hyde', 'retrieval_results': hyde_results, 'result_count': len(hyde_results), 'confidence': 0.9}
            else:
                hypothetical_answer = self.hyde.generate_hypothetical_answer(query)
                return {'enhanced': True, 'original_query': query, 'enhanced_query': hypothetical_answer, 'technique_used': 'hyde', 'retrieval_results': [], 'result_count': 0, 'confidence': 0.6}
        except Exception as e:
            print(f"HyDE application failed: {e}")
            return {'enhanced': False, 'original_query': query, 'enhanced_query': query, 'technique_used': 'hyde', 'error': str(e), 'confidence': 0.0}

    def _apply_translation(self, query: str) -> Dict[str, Any]:
        translation_info = self.translator.enhance_with_translation(query)
        enhanced_query = translation_info.get('translated_query', query)
        return {'enhanced': translation_info['translated'], 'original_query': query, 'enhanced_query': enhanced_query, 'technique_used': 'translation', 'translation_info': translation_info, 'confidence': 0.6}

    def get_available_techniques(self) -> List[str]:
        techniques = []
        for technique_name, config in self.techniques_config.items():
            if config.get('enabled', False):
                techniques.append(technique_name)
        return techniques

    def get_technique_info(self, technique: str) -> Dict[str, Any]:
        technique_info = {
            'rewriting': {'description': 'Expands queries with synonyms and related terms', 'best_for': 'General queries that need more context', 'priority': self.techniques_config.get('rewriting', {}).get('priority', 1)},
            'step_back': {'description': 'Identifies underlying concepts for ambiguous queries', 'best_for': 'Vague or ambiguous questions', 'priority': self.techniques_config.get('step_back', {}).get('priority', 2)},
            'decomposition': {'description': 'Breaks complex queries into simpler sub-queries', 'best_for': 'Multi-part or complex questions', 'priority': self.techniques_config.get('decomposition', {}).get('priority', 3)},
            'hyde': {'description': 'Uses hypothetical answer embeddings for retrieval', 'best_for': 'Short queries or questions', 'priority': self.techniques_config.get('hyde', {}).get('priority', 4)},
            'translation': {'description': 'Translates queries for better multilingual retrieval', 'best_for': 'Non-English queries', 'priority': self.techniques_config.get('translation', {}).get('priority', 5)}
        }
        return technique_info.get(technique, {})
```