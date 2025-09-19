# src/ir_core/query_enhancement/step_back.py

from typing import Optional
import openai
from ..config import settings
from .llm_client import LLMClient, create_llm_client, detect_client_type


class StepBackPrompting:
    """
    Step-Back Prompting for query enhancement.

    This technique takes a step back from the original query to identify
    the underlying concept, then creates a more abstract search query
    that can find relevant documents even when the original query is vague.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        openai_client: Optional[openai.OpenAI] = None,  # For backward compatibility
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the Step-Back Prompting enhancer.

        Args:
            llm_client: Pre-configured LLM client. If None, creates one based on model_name.
            openai_client: Pre-configured OpenAI client (for backward compatibility).
            model_name: Model to use. Defaults to settings value.
            max_tokens: Maximum tokens for response. Defaults to settings value.
            temperature: Temperature for generation. Defaults to settings value.
        """
        # Handle backward compatibility
        if llm_client is None:
            if openai_client:
                from .llm_client import OpenAIClient
                self.llm_client = OpenAIClient(openai_client)
            else:
                # Auto-detect based on model name
                if model_name is None:
                    model_name = getattr(settings, 'query_enhancement', {}).get('openai_model', 'gpt-3.5-turbo')
                # Ensure model_name is not None for detect_client_type
                if model_name is None:
                    model_name = 'gpt-3.5-turbo'
                client_type = detect_client_type(model_name)
                self.llm_client = create_llm_client(client_type, model_name=model_name)
        else:
            self.llm_client = llm_client

        self.model_name = model_name or getattr(settings, 'query_enhancement', {}).get('openai_model', 'gpt-3.5-turbo')
        self.max_tokens = max_tokens or getattr(settings, 'query_enhancement', {}).get('max_tokens', 500)
        self.temperature = temperature or getattr(settings, 'query_enhancement', {}).get('temperature', 0.3)

    def step_back(self, original_query: str) -> str:
        """
        Take a step back to find the underlying concept and create a search query.

        Args:
            original_query: The original user query

        Returns:
            Abstracted search query based on underlying concepts
        """
        # First, identify the underlying concept
        abstract_concept = self._identify_abstract_concept(original_query)

        # Then, convert the concept to searchable keywords
        search_query = self._concept_to_search_query(abstract_concept)

        return search_query

    def _identify_abstract_concept(self, query: str) -> str:
        """
        Identify the general, underlying concept being asked.

        Args:
            query: Original query

        Returns:
            Abstract description of the underlying concept
        """
        # Detect if the query is in Korean
        is_korean = any('\uac00' <= char <= '\ud7a3' for char in query)

        if is_korean:
            prompt = f"""
            사용자가 물은 내용: "{query}"

            이 질문의 일반적이고 근본적인 개념은 무엇인가요?
            사용자가 실제로 찾고 있는 정보에 대한 명확하고 추상적인 설명을 제공하세요.

            근본적인 주제나 원리에 초점을 맞추고, 구체적인 세부사항은 무시하세요.
            설명 없이 추상적 개념만 응답하세요.
            """
        else:
            prompt = f"""
            The user asked: "{query}"

            What is the general, underlying concept being asked?
            Provide a clear, abstract description of what information they're really seeking.

            Focus on the fundamental topic or principle, not the specific details.
            Respond with just the abstracted concept, no explanation.
            """

        try:
            result = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            if not result['success'] or not result['content']:
                return query  # Fallback to original query

            return result['content'].strip()

        except Exception as e:
            print(f"Abstract concept identification failed: {e}")
            return query

    def _concept_to_search_query(self, concept: str) -> str:
        """
        Convert abstract concept to specific search keywords.

        Args:
            concept: Abstract concept description

        Returns:
            Searchable query with specific keywords
        """
        # Detect if the original concept contains Korean characters
        is_korean = any('\uac00' <= char <= '\ud7a3' for char in concept)

        if is_korean:
            prompt = f"""
            이 추상적 개념을 특정 검색 키워드로 변환하세요:
            "{concept}"

            문서에서 나타날 가능성이 높은 용어에 초점을 맞춘 키워드를 쉼표로 구분하여 제공하세요.
            이 개념과 관련된 일반적이고 구체적인 용어를 모두 포함하세요.
            """
        else:
            prompt = f"""
            Convert this abstract concept into specific search keywords:
            "{concept}"

            Provide keywords separated by commas, focused on terms likely to appear in documents.
            Include both general and specific terms related to this concept.
            """

        try:
            result = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            if not result['success'] or not result['content']:
                return concept  # Fallback to concept

            return result['content'].strip()

        except Exception as e:
            print(f"Concept to keywords conversion failed: {e}")
            return concept

    def enhance_ambiguous_query(self, query: str) -> str:
        """
        Enhanced method specifically for ambiguous queries.

        Args:
            query: Potentially ambiguous query

        Returns:
            Enhanced query with better search terms
        """
        # Check if query seems ambiguous
        if self._is_ambiguous(query):
            return self.step_back(query)
        else:
            return query  # Return original if not ambiguous

    def _is_ambiguous(self, query: str) -> bool:
        """
        Simple heuristic to detect potentially ambiguous queries.

        Args:
            query: Query to analyze

        Returns:
            True if query appears ambiguous
        """
        # Check for simple math expressions first
        if self._is_simple_math(query):
            return False

        ambiguous_indicators = [
            '무엇', '어떻게', '왜', '어디', '언제',  # Korean question words
            'what', 'how', 'why', 'where', 'when',  # English question words
            '의미', '가치', '중요', '차이', '비교',  # Abstract concepts
            'meaning', 'value', 'important', 'difference', 'compare'
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in ambiguous_indicators)

    def _is_simple_math(self, query: str) -> bool:
        """
        Check if query is a simple math expression.

        Args:
            query: Query to analyze

        Returns:
            True if query appears to be simple math
        """
        import re

        # Pattern for simple math: "What is X op Y?" where op is +, -, *, /
        pattern = r"^What is \d+\s*[\+\-\*\/]\s*\d+\?$"
        return bool(re.match(pattern, query, re.IGNORECASE))