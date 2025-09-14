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
        ambiguous_indicators = [
            '무엇', '어떻게', '왜', '어디', '언제',  # Korean question words
            'what', 'how', 'why', 'where', 'when',  # English question words
            '의미', '가치', '중요', '차이', '비교',  # Abstract concepts
            'meaning', 'value', 'important', 'difference', 'compare'
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in ambiguous_indicators)