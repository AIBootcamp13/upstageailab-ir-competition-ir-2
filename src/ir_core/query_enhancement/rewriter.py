# src/ir_core/query_enhancement/rewriter.py

from typing import Optional
import openai
from ..config import settings
from .llm_client import LLMClient, create_llm_client, detect_client_type


class QueryRewriter:
    """
    Query Rewriting and Expansion using OpenAI.

    This class enhances queries by expanding them with relevant synonyms,
    related terms, and making them more specific for better document retrieval.
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
        Initialize the Query Rewriter.

        Args:
            llm_client: Pre-configured LLM client. If None, creates one based on model_name.
            openai_client: Pre-configured OpenAI client (for backward compatibility).
            model_name: OpenAI model to use. Defaults to settings value.
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
                # Ensure model_name is not None for type checker
                assert model_name is not None
                client_type = detect_client_type(model_name)
                self.llm_client = create_llm_client(client_type, model_name=model_name)
        else:
            self.llm_client = llm_client

        self.model_name = model_name or getattr(settings, 'query_enhancement', {}).get('openai_model', 'gpt-3.5-turbo')
        self.max_tokens = max_tokens or getattr(settings, 'query_enhancement', {}).get('max_tokens', 500)
        self.temperature = temperature or getattr(settings, 'query_enhancement', {}).get('temperature', 0.3)

    def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite and expand the query for better retrieval.

        Args:
            original_query: The original user query

        Returns:
            Enhanced query string optimized for retrieval
        """
        # Detect if the query is in Korean
        is_korean = any('\uac00' <= char <= '\ud7a3' for char in original_query)

        if is_korean:
            prompt = f"""
            이 쿼리를 효과적인 검색 쿼리로 변환하세요:
            1. 핵심 개념과 주요 용어를 추출
            2. 관련 동의어와 기술 용어 추가
            3. 문서 검색에 더 구체적이고 포괄적으로 만들기
            4. 주제의 일반적이고 구체적인 측면 모두 포함

            중요한 지침:
            - 입력이 한국어이면 출력도 한국어로 유지
            - 의미를 100% 유지하면서 검색 최적화
            - 불필요한 말줄임표나 이모지 제거
            - 더 자연스러운 표현으로 변경 (의미는 동일하게)

            원본 쿼리: {original_query}

            재작성된 쿼리만 제공하세요.
            """
        else:
            prompt = f"""
            Transform this query into an effective search query by:
            1. Extracting core concepts and key terms
            2. Adding relevant synonyms and related technical terms
            3. Making it more specific and comprehensive for document retrieval
            4. Including both general and specific aspects of the topic

            Original query: {original_query}

            Provide only the rewritten query, no explanation.
            """

        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            if not response['success']:
                print(f"Query rewriting failed: {response.get('error', 'Unknown error')}")
                return original_query

            content = response.get('content')
            if not content:
                return original_query

            rewritten_query = content.strip()

            # Fallback if response is empty or too short
            if not rewritten_query or len(rewritten_query) < len(original_query) * 0.5:
                return original_query

            return rewritten_query

        except Exception as e:
            print(f"Query rewriting failed: {e}")
            return original_query  # Return original query on failure

    def expand_query(self, query: str, expansion_factor: int = 2) -> str:
        """
        Expand query with additional related terms.

        Args:
            query: Original query
            expansion_factor: How much to expand (1-3, where 3 is most expansive)

        Returns:
            Expanded query with additional terms
        """
        expansion_levels = {
            1: "Add 2-3 closely related terms",
            2: "Add 4-6 related terms and synonyms",
            3: "Add comprehensive related terms, synonyms, and technical variations"
        }

        level_desc = expansion_levels.get(expansion_factor, expansion_levels[2])

        prompt = f"""
        Expand this query by adding relevant terms for better search results:

        Original query: {query}

        {level_desc} to improve document retrieval.
        Maintain the original query's intent and meaning.

        Provide only the expanded query, no explanation.
        """

        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            if not response['success']:
                print(f"Query expansion failed: {response.get('error', 'Unknown error')}")
                return query

            content = response.get('content')
            if not content:
                return query

            expanded_query = content.strip()

            if not expanded_query or len(expanded_query) < len(query):
                return query

            return expanded_query

        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query