# src/ir_core/query_enhancement/decomposer.py

from typing import List, Dict, Optional, Any
import openai
from ..config import settings
from .llm_client import LLMClient, create_llm_client, detect_client_type


class QueryDecomposer:
    """
    Query Decomposition for complex multi-part questions.

    This technique breaks complex queries into simpler, independent sub-queries
    that can be answered separately, then aggregates the results.
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
        Initialize the Query Decomposer.

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

    def decompose_query(self, original_query: str) -> List[str]:
        """
        Break complex query into simpler sub-queries.

        Args:
            original_query: The original complex query

        Returns:
            List of simpler sub-queries
        """
        prompt = f"""
        Break this complex query into 2-4 simpler, independent sub-queries:

        Original query: {original_query}

        Each sub-query should:
        - Be answerable independently
        - Focus on a specific aspect of the original question
        - Be clear and specific

        Provide each sub-query on a new line.
        Do not include numbers or bullets.
        """

        try:
            result = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            if not result['success'] or not result['content']:
                return [original_query]  # Fallback to original

            content = result['content']
            sub_queries = content.strip().split('\n')

            # Clean up sub-queries (remove empty lines, strip whitespace)
            cleaned_queries = []
            for query in sub_queries:
                query = query.strip()
                if query and not query.isdigit():  # Skip pure numbers
                    # Remove leading numbers/bullets if present
                    query = self._clean_query_formatting(query)
                    if query:
                        cleaned_queries.append(query)

            # Ensure we have at least the original query
            if not cleaned_queries:
                return [original_query]

            return cleaned_queries[:4]  # Limit to 4 sub-queries max

        except Exception as e:
            print(f"Query decomposition failed: {e}")
            return [original_query]

    def _clean_query_formatting(self, query: str) -> str:
        """
        Clean up formatting artifacts from the decomposed queries.

        Args:
            query: Raw query string

        Returns:
            Cleaned query string
        """
        import re

        # Remove leading numbers and bullets
        query = re.sub(r'^\d+\.?\s*', '', query)
        query = re.sub(r'^[-•*]\s*', '', query)

        return query.strip()

    def aggregate_results(self, sub_query_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Aggregate and deduplicate results from multiple sub-queries.

        Args:
            sub_query_results: List of result lists from each sub-query

        Returns:
            Aggregated and deduplicated results
        """
        all_results = []
        seen_ids = set()

        for results in sub_query_results:
            for result in results:
                # Create a unique identifier for the document
                doc_id = self._get_document_id(result)

                if doc_id not in seen_ids:
                    # Add relevance score boost for documents that appear in multiple sub-queries
                    result_copy = result.copy()
                    result_copy['_sub_query_count'] = 1
                    all_results.append(result_copy)
                    seen_ids.add(doc_id)
                else:
                    # Boost score for documents that appear in multiple sub-queries
                    for existing_result in all_results:
                        if self._get_document_id(existing_result) == doc_id:
                            existing_result['_sub_query_count'] = existing_result.get('_sub_query_count', 1) + 1
                            # Boost the score slightly
                            if 'score' in existing_result:
                                existing_result['score'] *= 1.1  # 10% boost
                            break

        # Sort by score (with boost applied)
        return sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)

    def _get_document_id(self, result: Dict[str, Any]) -> str:
        """
        Extract a unique document identifier from a result.

        Args:
            result: Search result dictionary

        Returns:
            Unique document identifier
        """
        # Try different possible ID fields
        for id_field in ['document_id', 'id', '_id', 'doc_id']:
            if id_field in result:
                return str(result[id_field])

        # Fallback: create ID from title/content hash
        content = result.get('title', '') + result.get('content', '')[:100]
        if content:
            import hashlib
            return hashlib.md5(content.encode()).hexdigest()

        # Last resort: use object hash
        return str(hash(str(result)))

    def should_decompose(self, query: str) -> bool:
        """
        Determine if a query should be decomposed.

        Args:
            query: Query to analyze

        Returns:
            True if query should be decomposed
        """
        # Simple heuristics for complex queries
        complexity_indicators = [
            ' and ', ' or ', ' vs ', ' vs. ', ' compared ', ' difference ', 'differences', 'compare', 'explain', 'describe', 'between', ' versus ',
            ' 비교 ', ' 차이 ', ' 대 ', ' versus ', ' vs ',
            ' how ', ' why ', ' what ', ' when ', ' where ',
            ' 어떻게 ', ' 왜 ', ' 무엇 ', ' 언제 ', ' 어디 ',
            ' 그리고 ', ' 또는 ', ' 대체 ', ' 대신 '
        ]

        query_lower = query.lower()

        # Count complexity indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)

        # Check query length
        word_count = len(query.split())

        # Decompose if query is long or has multiple complexity indicators
        return word_count > 15 or indicator_count >= 2

    def enhance_complex_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced method that decides whether to decompose and handles the process.

        Args:
            query: Query to potentially decompose

        Returns:
            Dictionary with decomposition info and sub-queries
        """
        if self.should_decompose(query):
            sub_queries = self.decompose_query(query)
            return {
                'should_decompose': True,
                'original_query': query,
                'sub_queries': sub_queries,
                'sub_query_count': len(sub_queries)
            }
        else:
            return {
                'should_decompose': False,
                'original_query': query,
                'sub_queries': [query],
                'sub_query_count': 1
            }