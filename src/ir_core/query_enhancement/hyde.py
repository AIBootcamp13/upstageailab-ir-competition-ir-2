# src/ir_core/query_enhancement/hyde.py

from typing import List, Dict, Any, Optional
import openai
import numpy as np
from ..config import settings
from ..embeddings.core import encode_query
from ..retrieval.core import dense_retrieve
from .llm_client import LLMClient, create_llm_client, detect_client_type


class HyDE:
    """
    Hypothetical Document Embeddings (HyDE) for query enhancement.

    This technique generates a hypothetical answer to the query, then uses
    the embedding of that hypothetical answer to find relevant documents.
    Particularly effective for short, keyword-poor queries.
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
        Initialize the HyDE enhancer.

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

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a detailed hypothetical answer to the query.

        Args:
            query: The original user query

        Returns:
            Hypothetical answer that would appear in a reference document
        """
        prompt = f"""
        Provide a detailed, comprehensive answer to this question as if you were writing a reference document:

        Question: {query}

        Write a detailed answer that would appear in an encyclopedia or textbook.
        Include specific facts, examples, and explanations that would be found in relevant documents.
        Be comprehensive but focused on the core topic.
        """

        try:
            result = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens * 2,  # Allow longer responses for detailed answers
                temperature=self.temperature
            )

            if not result['success'] or not result['content']:
                return query  # Fallback to original query

            return result['content'].strip()

        except Exception as e:
            print(f"Hypothetical answer generation failed: {e}")
            return query

    def get_hyde_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding of hypothetical answer.

        Args:
            query: Original query

        Returns:
            Embedding vector of the hypothetical answer
        """
        hypothetical_answer = self.generate_hypothetical_answer(query)
        return encode_query(hypothetical_answer)

    def retrieve_with_hyde(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using HyDE embedding.

        Args:
            query: Original query
            top_k: Number of top results to return

        Returns:
            List of retrieved documents with metadata
        """
        try:
            # Generate hypothetical answer and its embedding
            hyde_embedding = self.get_hyde_embedding(query)

            # Use dense retrieval with the hypothetical answer embedding
            es_results = dense_retrieve(hyde_embedding, size=top_k)

            # Format results to match the expected structure
            formatted_results = []
            for hit in es_results:
                source_doc = hit.get("_source", {})
                doc_id = source_doc.get("docid") or hit.get("_id")
                content = source_doc.get("content", "No content available.")
                score = hit.get("_score", 0.0)

                formatted_results.append({
                    "id": doc_id,
                    "content": content,
                    "score": float(score) if score is not None else 0.0,
                    "title": source_doc.get("title", ""),
                    "source": "hyde_retrieval"
                })

            return formatted_results

        except Exception as e:
            print(f"HyDE retrieval failed: {e}")
            return []

    def should_use_hyde(self, query: str) -> bool:
        """
        Determine if HyDE should be used for this query.

        Args:
            query: Query to analyze

        Returns:
            True if HyDE is recommended
        """
        # Simple heuristics for when HyDE is effective
        word_count = len(query.split())

        # HyDE works well for short queries
        if word_count <= 5:
            return True

        # HyDE works well for queries that are more like questions than keyword searches
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose']
        query_lower = query.lower()

        has_question_word = any(word in query_lower for word in question_words)

        # Also check for Korean question words
        korean_question_words = ['무엇', '어떻게', '왜', '언제', '어디', '누구', '어느', '누구의']
        has_korean_question = any(word in query for word in korean_question_words)

        return has_question_word or has_korean_question or word_count <= 8

    def enhance_with_hyde(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Enhanced method that decides whether to use HyDE and handles the process.

        Args:
            query: Query to potentially enhance with HyDE
            top_k: Number of results to return

        Returns:
            Dictionary with enhancement info and results
        """
        if self.should_use_hyde(query):
            results = self.retrieve_with_hyde(query, top_k)
            hypothetical_answer = self.generate_hypothetical_answer(query)

            return {
                'used_hyde': True,
                'original_query': query,
                'hypothetical_answer': hypothetical_answer,
                'results': results,
                'result_count': len(results)
            }
        else:
            return {
                'used_hyde': False,
                'original_query': query,
                'hypothetical_answer': None,
                'results': [],
                'result_count': 0
            }