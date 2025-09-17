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
        # Detect if the query is in Korean
        is_korean = any('\uac00' <= char <= '\ud7a3' for char in query)

        if is_korean:
            prompt = f"""
            이 질문에 대한 가상의 문서를 작성하세요. 마치 위키피디아나 과학 교과서의 한 단락처럼 자연스럽게 작성하세요:

            질문: {query}

            다음 형식으로 작성하세요:
            - 자연스러운 문장으로 답변
            - 관련 사실과 예시 포함
            - 전문 용어 적절히 사용
            - 2-3개의 문단으로 구성
            - 검색에 유용한 구체적인 내용 포함

            가상의 참고 문서 내용:
            """
        else:
            prompt = f"""
            Write a hypothetical document passage that would answer this question. Write it as if it were a paragraph from Wikipedia or a science textbook:

            Question: {query}

            Write in this format:
            - Natural, flowing sentences
            - Include relevant facts and examples
            - Use appropriate technical terms
            - 2-3 paragraphs
            - Specific content useful for search

            Hypothetical reference document content:
            """

        try:
            result = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,  # Add required model parameter
                max_tokens=self.max_tokens,  # Use configured max_tokens directly
                temperature=self.temperature
            )

            if not result['success']:
                print(f"HyDE LLM call failed: {result.get('error', 'Unknown error')}")
                return query  # Fallback to original query

            content = result.get('content', '').strip()
            if not content:
                print("HyDE generated empty content")
                return query  # Fallback to original query

            # Validate content quality - must be substantially longer than original query
            if len(content) < len(query) * 2:
                print(f"HyDE generated insufficient content: {len(content)} chars (original: {len(query)} chars)")
                return query  # Fallback to original query

            # Check if content is just repeating the query
            if content.lower() == query.lower() or query.lower() in content.lower() and len(content) < 100:
                print(f"HyDE generated trivial content: '{content}'")
                return query  # Fallback to original query

            return content

        except Exception as e:
            print(f"Hypothetical answer generation failed: {e}")
            return query

    def get_hyde_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Get embedding of hypothetical answer.

        Args:
            query: Original query

        Returns:
            Embedding vector of the hypothetical answer, or None if failed
        """
        try:
            hypothetical_answer = self.generate_hypothetical_answer(query)

            # Additional validation of hypothetical answer
            if not hypothetical_answer or len(hypothetical_answer.strip()) < 10:
                print(f"HyDE generated insufficient content: '{hypothetical_answer}'")
                return None

            # Check if content is just the query repeated
            if hypothetical_answer.lower().strip() == query.lower().strip():
                print(f"HyDE generated identical content to query: '{hypothetical_answer}'")
                return None

            embedding = encode_query(hypothetical_answer)

            # Validate embedding
            if embedding is None or len(embedding) == 0:
                print(f"HyDE embedding encoding failed for content: '{hypothetical_answer[:50]}...'")
                return None

            return embedding

        except Exception as e:
            print(f"HyDE embedding generation failed: {e}")
            return None

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

            # Validate embedding
            if hyde_embedding is None or len(hyde_embedding) == 0:
                print(f"HyDE embedding is invalid for query: {query[:50]}...")
                return []

            # Ensure embedding is proper shape and type
            if not isinstance(hyde_embedding, np.ndarray):
                print(f"HyDE embedding is not numpy array for query: {query[:50]}...")
                return []

            if hyde_embedding.ndim != 1:
                print(f"HyDE embedding has wrong dimensions {hyde_embedding.shape} for query: {query[:50]}...")
                return []

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
            # Fallback to standard dense retrieval
            try:
                print(f"Falling back to standard dense retrieval for query: {query[:50]}...")
                from ..embeddings.core import encode_query as core_encode_query
                from ..retrieval.core import dense_retrieve as core_dense_retrieve

                query_embedding = core_encode_query(query)
                es_results = core_dense_retrieve(query_embedding, size=top_k)

                # Format results
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
                        "source": "dense_fallback"
                    })

                return formatted_results

            except Exception as fallback_e:
                print(f"Fallback dense retrieval also failed: {fallback_e}")
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
            try:
                results = self.retrieve_with_hyde(query, top_k)
                hypothetical_answer = self.generate_hypothetical_answer(query)

                return {
                    'used_hyde': True,
                    'original_query': query,
                    'hypothetical_answer': hypothetical_answer,
                    'results': results,
                    'result_count': len(results)
                }
            except Exception as e:
                print(f"HyDE enhancement failed: {e}")
                # Fallback: generate hypothetical answer without retrieval
                try:
                    hypothetical_answer = self.generate_hypothetical_answer(query)
                    return {
                        'used_hyde': True,
                        'original_query': query,
                        'hypothetical_answer': hypothetical_answer,
                        'results': [],
                        'result_count': 0,
                        'error': str(e)
                    }
                except Exception as e2:
                    print(f"HyDE hypothetical answer generation also failed: {e2}")
                    return {
                        'used_hyde': False,
                        'original_query': query,
                        'hypothetical_answer': None,
                        'results': [],
                        'result_count': 0,
                        'error': f"{e}, {e2}"
                    }
        else:
            return {
                'used_hyde': False,
                'original_query': query,
                'hypothetical_answer': None,
                'results': [],
                'result_count': 0
            }