# src/ir_core/query_enhancement/translator.py

from typing import List, Dict, Any, Optional
import openai
from ..config import settings
from ..retrieval.core import hybrid_retrieve
from .llm_client import LLMClient, create_llm_client, detect_client_type

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    print("Warning: googletrans not available. Using basic language detection fallback.")
    Translator = None


class QueryTranslator:
    """
    Query Translation for multilingual search enhancement.

    This technique translates queries to English for better retrieval,
    then optionally searches in both languages and merges results.
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
        Initialize the Query Translator.

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

        # Initialize translator if available
        if GOOGLETRANS_AVAILABLE and Translator:
            self.translator = Translator()
        else:
            self.translator = None

        # Get configuration settings
        self.fallback_on_error = getattr(settings, 'query_enhancement', {}).get('translation', {}).get('fallback_on_error', True)
        self.bilingual_search = getattr(settings, 'query_enhancement', {}).get('translation', {}).get('bilingual_search', True)

    def translate_to_english(self, query: str) -> str:
        """
        Translate query to English.

        For now, return the original query to avoid async issues.
        Translation can be implemented later with a synchronous approach.

        Args:
            query: Query in any language

        Returns:
            Query (currently returns original, translation disabled)
        """
        # Temporarily disable translation to avoid async issues
        # TODO: Implement synchronous translation
        return query

    def detect_language(self, query: str) -> str:
        """
        Detect the language of the query using a simple heuristic approach.

        Args:
            query: Query text

        Returns:
            Language code (e.g., 'en', 'ko', 'ja')
        """
        # Simple language detection based on character sets
        # This is a fallback when googletrans async issues occur

        # Korean characters (Hangul)
        korean_chars = sum(1 for char in query if '\uac00' <= char <= '\ud7a3')

        # Chinese characters
        chinese_chars = sum(1 for char in query if '\u4e00' <= char <= '\u9fff')

        # Japanese characters (Hiragana, Katakana, Kanji)
        japanese_chars = sum(1 for char in query if ('\u3040' <= char <= '\u309f') or ('\u30a0' <= char <= '\u30ff') or ('\u4e00' <= char <= '\u9fff'))

        total_chars = len(query.replace(' ', ''))

        if total_chars == 0:
            return 'unknown'

        # Determine language based on character ratios
        korean_ratio = korean_chars / total_chars
        chinese_ratio = chinese_chars / total_chars
        japanese_ratio = japanese_chars / total_chars

        if korean_ratio > 0.1:
            return 'ko'
        elif chinese_ratio > 0.1:
            return 'zh'
        elif japanese_ratio > 0.1:
            return 'ja'
        else:
            return 'en'  # Default to English

    def bilingual_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search in both original language and English, then merge results.

        Args:
            query: Original query
            top_k: Number of top results to return

        Returns:
            Merged search results
        """
        # Get results for original query
        original_results = hybrid_retrieve(query=query, rerank_k=top_k)

        # Translate and get results for English query
        english_query = self.translate_to_english(query)
        english_results = hybrid_retrieve(query=english_query, rerank_k=top_k)

        # Format results consistently
        formatted_original = self._format_hybrid_results(original_results, "original")
        formatted_english = self._format_hybrid_results(english_results, "translated")

        # Merge and deduplicate results
        combined_results = formatted_original + formatted_english
        merged_results = self._merge_results(combined_results, top_k)

        return merged_results

    def _format_hybrid_results(self, hybrid_results: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """
        Format hybrid retrieval results to consistent structure.

        Args:
            hybrid_results: Results from hybrid_retrieve
            source: Source identifier ("original" or "translated")

        Returns:
            Formatted results
        """
        formatted_results = []

        for result in hybrid_results:
            hit = result.get("hit", {})
            source_doc = hit.get("_source", {})
            doc_id = source_doc.get("docid") or hit.get("_id")
            content = source_doc.get("content", "No content available.")
            score = result.get("score", 0.0)

            formatted_results.append({
                "id": doc_id,
                "content": content,
                "score": float(score) if score is not None else 0.0,
                "title": source_doc.get("title", ""),
                "source": source,
                "cosine": result.get("cosine", 0.0)
            })

        return formatted_results

    def _merge_results(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Merge results from multiple sources, removing duplicates and boosting cross-lingual matches.

        Args:
            results: Combined results from original and translated queries
            top_k: Number of top results to return

        Returns:
            Merged and deduplicated results
        """
        seen_ids = {}
        merged_results = []

        for result in results:
            doc_id = result.get("id")

            if doc_id in seen_ids:
                # Document appears in both original and translated results
                # Boost the score slightly and mark as cross-lingual match
                existing_result = seen_ids[doc_id]
                existing_result["score"] *= 1.1  # 10% boost for cross-lingual matches
                existing_result["cross_lingual"] = True
                existing_result["sources"] = list(set(existing_result.get("sources", [existing_result["source"]]) + [result["source"]]))
            else:
                # New document
                result_copy = result.copy()
                result_copy["cross_lingual"] = False
                result_copy["sources"] = [result["source"]]
                seen_ids[doc_id] = result_copy
                merged_results.append(result_copy)

        # Sort by score and return top_k
        sorted_results = sorted(merged_results, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_results[:top_k]

    def should_translate(self, query: str) -> bool:
        """
        Determine if query should be translated.

        Args:
            query: Query to analyze

        Returns:
            True if translation is recommended
        """
        if not self.translator:
            return False

        detected_lang = self.detect_language(query)

        # Translate if not already in English
        return detected_lang not in ['en', 'unknown']

    def enhance_with_translation(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Enhanced method that handles translation and bilingual search.

        Args:
            query: Query to potentially translate
            top_k: Number of results to return

        Returns:
            Dictionary with translation info and results
        """
        detected_lang = self.detect_language(query)
        translated_query = self.translate_to_english(query)

        if self.should_translate(query) and self.bilingual_search:
            # Use bilingual search
            results = self.bilingual_search(query, top_k)
            return {
                'translated': True,
                'original_query': query,
                'translated_query': translated_query,
                'detected_language': detected_lang,
                'bilingual_search': True,
                'results': results,
                'result_count': len(results)
            }
        elif self.should_translate(query):
            # Use only translated query
            translated_results = hybrid_retrieve(query=translated_query, rerank_k=top_k)
            results = self._format_hybrid_results(translated_results, "translated")
            return {
                'translated': True,
                'original_query': query,
                'translated_query': translated_query,
                'detected_language': detected_lang,
                'bilingual_search': False,
                'results': results,
                'result_count': len(results)
            }
        else:
            # No translation needed
            return {
                'translated': False,
                'original_query': query,
                'translated_query': query,
                'detected_language': detected_lang,
                'bilingual_search': False,
                'results': [],
                'result_count': 0
            }