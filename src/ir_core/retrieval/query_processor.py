#!/usr/bin/env python3
"""
Query Processor Module

Handles all query enhancement and preprocessing logic for the retrieval pipeline.
This            if hasattr(generator, 'client') and hasattr(generator, 'model_name'):
                response = generator.client.chat.completions.create(  # type: ignore
                    model=generator.model_name,  # type: ignore
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
                keywords_str = response.choices[0].message.content or ""
                keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

                # Cache the result
                if self.redis_client and keywords:
                    try:
                        self.redis_client.setex(cache_key, 3600, json.dumps(keywords))  # Cache for 1 hour
                    except:
                        pass  # Ignore caching errors

                return keywordsLLM-based keyword extraction, curated keywords integration,
and profiling-based query expansion.
"""

import redis
from typing import List, Dict, Any, Optional
import json
from ..config import settings
from .insights_manager import (
    insights_manager,
    get_query_expansion_terms
)
from .keywords_integration import get_curated_keywords_integrator, enhance_query_with_curated_keywords


class QueryProcessor:
    """Handles query enhancement and preprocessing"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client

    def process(self, query: str) -> Dict[str, Any]:
        """
        Process and enhance the input query

        Args:
            query: Original search query

        Returns:
            Dict containing enhanced query and metadata
        """
        enhanced_query = query
        query_expansion_terms = []
        all_keywords = []

        # Apply profiling-based query expansion
        enhanced_query, expansion_terms = self._apply_profiling_expansion(query)
        query_expansion_terms = expansion_terms

        # Extract dynamic keywords using LLM
        dynamic_keywords = self._extract_keywords_from_query(enhanced_query)

        # Enhance with curated scientific keywords
        enhanced_query, all_keywords = self._apply_curated_keywords(
            enhanced_query, dynamic_keywords
        )

        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "dynamic_keywords": dynamic_keywords,
            "all_keywords": all_keywords,
            "expansion_terms": query_expansion_terms
        }

    def _apply_profiling_expansion(self, query: str) -> tuple[str, List[str]]:
        """Apply profiling-based query expansion"""
        enhanced_query = query
        expansion_terms = []

        insights_config = getattr(settings, 'profiling_insights', {})
        use_query_expansion = insights_config.get('use_query_expansion', True)
        query_expansion_terms_count = insights_config.get('query_expansion_terms', 3)

        if settings.PROFILE_REPORT_DIR and use_query_expansion:
            try:
                insights = insights_manager.get_insights()
                vocab_sources = list(insights.get('vocab_overlap', {}).keys())

                if vocab_sources:
                    representative_src = vocab_sources[0]
                    expansion_terms = get_query_expansion_terms(
                        representative_src, top_k=query_expansion_terms_count
                    )

                    # Enhance query with expansion terms if they seem relevant
                    if expansion_terms:
                        query_lower = query.lower()
                        relevant_terms = [
                            term for term in expansion_terms
                            if term.lower() in query_lower or
                               any(word in query_lower for word in term.split())
                        ]

                        if relevant_terms:
                            enhanced_query = f"{query} {' '.join(relevant_terms)}"

            except Exception as e:
                print(f"Warning: Failed to apply query expansion: {e}")

        return enhanced_query, expansion_terms

    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract keywords from query using LLM"""
        try:
            from ..generation import get_generator
            from omegaconf import OmegaConf

            cfg_dict = {
                'pipeline': {
                    'generator_type': getattr(settings, 'GENERATOR_TYPE', 'openai'),
                    'generator_model_name': getattr(settings, 'GENERATOR_MODEL_NAME', 'gpt-4o-mini'),
                },
                'prompts': {
                    'generation_qa': getattr(settings, 'GENERATOR_SYSTEM_MESSAGE_FILE', ''),
                    'persona': getattr(settings, 'GENERATOR_SYSTEM_MESSAGE_FILE', ''),
                }
            }
            cfg = OmegaConf.create(cfg_dict)

            generator = get_generator(cfg)

            prompt = f"""
            Extract the most important and specific keywords from the following user query.
            Focus on nouns, technical terms, and core concepts that would be most relevant for document retrieval.
            Return only the keywords as a comma-separated list, no explanations.

            Query: "{query}"

            Keywords:
            """

            # Use the generator's generate method
            keywords_str = generator.generate(query=prompt, context_docs=[])
            keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
            return keywords
        except Exception as e:
            print(f"Warning: Keyword extraction failed: {e}")
            return [word.strip() for word in query.split() if len(word.strip()) > 1]

    def _apply_curated_keywords(self, query: str, dynamic_keywords: List[str]) -> tuple[str, List[str]]:
        """Apply curated scientific keywords enhancement"""
        try:
            integrator = get_curated_keywords_integrator()
            enhanced_query, all_keywords = integrator.enhance_query_with_keywords(
                query, dynamic_keywords, use_semantic_matching=True, max_additional=3
            )

            combined_keywords = all_keywords if all_keywords else dynamic_keywords

            return enhanced_query, all_keywords

        except Exception as e:
            print(f"Warning: Curated keywords integration failed: {e}")
            return query, dynamic_keywords