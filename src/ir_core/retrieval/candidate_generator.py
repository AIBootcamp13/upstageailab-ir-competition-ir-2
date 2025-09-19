#!/usr/bin/env python3
"""
Candidate Generator Module

Handles the generation of candidate documents from different retrieval sources.
Provides a unified interface for sparse and dense retrieval methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..infra import get_es
from ..config import settings
from .boosting import load_keywords_per_src, build_boosted_query
from .preprocessing import filter_stopwords
from .insights_manager import get_domain_cluster


class CandidateGenerator(ABC):
    """Abstract base class for candidate generators"""

    @abstractmethod
    def retrieve(self, query: str, size: int = 10, query_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve candidate documents"""
        pass


class BM25Retriever(CandidateGenerator):
    """BM25-based sparse retrieval"""

    def __init__(self, index_name: Optional[str] = None):
        self.index_name = index_name or settings.INDEX_NAME
        self.es_client = get_es()

    def retrieve(self, query: str, size: int = 10, query_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using BM25"""
        return self._sparse_retrieve(query, size, query_info)

    def _sparse_retrieve(self, query: str, size: int, query_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Internal sparse retrieval implementation"""
        # Optional query preprocessing
        processed_query = query
        if settings.USE_STOPWORD_FILTERING:
            processed_query = filter_stopwords(query)

        # Create flexible boolean query
        q = self._build_flexible_match_query(processed_query, size, query_info)

        # Optional boosting using profiling artifacts
        if settings.USE_SRC_BOOSTS and settings.PROFILE_REPORT_DIR:
            kw = load_keywords_per_src(settings.PROFILE_REPORT_DIR)
            if kw:
                boosting_result = build_boosted_query(processed_query, size, kw)
                boosting_clauses = boosting_result.get("boosting_clauses", [])

                # Integrate boosting clauses into the existing flexible query
                if boosting_clauses and "query" in q and "bool" in q["query"]:
                    if "should" not in q["query"]["bool"]:
                        q["query"]["bool"]["should"] = []
                    q["query"]["bool"]["should"].extend(boosting_clauses)

        res = self.es_client.search(index=self.index_name, body=q)
        return res.get("hits", {}).get("hits", [])

    def _build_flexible_match_query(self, query: str, size: int, query_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build enhanced multi-field boolean query"""
        # Use pre-processed keywords if available, otherwise extract them
        if query_info and "all_keywords" in query_info:
            combined_keywords = query_info["all_keywords"]
            enhanced_query = query_info.get("enhanced_query", query)
        else:
            # Extract dynamic keywords from the query using LLM
            dynamic_keywords = self._extract_keywords_from_query(query)

            # Enhance with curated scientific keywords
            try:
                from .keywords_integration import get_curated_keywords_integrator
                integrator = get_curated_keywords_integrator()
                enhanced_query, all_keywords = integrator.enhance_query_with_keywords(
                    query, dynamic_keywords, use_semantic_matching=True, max_additional=3
                )

                # Use all keywords (LLM + curated) for the keywords field
                combined_keywords = all_keywords if all_keywords else dynamic_keywords

            except Exception as e:
                print(f"Warning: Curated keywords integration failed: {e}")
                enhanced_query = query
                combined_keywords = dynamic_keywords

        # Build multi-field query with appropriate boosts
        bool_query = {
            "should": [
                # High boost for keywords field
                {"match": {"keywords": {"query": ' '.join(combined_keywords), "boost": 2.0}}},
                # High boost for hypothetical questions
                {"match": {"hypothetical_questions": {"query": enhanced_query, "boost": 2.0}}},
                # Medium boost for summary
                {"match": {"summary": {"query": enhanced_query, "boost": 2.0}}},
                # Highest boost for full content
                {"match": {"content": {"query": enhanced_query, "boost": 3.0}}}
            ],
            "minimum_should_match": 1
        }

        # Add phrase matching for better precision on Korean queries
        if any('\uac00' <= char <= '\ud7a3' for char in enhanced_query):
            bool_query["should"].append({
                "match_phrase": {
                    "content": {
                        "query": enhanced_query,
                        "slop": 2,
                        "boost": 1.5
                    }
                }
            })

        return {
            "query": {
                "bool": bool_query
            },
            "size": size
        }

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

            if hasattr(generator, 'client') and hasattr(generator, 'model_name'):
                response = generator.client.chat.completions.create(  # type: ignore
                    model=generator.model_name,  # type: ignore
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
                keywords_str = response.choices[0].message.content or ""
                keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                print(f"Extracted keywords: {keywords}")
                return keywords
            else:
                print("Warning: No LLM client available, falling back to simple keyword extraction")
                return [word.strip() for word in query.split() if len(word.strip()) > 1]

        except Exception as e:
            print(f"Warning: Keyword extraction failed: {e}")
            return [word.strip() for word in query.split() if len(word.strip()) > 1]


class DenseRetriever(CandidateGenerator):
    """Dense retrieval using vector similarity"""

    def __init__(self, index_name: Optional[str] = None):
        self.index_name = index_name or settings.INDEX_NAME
        self.es_client = get_es()

    def retrieve(self, query: Any, size: int = 10, query_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using dense vector similarity"""
        return self._dense_retrieve(query, size)

    def _dense_retrieve(self, query_emb, size: int) -> List[Dict[str, Any]]:
        """Internal dense retrieval implementation"""
        import numpy as np

        # Validate query embedding
        if np.isnan(query_emb).any() or np.isinf(query_emb).any():
            print("⚠️ Invalid query embedding detected, cannot perform dense retrieval")
            return []

        q = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                        "params": {"query_vector": query_emb.tolist()},
                    },
                }
            },
        }

        res = self.es_client.search(index=self.index_name, body=q)
        return res.get("hits", {}).get("hits", [])