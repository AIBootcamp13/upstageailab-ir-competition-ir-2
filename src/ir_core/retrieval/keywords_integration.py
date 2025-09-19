#!/usr/bin/env python3
"""
Keyword Integration Module for RAG Retrieval

This module provides functionality to integrate curated scientific keywords
into the retrieval system using a hybrid approach that combines:
1. LLM-based dynamic keyword extraction (existing)
2. Curated domain-specific keywords (new)
3. Semantic relevance filtering
"""

import json
import os
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


class CuratedKeywordsIntegrator:
    """Integrates curated scientific keywords into retrieval queries"""

    def __init__(self, keywords_file: str = "data/processed_scientific_keywords.json"):
        self.keywords_file = Path(keywords_file)
        self.keywords_data = self._load_keywords()
        self.embedding_model = None
        self.keyword_embeddings = {}

    def _load_keywords(self) -> Dict:
        """Load processed keywords from JSON file"""
        if not self.keywords_file.exists():
            print(f"Warning: Keywords file {self.keywords_file} not found")
            return {"domains": {}, "all_keywords": []}

        with open(self.keywords_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def initialize_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize sentence transformer for semantic similarity"""
        try:
            self.embedding_model = SentenceTransformer(model_name)
            print(f"Initialized embedding model: {model_name}")
        except Exception as e:
            print(f"Warning: Could not initialize embedding model: {e}")
            self.embedding_model = None

    def get_domain_keywords(self, domain: str, max_keywords: int = 10) -> List[str]:
        """Get keywords for a specific scientific domain"""
        if domain not in self.keywords_data.get("domains", {}):
            return []

        keywords = self.keywords_data["domains"][domain]
        return keywords[:max_keywords]

    def find_relevant_keywords(self, query: str, top_k: int = 5, domain_hint: Optional[str] = None) -> List[str]:
        """
        Find semantically relevant keywords for a query with improved filtering

        Args:
            query: The search query
            top_k: Number of keywords to return
            domain_hint: Optional domain hint to focus keyword selection

        Returns:
            List of relevant keywords
        """
        if not self.embedding_model:
            print("Warning: Embedding model not initialized")
            return []

        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])[0]

            # Get keywords to search through
            if domain_hint and domain_hint in self.keywords_data.get("domains", {}):
                # Use domain-specific keywords if domain is known
                all_keywords = self.keywords_data["domains"][domain_hint]
                print(f"Using domain-specific keywords for '{domain_hint}' ({len(all_keywords)} keywords)")
            else:
                # Use all keywords but with better filtering
                all_keywords = self.keywords_data.get("all_keywords", [])
                print(f"Using all keywords ({len(all_keywords)} keywords)")

            if not all_keywords:
                return []

            # Ensure keyword embeddings are cached
            if not self.keyword_embeddings:
                print("Encoding keyword embeddings...")
                keyword_embeddings = self.embedding_model.encode(all_keywords)
                self.keyword_embeddings = dict(zip(all_keywords, keyword_embeddings))
                print(f"Cached embeddings for {len(all_keywords)} keywords")

            # Calculate similarities
            similarities = []
            for keyword in all_keywords:
                if keyword in self.keyword_embeddings:
                    similarity = np.dot(query_embedding, self.keyword_embeddings[keyword]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(self.keyword_embeddings[keyword])
                    )
                    similarities.append((keyword, similarity))

            # Sort by similarity and return top-k with threshold filtering
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Apply similarity threshold (only include keywords with similarity > 0.4 for domain-specific, > 0.5 for general)
            if domain_hint:
                SIMILARITY_THRESHOLD = 0.4  # More lenient for domain-specific keywords
            else:
                SIMILARITY_THRESHOLD = 0.5  # Stricter for general keywords

            filtered_similarities = [(kw, sim) for kw, sim in similarities if sim >= SIMILARITY_THRESHOLD]

            # If we don't have enough keywords above threshold, relax it slightly
            if len(filtered_similarities) < min(3, top_k):
                if domain_hint:
                    SIMILARITY_THRESHOLD = 0.3  # Relax to 0.3 for domain-specific
                else:
                    SIMILARITY_THRESHOLD = 0.4  # Relax to 0.4 for general
                filtered_similarities = [(kw, sim) for kw, sim in similarities if sim >= SIMILARITY_THRESHOLD]

            relevant_keywords = [kw for kw, sim in filtered_similarities[:top_k]]

            # Additional filtering: remove clearly irrelevant keywords based on query content
            query_lower = query.lower()
            filtered_keywords = []

            for kw in relevant_keywords:
                kw_lower = kw.lower()
                should_exclude = False

                # For plant/tree related queries, exclude animal-specific keywords
                if any(plant_word in query_lower for plant_word in ['나무', '식물', '열매', '꽃', '잎']):
                    animal_keywords = ['냉혈동물', '뇌', '신경', '혈액', '근육', '호르몬', '면역']
                    if any(animal_kw in kw_lower for animal_kw in animal_keywords):
                        should_exclude = True

                # For education queries, exclude biological sample keywords
                if any(edu_word in query_lower for edu_word in ['교육', '학교', '학습', '교과']):
                    sample_keywords = ['표본', '샘플', '시료']
                    if any(sample_kw in kw_lower for sample_kw in sample_keywords):
                        should_exclude = True

                if not should_exclude:
                    filtered_keywords.append(kw)

            relevant_keywords = filtered_keywords[:top_k]

            print(f"Found {len(relevant_keywords)} relevant keywords for query: {query[:50]}...")
            if relevant_keywords:
                print(f"  Keywords: {relevant_keywords}")
            return relevant_keywords

        except Exception as e:
            print(f"Warning: Error finding relevant keywords: {e}")
            # Fallback to domain-based selection
            return self.get_domain_keywords("general_science", top_k)

    def enhance_query_with_keywords(self, query: str, llm_keywords: Optional[List[str]] = None,
                                  use_semantic_matching: bool = True, max_additional: int = 3) -> Tuple[str, List[str]]:
        """
        Enhance a query with relevant curated keywords

        Args:
            query: Original query
            llm_keywords: Keywords already extracted by LLM
            use_semantic_matching: Whether to use semantic similarity for keyword selection
            max_additional: Maximum number of additional keywords to add

        Returns:
            Tuple of (enhanced_query, all_keywords_used)
        """
        llm_keywords = llm_keywords or []

        # Detect query domain for better keyword selection
        detected_domain = self.detect_query_domain(query)
        domain_hint = detected_domain if detected_domain in self.keywords_data.get("domains", {}) else None

        # Get relevant curated keywords
        if use_semantic_matching:
            if self.embedding_model is not None:
                # Only add curated keywords when we can score semantic relevance
                curated_keywords = self.find_relevant_keywords(query, top_k=max_additional, domain_hint=domain_hint)
            else:
                # Safety: do NOT inject generic keywords without semantic filter
                # Returning an empty list here prevents irrelevant terms (e.g., "Joule", "Nmap", "Q 인자")
                # from polluting BM25 queries when embeddings aren't initialized.
                curated_keywords = []
        else:
            # Explicit non-semantic mode: allow domain-specific curated keywords if domain is known
            curated_keywords = self.get_domain_keywords(domain_hint or "", max_additional) if domain_hint else []

        # Filter out keywords that are too similar to existing LLM keywords
        filtered_curated: List[str] = []
        for curated_kw in curated_keywords:
            # Simple check: don't add if too similar to existing keywords
            curated_lower = curated_kw.lower()
            too_similar = any(
                (curated_lower in llm_kw.lower()) or (llm_kw.lower() in curated_lower)
                for llm_kw in llm_keywords
            )
            if not too_similar:
                filtered_curated.append(curated_kw)

        # Combine keywords (always keep LLM-extracted keywords; curated are optional)
        all_keywords = llm_keywords + filtered_curated

        # Create enhanced query
        if filtered_curated:
            enhanced_query = f"{query} {' '.join(filtered_curated)}"
            print(f"Enhanced query with {len(filtered_curated)} curated keywords: {filtered_curated}")
        else:
            enhanced_query = query

        return enhanced_query, all_keywords

    def get_boosting_keywords(self, query: str, domain_hint: Optional[str] = None) -> List[str]:
        """
        Get keywords suitable for Elasticsearch boosting

        Args:
            query: The search query
            domain_hint: Optional domain hint for more targeted boosting

        Returns:
            List of keywords for boosting
        """
        if domain_hint and domain_hint in self.keywords_data.get("domains", {}):
            # Use domain-specific keywords
            domain_keywords = self.get_domain_keywords(domain_hint, 5)
            return domain_keywords
        else:
            # Use semantically relevant keywords
            return self.find_relevant_keywords(query, 5)

    def detect_query_domain(self, query: str) -> Optional[str]:
        """
        Detect the scientific domain of a query based on keyword matching

        Args:
            query: The search query

        Returns:
            Domain name if detected, None otherwise
        """
        query_lower = query.lower()

        # Domain detection patterns
        domain_patterns = {
            "biology": [
                "나무", "식물", "동물", "세포", "유전자", "단백질", "생물", "생태계",
                "진화", "번식", "대사", "해부학", "생물학", "생명", "미생물", "바이러스",
                "면역", "호르몬", "신경", "뇌", "혈액", "호흡", "소화", "생식"
            ],
            "chemistry": [
                "화합물", "분자", "원소", "반응", "산", "염기", "용액", "결합",
                "화학", "원자", "이온", "산화", "환원", "촉매", "용매", "용질",
                "결정", "용융", "증발", "응축", "혼합물", "순물질"
            ],
            "physics": [
                "힘", "에너지", "운동", "속도", "가속도", "질량", "중력", "전자기",
                "파동", "광학", "열", "온도", "압력", "밀도", "전기", "자기",
                "물리", "방사능", "원자력", "양자", "상대성"
            ],
            "earth_science": [
                "지구", "지층", "화산", "지진", "대기", "기후", "날씨", "바다",
                "강", "호수", "산", "사막", "빙하", "지형", "광물", "암석",
                "토양", "식생", "생태", "환경", "오염", "재활용"
            ],
            "mathematics": [
                "수학", "함수", "방정식", "미분", "적분", "확률", "통계", "기하학",
                "대수", "삼각함수", "벡터", "행렬", "그래프", "계산", "측정"
            ],
            "computer_science": [
                "컴퓨터", "프로그래밍", "알고리즘", "데이터", "네트워크", "보안",
                "데이터베이스", "인공지능", "머신러닝", "소프트웨어", "하드웨어"
            ],
            "medicine": [
                "질병", "치료", "진단", "약물", "의학", "병원", "환자", "증상",
                "예방", "백신", "수술", "치료법", "건강", "영양", "면역력"
            ]
        }

        # Count matches for each domain
        domain_scores = {}
        for domain, patterns in domain_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query_lower:
                    score += 1
            if score > 0:
                domain_scores[domain] = score

        # Return the domain with highest score if any matches found
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            print(f"Detected query domain: {best_domain} (score: {domain_scores[best_domain]})")
            return best_domain

        return None


# Global instance for easy access
_curated_keywords_integrator = None

def get_curated_keywords_integrator() -> CuratedKeywordsIntegrator:
    """Get or create the global keywords integrator instance"""
    global _curated_keywords_integrator
    if _curated_keywords_integrator is None:
        _curated_keywords_integrator = CuratedKeywordsIntegrator()
    return _curated_keywords_integrator


def initialize_keywords_integration(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Initialize the keywords integration system"""
    integrator = get_curated_keywords_integrator()
    integrator.initialize_embeddings(model_name)
    print("Curated keywords integration initialized")


def enhance_query_with_curated_keywords(query: str, llm_keywords: Optional[List[str]] = None,
                                       use_semantic: bool = True, max_additional: int = 3) -> Tuple[str, List[str]]:
    """
    Convenience function to enhance query with curated keywords

    Args:
        query: Original query
        llm_keywords: Keywords from LLM extraction
        use_semantic: Whether to use semantic matching
        max_additional: Max additional keywords to add

    Returns:
        Tuple of (enhanced_query, all_keywords)
    """
    integrator = get_curated_keywords_integrator()
    return integrator.enhance_query_with_keywords(query, llm_keywords, use_semantic, max_additional)


if __name__ == "__main__":
    # Test the integration
    integrator = CuratedKeywordsIntegrator()

    # Initialize embeddings
    integrator.initialize_embeddings()

    # Test keyword enhancement
    test_query = "DNA 복제 과정"
    enhanced_query, keywords = integrator.enhance_query_with_keywords(test_query)

    print(f"Original: {test_query}")
    print(f"Enhanced: {enhanced_query}")
    print(f"Keywords: {keywords}")