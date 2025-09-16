# src/ir_core/analysis/query_components.py

"""
Refactored query analysis components for better modularity.

This module contains smaller, focused classes extracted from the monolithic
QueryAnalyzer to improve code organization and reusability.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

from .constants import (
    DOMAIN_KEYWORDS,
    SCIENTIFIC_TERMS,
    QUERY_TYPE_PATTERNS,
    LANGUAGE_COMPLEXITY_INDICATORS,
    FORMULA_PATTERNS,
    QUERY_LENGTH_NORMALIZATION
)
from .config.config_loader import ConfigLoader

# Load configuration for query analysis settings
_config_loader = ConfigLoader()
_query_config = _config_loader.get('query_analysis', {})


@dataclass
class QueryFeatures:
    """Detailed features extracted from a query."""
    length: int
    word_count: int
    scientific_terms: List[str]
    complexity_score: float
    query_type: str
    domain: List[str]  # Support multiple domains
    has_numbers: bool
    has_formulas: bool
    language_complexity: float


class QueryFeatureExtractor:
    """
    Extracts basic features from queries.
    """

    def extract_features(self, query: str) -> QueryFeatures:
        """
        Extract comprehensive features from a single query.

        Args:
            query: The query string to analyze

        Returns:
            QueryFeatures: Detailed analysis results
        """
        # Basic features
        length = len(query)
        words = query.split()
        word_count = len(words)

        # Scientific term extraction
        scientific_terms = self._extract_scientific_terms(query)

        # Complexity scoring
        complexity_score = self._calculate_complexity(query, scientific_terms)

        # Query type classification
        query_type = self._classify_query_type(query)

        # Domain classification
        domain = self._classify_domain(query)

        # Additional features
        has_numbers = bool(re.search(r'\d', query))
        has_formulas = self._detect_formulas(query)
        language_complexity = self._assess_language_complexity(query)

        return QueryFeatures(
            length=length,
            word_count=word_count,
            scientific_terms=scientific_terms,
            complexity_score=complexity_score,
            query_type=query_type,
            domain=domain,
            has_numbers=has_numbers,
            has_formulas=has_formulas,
            language_complexity=language_complexity
        )

    def _extract_scientific_terms(self, query: str) -> List[str]:
        """
        Extract scientific terms from the query.

        Args:
            query: The query string

        Returns:
            List[str]: List of detected scientific terms
        """
        query_lower = query.lower()
        found_terms = []

        for term in SCIENTIFIC_TERMS:
            if term in query_lower:
                found_terms.append(term)

        return found_terms

    def _calculate_complexity(self, query: str, scientific_terms: List[str]) -> float:
        """
        Calculate query complexity score based on multiple factors.

        Args:
            query: The query string
            scientific_terms: List of scientific terms found in the query

        Returns:
            float: Complexity score between 0 and 1
        """
        # Length-based complexity (0-0.3)
        length_score = min(len(query) / 200, 0.3)

        # Scientific term density (0-0.4)
        words = query.split()
        term_density = len(scientific_terms) / len(words) if words else 0
        term_score = min(term_density * 2, 0.4)

        # Language complexity (0-0.3)
        lang_score = self._assess_language_complexity(query) * 0.3

        return length_score + term_score + lang_score

    def _classify_query_type(self, query: str) -> str:
        """
        Classify the query type based on question words and patterns.

        Args:
            query: The query string

        Returns:
            str: Query type (what, how, why, when, where, calculate, or general)
        """
        query_lower = query.lower()

        # Check for question words
        for qtype, pattern in QUERY_TYPE_PATTERNS.items():
            if pattern.search(query):
                return qtype

        # Check for calculation indicators
        if any(term in query_lower for term in ['=', '계산', '구하라', '값은']):
            return "calculate"

        # Default to general
        return "general"

    def _classify_domain(self, query: str) -> List[str]:
        """
        Classify the query into scientific domains (supports multiple domains).

        Args:
            query: The query string

        Returns:
            List[str]: List of matched domains
        """
        query_lower = query.lower()
        matched_domains = []

        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                matched_domains.append(domain)

        # Remove duplicates and return
        return list(set(matched_domains)) if matched_domains else ["unknown"]

    def _detect_formulas(self, query: str) -> bool:
        """
        Detect if the query contains mathematical or chemical formulas.

        Args:
            query: The query string

        Returns:
            bool: True if formulas are detected
        """
        # Simple pattern for formulas (can be enhanced)
        for pattern in FORMULA_PATTERNS:
            if re.search(pattern, query):
                return True

        return False

    def _assess_language_complexity(self, query: str) -> float:
        """
        Assess the linguistic complexity of the query.

        Args:
            query: The query string

        Returns:
            float: Language complexity score (0-1)
        """
        words = query.split()

        if not words:
            return 0.0

        # Average word length (longer words suggest complexity)
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_score = min(avg_word_length / QUERY_LENGTH_NORMALIZATION["avg_word_length_factor"], 1.0)

        # Sentence structure complexity (presence of clauses)
        clause_count = sum(1 for indicator in LANGUAGE_COMPLEXITY_INDICATORS if indicator in query)
        clause_score = min(clause_count / QUERY_LENGTH_NORMALIZATION["clause_count_factor"], 1.0)

        return (length_score + clause_score) / 2


class BatchQueryProcessor:
    """
    Handles batch processing of queries with parallel support.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the batch query processor.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})
        self.feature_extractor = QueryFeatureExtractor()
        self.max_workers = self.config.get('analysis', {}).get('max_workers', None)
        self.enable_parallel = self.config.get('analysis', {}).get('enable_parallel', True)

    def process_batch(self, queries: List[str], max_workers: Optional[int] = None) -> List[QueryFeatures]:
        """
        Process a batch of queries with optional parallel processing.

        Args:
            queries: List of query strings
            max_workers: Maximum number of worker threads

        Returns:
            List[QueryFeatures]: Analysis results for each query
        """
        if not queries:
            return []

        # Use parallel processing for batches larger than threshold
        batch_threshold = _query_config.get('batch_threshold', 10)
        if len(queries) > batch_threshold and self.enable_parallel and max_workers != 0:  # Allow disabling with max_workers=0
            if max_workers is None:
                default_max_workers = _query_config.get('default_max_workers', 4)
                max_workers = self.max_workers or min(default_max_workers, len(queries))  # More conservative default

            print(f"🔄 Analyzing {len(queries)} queries using {max_workers} parallel workers...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all analysis tasks
                future_to_query = {executor.submit(self.feature_extractor.extract_features, query): query for query in queries}

                # Collect results in order
                results = []
                for future in as_completed(future_to_query):
                    try:
                        result = future.result()
                        results.append((future_to_query[future], result))
                    except Exception as e:
                        print(f"Error analyzing query: {e}")
                        # Return empty features for failed queries
                        results.append((future_to_query[future], QueryFeatures(
                            length=0, word_count=0, scientific_terms=[], complexity_score=0.0,
                            query_type="unknown", domain=["unknown"], has_numbers=False,
                            has_formulas=False, language_complexity=0.0
                        )))

                # Sort results back to original order
                results.sort(key=lambda x: queries.index(x[0]))
                return [result for _, result in results]
        else:
            # Use sequential processing for small batches
            return [self.feature_extractor.extract_features(query) for query in queries]
