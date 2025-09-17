# src/ir_core/analysis/query_rewrite_evaluator.py

"""
Query rewrite evaluation functionality.

This module provides functionality to measure the effectiveness
of query rewriting operations.
"""

from typing import Dict, List, Any, Optional
from omegaconf import DictConfig

from .query_components import QueryFeatures, QueryFeatureExtractor
from .constants import ANALYSIS_THRESHOLDS


class QueryRewriteEvaluator:
    """
    Evaluates the effectiveness of query rewriting operations.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the query rewrite evaluator.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})
        self.feature_extractor = QueryFeatureExtractor()

    def measure_rewrite_effectiveness(
        self,
        original_query: str,
        rewritten_query: str
    ) -> Dict[str, Any]:
        """
        Measure the effectiveness of query rewriting.

        Args:
            original_query: The original query
            rewritten_query: The rewritten query

        Returns:
            Dict[str, Any]: Effectiveness metrics
        """
        if original_query == rewritten_query:
            return {
                "was_rewritten": False,
                "effectiveness_score": 0.0,
                "changes": []
            }

        # Analyze changes
        orig_features = self.feature_extractor.extract_features(original_query)
        rew_features = self.feature_extractor.extract_features(rewritten_query)

        changes = self._analyze_changes(orig_features, rew_features)
        effectiveness_score = self._calculate_effectiveness_score(changes, orig_features, rew_features)

        return {
            "was_rewritten": True,
            "effectiveness_score": effectiveness_score,
            "changes": changes,
            "original_features": orig_features,
            "rewritten_features": rew_features
        }

    def _analyze_changes(
        self,
        orig_features: QueryFeatures,
        rew_features: QueryFeatures
    ) -> List[str]:
        """
        Analyze what changes occurred between original and rewritten queries.

        Args:
            orig_features: Features of the original query
            rew_features: Features of the rewritten query

        Returns:
            List[str]: List of changes detected
        """
        changes = []

        # Length change
        length_diff = rew_features.length - orig_features.length
        if abs(length_diff) > 10:
            changes.append(f"length_change_{'increase' if length_diff > 0 else 'decrease'}")

        # Complexity change
        complexity_diff = rew_features.complexity_score - orig_features.complexity_score
        if abs(complexity_diff) > ANALYSIS_THRESHOLDS["query_complexity_change_threshold"]:
            changes.append(f"complexity_{'increase' if complexity_diff > 0 else 'decrease'}")

        # Domain change
        if set(orig_features.domain) != set(rew_features.domain):
            changes.append("domain_change")

        # Query type change
        if orig_features.query_type != rew_features.query_type:
            changes.append("query_type_change")

        # Scientific terms change
        orig_terms = set(orig_features.scientific_terms)
        rew_terms = set(rew_features.scientific_terms)
        if orig_terms != rew_terms:
            changes.append("scientific_terms_change")

        return changes

    def _calculate_effectiveness_score(
        self,
        changes: List[str],
        orig_features: QueryFeatures,
        rew_features: QueryFeatures
    ) -> float:
        """
        Calculate the effectiveness score based on detected changes.

        Args:
            changes: List of detected changes
            orig_features: Features of the original query
            rew_features: Features of the rewritten query

        Returns:
            float: Effectiveness score between 0 and 1
        """
        # Base score
        effectiveness_score = 0.5

        # Some changes are generally good
        if changes:
            effectiveness_score += 0.2

        # Significant complexity change
        complexity_diff = rew_features.complexity_score - orig_features.complexity_score
        if abs(complexity_diff) > ANALYSIS_THRESHOLDS["significant_complexity_change"]:
            effectiveness_score += 0.1

        return min(effectiveness_score, 1.0)

    def compare_queries(
        self,
        original_query: str,
        rewritten_query: str
    ) -> Dict[str, Any]:
        """
        Compare two queries and provide detailed comparison metrics.

        Args:
            original_query: The original query
            rewritten_query: The rewritten query

        Returns:
            Dict[str, Any]: Detailed comparison results
        """
        effectiveness = self.measure_rewrite_effectiveness(original_query, rewritten_query)

        # Add additional comparison metrics
        orig_features = effectiveness["original_features"]
        rew_features = effectiveness["rewritten_features"]

        comparison = {
            "effectiveness": effectiveness,
            "metrics": {
                "length_ratio": rew_features.length / orig_features.length if orig_features.length > 0 else 0,
                "complexity_ratio": rew_features.complexity_score / orig_features.complexity_score if orig_features.complexity_score > 0 else 0,
                "word_count_ratio": rew_features.word_count / orig_features.word_count if orig_features.word_count > 0 else 0,
                "scientific_terms_added": len(set(rew_features.scientific_terms) - set(orig_features.scientific_terms)),
                "scientific_terms_removed": len(set(orig_features.scientific_terms) - set(rew_features.scientific_terms)),
            }
        }

        return comparison