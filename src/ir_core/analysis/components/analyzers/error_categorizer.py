# src/ir_core/analysis/components/analyzers/error_categorizer.py

"""
Error Categorizer Component

Handles error categorization logic for the Scientific QA retrieval system.
Classifies different types of retrieval errors and failures.
"""

from typing import Dict, List, Any, Optional
from omegaconf import DictConfig

from ...constants import (
    ERROR_ANALYSIS_THRESHOLDS,
    SCIENTIFIC_TERMS
)


class ErrorCategorizer:
    """
    Categorizes different types of retrieval errors.

    Analyzes failed retrievals and classifies them into categories:
    - Query Understanding Failures (ambiguous, out-of-domain, complex queries)
    - Retrieval Failures (false positives, false negatives, ranking errors)
    - System Failures (timeouts, parsing errors, infrastructure issues)
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the error categorizer.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})

    def categorize_error(
        self,
        pred_docs: List[Dict[str, Any]],
        gt_id: str,
        query: str,
        query_domain: List[str]
    ) -> str:
        """
        Categorize the type of error for a failed retrieval.

        Args:
            pred_docs: Predicted documents
            gt_id: Ground truth ID
            query: Original query
            query_domain: Query domain classification

        Returns:
            str: Error category key
        """
        query_lower = query.lower()

        # Check for ambiguous query (multiple domains or vague terms)
        domain_count = len([d for d in query_domain if d != "unknown"])
        if domain_count > 2:
            return "complex_multi_concept"

        # Check for out-of-domain
        if "unknown" in query_domain or domain_count == 0:
            return "out_of_domain"

        # Check for ambiguous terms
        ambiguous_indicators = ["무엇", "어떻게", "왜", "어디", "언제"]
        if any(indicator in query_lower for indicator in ambiguous_indicators):
            return "ambiguous_query"

        # Check retrieval quality
        top_scores = []
        for doc in pred_docs[:3]:
            if isinstance(doc, dict):
                top_scores.append(doc.get("score", 0.0))
            else:
                top_scores.append(0.0)
        avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        # Check if ground truth has low score (false negative)
        gt_doc = None
        for doc in pred_docs:
            if isinstance(doc, dict) and doc.get("id") == gt_id:
                gt_doc = doc
                break
            elif isinstance(doc, str) and doc == gt_id:
                gt_doc = {"id": doc, "score": 0.0}
                break
        if gt_doc and isinstance(gt_doc, dict) and gt_doc.get("score", 0.0) < ERROR_ANALYSIS_THRESHOLDS["false_negative_threshold"]:
            return "false_negative"

        # Check for false positives (high score but wrong)
        if avg_top_score > ERROR_ANALYSIS_THRESHOLDS["false_positive_threshold"]:
            return "false_positive"

        # Check ranking error
        gt_rank = -1
        for i, doc in enumerate(pred_docs):
            if isinstance(doc, dict) and doc.get("id") == gt_id:
                gt_rank = i
                break
            elif isinstance(doc, str) and doc == gt_id:
                gt_rank = i
                break
        if gt_rank > 5:  # Ground truth not in top 5
            return "ranking_error"

        # Default to false negative
        return "false_negative"

    def classify_query_type(self, query: str) -> str:
        """
        Classify query type for pattern analysis.

        Args:
            query: Query string to classify

        Returns:
            str: Query type classification
        """
        from ...constants import QUERY_TYPE_PATTERNS

        for type_name, pattern in QUERY_TYPE_PATTERNS.items():
            if pattern.search(query):
                return type_name
        return "general"