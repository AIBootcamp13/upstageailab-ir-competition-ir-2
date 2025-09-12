# src/ir_core/analysis/components/processors/query_processor.py

"""
Query Processor Component

Handles batch processing of queries with feature extraction,
domain classification, and statistical analysis.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from omegaconf import DictConfig

from ...query_analyzer import QueryAnalyzer


@dataclass
class QueryProcessingResult:
    """Result of query processing."""
    original_queries: List[str]
    rewritten_queries: List[str]
    query_features_list: List[Any]  # QueryFeatures
    avg_query_length: float
    avg_query_complexity: float
    rewrite_rate: float
    domain_distribution: Dict[str, int]


class QueryBatchProcessor:
    """
    Handles batch processing of queries with analysis.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the query batch processor.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})
        self.query_analyzer = QueryAnalyzer(config)

    def process_batch(
        self,
        queries: List[Dict[str, Any]],
        rewritten_queries: Optional[List[str]] = None,
        max_workers: Optional[int] = None
    ) -> QueryProcessingResult:
        """
        Process a batch of queries and extract features.

        Args:
            queries: List of query dictionaries
            rewritten_queries: Optional rewritten queries
            max_workers: Maximum workers for parallel processing

        Returns:
            QueryProcessingResult: Processed query data
        """
        original_queries = [q.get("msg", [{}])[0].get("content", "") for q in queries]

        if rewritten_queries is None:
            rewritten_queries = original_queries

        # Analyze queries
        query_features_list = self.query_analyzer.analyze_batch(original_queries, max_workers)

        total_queries = len(original_queries)

        # Calculate aggregate statistics
        avg_query_length = sum(f.length for f in query_features_list) / total_queries if total_queries > 0 else 0
        avg_query_complexity = sum(f.complexity_score for f in query_features_list) / total_queries if total_queries > 0 else 0

        # Rewrite analysis
        rewrite_changes = sum(1 for orig, rew in zip(original_queries, rewritten_queries) if orig != rew)
        rewrite_rate = rewrite_changes / total_queries if total_queries > 0 else 0

        # Domain distribution
        domain_distribution = {}
        for features in query_features_list:
            for domain in features.domain:
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

        return QueryProcessingResult(
            original_queries=original_queries,
            rewritten_queries=rewritten_queries,
            query_features_list=query_features_list,
            avg_query_length=avg_query_length,
            avg_query_complexity=avg_query_complexity,
            rewrite_rate=rewrite_rate,
            domain_distribution=domain_distribution
        )
