# src/ir_core/analysis/core.py

"""
Core analysis classes and data structures for the Scientific QA retrieval system.
"""

from typing import Dict, List, Any, Optional, Union, cast
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .metrics import RetrievalMetrics
from .query_analyzer import QueryAnalyzer
from .components import (
    MetricCalculator,
    QueryBatchProcessor,
    ErrorAnalyzer,
    ResultAggregator,
    MetricCalculationResult,
    QueryProcessingResult,
    ErrorAnalysisResult
)
from .retrieval_analyzer import RetrievalQualityAnalyzer
from .utils import find_rank_of_ground_truth, calculate_top_k_precision
from .constants import (
    SCIENTIFIC_TERMS,
    DOMAIN_KEYWORDS,
    ANALYSIS_THRESHOLDS,
    ERROR_ANALYSIS_DOMAIN_CHECKS,
    DEFAULT_K_VALUES,
    PARALLEL_PROCESSING_DEFAULTS
)


@dataclass
class QueryAnalysis:
    """Analysis results for a single query."""
    original_query: str
    rewritten_query: str
    query_length: int
    domain: List[str]  # Support multiple domains
    complexity_score: float
    processing_time: float
    rewrite_effective: bool


@dataclass
class RetrievalResult:
    """Results from a single retrieval operation."""
    query: str
    ground_truth_id: str
    predicted_ids: List[str]
    predicted_scores: List[float]
    ap_score: float
    rank_of_ground_truth: Optional[int]
    top_k_precision: Dict[int, float]
    retrieval_time: float


@dataclass
class AnalysisResult:
    """Comprehensive analysis results for a batch of queries."""
    # Basic metrics
    map_score: float
    mean_ap: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]

    # Query analysis
    total_queries: int
    avg_query_length: float
    rewrite_rate: float
    domain_distribution: Dict[str, int]

    # Retrieval analysis
    retrieval_success_rate: float
    avg_retrieval_time: float
    error_categories: Dict[str, int]

    # Error analysis (Phase 4 enhancement)
    query_understanding_failures: Dict[str, int] = field(default_factory=dict)
    retrieval_failures: Dict[str, int] = field(default_factory=dict)
    system_failures: Dict[str, int] = field(default_factory=dict)
    error_patterns: Dict[str, Any] = field(default_factory=dict)
    domain_error_rates: Dict[str, float] = field(default_factory=dict)
    temporal_trends: Dict[str, List[float]] = field(default_factory=dict)
    error_recommendations: List[str] = field(default_factory=list)

    # Phase 3: Retrieval Quality Assessment
    performance_segmentation: Dict[str, int] = field(default_factory=dict)  # High/Medium/Low/Failed counts
    ranking_quality_metrics: Dict[str, Any] = field(default_factory=dict)   # NDCG, reciprocal rank, etc.
    score_distribution_metrics: Dict[str, Any] = field(default_factory=dict) # Mean, std, percentiles
    consistency_metrics: Dict[str, Any] = field(default_factory=dict)       # Jaccard, correlation
    false_positive_analysis: Dict[str, Any] = field(default_factory=dict)   # FP/FN rates
    confidence_analysis: Dict[str, Any] = field(default_factory=dict)       # Confidence intervals

    # Detailed results
    query_analyses: List[QueryAnalysis] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


class RetrievalAnalyzer:
    """
    Main analysis orchestrator for Scientific QA retrieval evaluation.

    This class coordinates various analysis components to provide comprehensive
    insights into retrieval performance.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the retrieval analyzer.

        Args:
            config: Hydra configuration object containing analysis settings
        """
        self.config = config
        self.metrics_calculator = MetricCalculator(config)
        self.query_processor = QueryBatchProcessor(config)
        self.error_analyzer = ErrorAnalyzer()
        self.result_aggregator = ResultAggregator()
        self.retrieval_quality_analyzer = RetrievalQualityAnalyzer(cast(Dict[str, Any], OmegaConf.to_container(config)) if config else None)
        self.enable_parallel = PARALLEL_PROCESSING_DEFAULTS.get("enable_parallel", True)
        self.max_workers = PARALLEL_PROCESSING_DEFAULTS.get("max_workers_analysis", 4)

    def analyze_batch(
        self,
        queries: List[Dict[str, Any]],
        retrieval_results: List[Dict[str, Any]],
        rewritten_queries: Optional[List[str]] = None,
        max_workers: Optional[int] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis on a batch of retrieval results with optional parallel processing.

        Args:
            queries: List of query dictionaries with original queries
            retrieval_results: List of retrieval result dictionaries
            rewritten_queries: Optional list of rewritten queries
            max_workers: Maximum number of worker threads for parallel processing

        Returns:
            AnalysisResult: Comprehensive analysis results
        """
        start_time = time.time()

        # Extract query information
        original_queries = [q.get("msg", [{}])[0].get("content", "") for q in queries]
        ground_truth_ids = [q.get("ground_truth_doc_id", "") for q in queries]

        # Extract retrieval results
        predicted_docs_list = []
        for result in retrieval_results:
            if result and "docs" in result:
                predicted_docs_list.append(result["docs"])
            else:
                predicted_docs_list.append([])

        # Prepare rewritten queries
        if rewritten_queries is None:
            rewritten_queries = original_queries

        # Calculate metrics using MetricCalculator
        metrics_result = self.metrics_calculator.calculate_batch_metrics(
            predicted_docs_list, ground_truth_ids, max_workers
        )
        map_score = metrics_result.map_score
        mean_ap = metrics_result.mean_ap
        precision_at_k = metrics_result.precision_at_k
        recall_at_k = metrics_result.recall_at_k
        ap_scores = metrics_result.ap_scores

        # Enhanced query analysis using QueryBatchProcessor
        query_result = self.query_processor.process_batch(queries, rewritten_queries, max_workers)
        query_features_list = query_result.query_features_list
        total_queries = len(original_queries)

        # Use pre-calculated aggregates from QueryProcessingResult
        avg_query_length = query_result.avg_query_length
        rewrite_rate = query_result.rewrite_rate
        domain_distribution = query_result.domain_distribution

        # Error categorization using ErrorAnalyzer
        # Extract query domains from features
        query_domains = [features.domain for features in query_features_list] if query_features_list else None

        error_result = self.error_analyzer.analyze_errors(
            predicted_docs_list, ground_truth_ids, original_queries, query_domains
        )
        retrieval_success_rate = error_result.retrieval_success_rate
        error_categories = error_result.error_categories

        # Create detailed results
        query_analyses = []
        retrieval_results_detailed = []

        for i, (orig_q, rew_q, gt_id, pred_docs, features) in enumerate(zip(
            original_queries, rewritten_queries, ground_truth_ids, predicted_docs_list, query_features_list
        )):
            # Query analysis using QueryAnalyzer features
            query_analysis = QueryAnalysis(
                original_query=orig_q,
                rewritten_query=rew_q,
                query_length=features.length,
                domain=features.domain,
                complexity_score=features.complexity_score,
                processing_time=0.0,  # Placeholder
                rewrite_effective=orig_q != rew_q
            )
            query_analyses.append(query_analysis)

            # Retrieval result details
            pred_ids = [doc.get("id", "") for doc in pred_docs]
            pred_scores = [doc.get("score", 0.0) for doc in pred_docs]

            retrieval_result = RetrievalResult(
                query=orig_q,
                ground_truth_id=gt_id,
                predicted_ids=pred_ids,
                predicted_scores=pred_scores,
                ap_score=ap_scores[i],
                rank_of_ground_truth=find_rank_of_ground_truth(pred_ids, gt_id),
                top_k_precision=calculate_top_k_precision(pred_ids, [gt_id], DEFAULT_K_VALUES),
                retrieval_time=0.0  # Placeholder
            )
            retrieval_results_detailed.append(retrieval_result)

        # Generate recommendations using ResultAggregator (legacy) and ErrorAnalyzer (enhanced)
        legacy_recommendations = self.result_aggregator.generate_recommendations(
            map_score, retrieval_success_rate, rewrite_rate
        )

        # Combine legacy and enhanced recommendations
        recommendations = legacy_recommendations + error_result.recommendations

        # Phase 3: Perform retrieval quality assessment
        retrieval_quality_result = self.retrieval_quality_analyzer.analyze_retrieval_quality(
            queries, retrieval_results, ground_truth_ids
        )

        analysis_time = time.time() - start_time
        print(f"Analysis completed in {analysis_time:.2f} seconds")

        return AnalysisResult(
            map_score=map_score,
            mean_ap=mean_ap,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            total_queries=total_queries,
            avg_query_length=avg_query_length,
            rewrite_rate=rewrite_rate,
            domain_distribution=domain_distribution,
            retrieval_success_rate=retrieval_success_rate,
            avg_retrieval_time=0.0,  # Placeholder
            error_categories=error_categories,
            query_understanding_failures=error_result.query_understanding_failures,
            retrieval_failures=error_result.retrieval_failures,
            system_failures=error_result.system_failures,
            error_patterns=error_result.error_patterns,
            domain_error_rates=error_result.domain_error_rates,
            temporal_trends=error_result.temporal_trends,
            error_recommendations=error_result.recommendations,
            performance_segmentation=retrieval_quality_result.performance_segmentation.segmentation_stats,
            ranking_quality_metrics={
                "ndcg_at_k": retrieval_quality_result.ranking_quality.ndcg_at_k,
                "reciprocal_rank": retrieval_quality_result.ranking_quality.reciprocal_rank,
                "rank_distribution": retrieval_quality_result.ranking_quality.rank_distribution,
                "score_variance": retrieval_quality_result.ranking_quality.score_variance,
                "score_range": retrieval_quality_result.ranking_quality.score_range,
                "top_k_consistency": retrieval_quality_result.ranking_quality.top_k_consistency
            },
            score_distribution_metrics={
                "mean_score": retrieval_quality_result.score_distribution.mean_score,
                "median_score": retrieval_quality_result.score_distribution.median_score,
                "score_std": retrieval_quality_result.score_distribution.score_std,
                "score_skewness": retrieval_quality_result.score_distribution.score_skewness,
                "score_percentiles": retrieval_quality_result.score_distribution.score_percentiles,
                "score_histogram": retrieval_quality_result.score_distribution.score_histogram
            },
            consistency_metrics={
                "jaccard_similarity": retrieval_quality_result.consistency_metrics.jaccard_similarity,
                "rank_correlation": retrieval_quality_result.consistency_metrics.rank_correlation,
                "score_stability": retrieval_quality_result.consistency_metrics.score_stability,
                "top_k_overlap": retrieval_quality_result.consistency_metrics.top_k_overlap
            },
            false_positive_analysis={
                "false_positive_rate": retrieval_quality_result.false_positive_analysis.false_positive_rate,
                "false_negative_rate": retrieval_quality_result.false_positive_analysis.false_negative_rate,
                "precision_by_rank": retrieval_quality_result.false_positive_analysis.precision_by_rank,
                "false_positive_by_domain": retrieval_quality_result.false_positive_analysis.false_positive_by_domain,
                "false_negative_by_domain": retrieval_quality_result.false_positive_analysis.false_negative_by_domain
            },
            confidence_analysis={
                "confidence_intervals": retrieval_quality_result.confidence_analysis.confidence_intervals,
                "uncertainty_scores": retrieval_quality_result.confidence_analysis.uncertainty_scores,
                "score_calibration_error": retrieval_quality_result.confidence_analysis.score_calibration_error,
                "confidence_by_rank": retrieval_quality_result.confidence_analysis.confidence_by_rank
            },
            query_analyses=query_analyses,
            retrieval_results=retrieval_results_detailed,
            recommendations=recommendations,
            timestamp=time.time(),
            config_snapshot=cast(Dict[str, Any], OmegaConf.to_container(self.config)) if self.config else {}
        )

