# src/ir_core/analysis/retrieval_analyzer.py

"""
Specialized retrieval quality assessment module for Phase 3.
Provides detailed analysis of document ranking, score distributions,
retrieval consistency, and performance segmentation.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import statistics
from collections import defaultdict, Counter

from .utils import find_rank_of_ground_truth, calculate_top_k_precision
from .constants import DEFAULT_K_VALUES
from .config.config_loader import ConfigLoader

# Load configuration for retrieval quality settings
_config_loader = ConfigLoader()
_performance_config = _config_loader.get('retrieval_quality.performance_segmentation', {})


@dataclass
class RankingQualityMetrics:
    """Metrics for assessing document ranking quality."""
    ndcg_at_k: Dict[int, float]
    reciprocal_rank: float
    rank_distribution: Dict[int, int]
    score_variance: float
    score_range: Tuple[float, float]
    top_k_consistency: Dict[int, float]


@dataclass
class ScoreDistributionAnalysis:
    """Analysis of retrieval score distributions."""
    mean_score: float
    median_score: float
    score_std: float
    score_skewness: float
    score_percentiles: Dict[int, float]
    score_histogram: Dict[str, int]  # Binned score distribution


@dataclass
class RetrievalConsistencyMetrics:
    """Metrics for measuring retrieval consistency."""
    jaccard_similarity: float
    rank_correlation: float
    score_stability: float
    top_k_overlap: Dict[int, float]


@dataclass
class FalsePositiveNegativeAnalysis:
    """Analysis of false positives and negatives."""
    false_positive_rate: float
    false_negative_rate: float
    precision_by_rank: Dict[int, float]
    false_positive_by_domain: Dict[str, int]
    false_negative_by_domain: Dict[str, int]


@dataclass
class PerformanceSegmentation:
    """Performance segmentation based on AP scores."""
    high_performance_queries: List[Dict[str, Any]]  # High performance queries
    medium_performance_queries: List[Dict[str, Any]]  # Medium performance queries
    low_performance_queries: List[Dict[str, Any]]   # Low performance queries
    failed_queries: List[Dict[str, Any]]            # AP = 0.0

    @property
    def segmentation_stats(self) -> Dict[str, int]:
        """Get statistics for each performance segment."""
        return {
            "high_performance": len(self.high_performance_queries),
            "medium_performance": len(self.medium_performance_queries),
            "low_performance": len(self.low_performance_queries),
            "failed": len(self.failed_queries)
        }


@dataclass
class ConfidenceAnalysis:
    """Confidence analysis for retrieval results."""
    confidence_intervals: Dict[int, Tuple[float, float]]
    uncertainty_scores: List[float]
    score_calibration_error: float
    confidence_by_rank: Dict[int, float]


@dataclass
class RetrievalQualityResult:
    """Comprehensive retrieval quality assessment result."""
    ranking_quality: RankingQualityMetrics
    score_distribution: ScoreDistributionAnalysis
    consistency_metrics: RetrievalConsistencyMetrics
    false_positive_analysis: FalsePositiveNegativeAnalysis
    performance_segmentation: PerformanceSegmentation
    confidence_analysis: ConfidenceAnalysis
    top_k_accuracy_curves: Dict[int, List[float]]


class RetrievalQualityAnalyzer:
    """
    Specialized analyzer for retrieval quality assessment.
    Provides detailed metrics for ranking quality, score distributions,
    consistency measurement, and performance segmentation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the retrieval quality analyzer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.k_values = DEFAULT_K_VALUES

    def analyze_retrieval_quality(
        self,
        queries: List[Dict[str, Any]],
        retrieval_results: List[Dict[str, Any]],
        ground_truth_ids: List[str]
    ) -> RetrievalQualityResult:
        """
        Perform comprehensive retrieval quality analysis.

        Args:
            queries: List of query dictionaries
            retrieval_results: List of retrieval result dictionaries
            ground_truth_ids: List of ground truth document IDs

        Returns:
            RetrievalQualityResult: Comprehensive quality assessment
        """
        # Extract predicted documents and scores
        predicted_docs_list = self._extract_predicted_docs(retrieval_results)
        predicted_scores_list = self._extract_predicted_scores(retrieval_results)

        # Calculate ranking quality metrics
        ranking_quality = self._analyze_ranking_quality(
            predicted_docs_list, predicted_scores_list, ground_truth_ids
        )

        # Analyze score distributions
        score_distribution = self._analyze_score_distribution(predicted_scores_list)

        # Measure retrieval consistency
        consistency_metrics = self._analyze_retrieval_consistency(
            predicted_docs_list, predicted_scores_list
        )

        # Analyze false positives and negatives
        false_positive_analysis = self._analyze_false_positives_negatives(
            predicted_docs_list, ground_truth_ids, queries
        )

        # Perform performance segmentation
        performance_segmentation = self._segment_performance(
            queries, predicted_docs_list, ground_truth_ids
        )

        # Analyze confidence
        confidence_analysis = self._analyze_confidence(
            predicted_scores_list, predicted_docs_list, ground_truth_ids
        )

        # Generate top-K accuracy curves
        top_k_accuracy_curves = self._generate_top_k_accuracy_curves(
            predicted_docs_list, ground_truth_ids
        )

        return RetrievalQualityResult(
            ranking_quality=ranking_quality,
            score_distribution=score_distribution,
            consistency_metrics=consistency_metrics,
            false_positive_analysis=false_positive_analysis,
            performance_segmentation=performance_segmentation,
            confidence_analysis=confidence_analysis,
            top_k_accuracy_curves=top_k_accuracy_curves
        )

    def _extract_predicted_docs(self, retrieval_results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Extract predicted documents from retrieval results."""
        predicted_docs_list = []
        for result in retrieval_results:
            if result and "docs" in result:
                predicted_docs_list.append(result["docs"])
            else:
                predicted_docs_list.append([])
        return predicted_docs_list

    def _extract_predicted_scores(self, retrieval_results: List[Dict[str, Any]]) -> List[List[float]]:
        """Extract predicted scores from retrieval results."""
        predicted_scores_list = []
        for result in retrieval_results:
            if result and "docs" in result:
                scores = []
                for doc in result["docs"]:
                    if isinstance(doc, dict):
                        score = doc.get("score", 0.0)
                        # Filter out NaN values and replace with 0.0
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            scores.append(float(score))
                        else:
                            scores.append(0.0)
                    else:
                        scores.append(0.0)
                predicted_scores_list.append(scores)
            else:
                predicted_scores_list.append([])
        return predicted_scores_list

    def _analyze_ranking_quality(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        predicted_scores_list: List[List[float]],
        ground_truth_ids: List[str]
    ) -> RankingQualityMetrics:
        """Analyze document ranking quality."""
        ndcg_scores = defaultdict(list)
        reciprocal_ranks = []
        rank_distributions = defaultdict(int)
        score_variances = []
        score_ranges = []

        for pred_docs, pred_scores, gt_id in zip(predicted_docs_list, predicted_scores_list, ground_truth_ids):
            if not pred_docs:
                continue

            pred_ids = []
            for doc in pred_docs:
                if isinstance(doc, dict):
                    pred_ids.append(doc.get("id", ""))
                elif isinstance(doc, str):
                    pred_ids.append(doc)
                else:
                    pred_ids.append(str(doc) if doc else "")

            # Calculate NDCG@K
            for k in self.k_values:
                ndcg = self._calculate_ndcg_at_k(pred_ids, gt_id, k)
                ndcg_scores[k].append(ndcg)

            # Calculate reciprocal rank
            rank = find_rank_of_ground_truth(pred_ids, gt_id)
            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
                rank_distributions[rank] += 1
            else:
                reciprocal_ranks.append(0.0)

            # Score statistics
            if pred_scores:
                score_variances.append(np.var(pred_scores))
                score_ranges.append((min(pred_scores), max(pred_scores)))

        # Aggregate metrics
        avg_ndcg_at_k = {k: float(np.mean(scores)) if scores else 0.0 for k, scores in ndcg_scores.items()}
        avg_reciprocal_rank = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        avg_score_variance = float(np.mean(score_variances)) if score_variances else 0.0

        # Calculate score range
        if score_ranges:
            min_scores = [r[0] for r in score_ranges]
            max_scores = [r[1] for r in score_ranges]
            overall_score_range = (min(min_scores), max(max_scores))
        else:
            overall_score_range = (0.0, 0.0)

        # Calculate top-K consistency (simplified)
        top_k_consistency = {}
        for k in self.k_values:
            consistency_scores = []
            for pred_docs in predicted_docs_list:
                if len(pred_docs) >= k:
                    # Simplified consistency measure
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.0)
            top_k_consistency[k] = np.mean(consistency_scores) if consistency_scores else 0.0

        return RankingQualityMetrics(
            ndcg_at_k=avg_ndcg_at_k,
            reciprocal_rank=avg_reciprocal_rank,
            rank_distribution=dict(rank_distributions),
            score_variance=avg_score_variance,
            score_range=overall_score_range,
            top_k_consistency=top_k_consistency
        )

    def _calculate_ndcg_at_k(self, predicted_ids: List[str], ground_truth_id: str, k: int) -> float:
        """Calculate NDCG@K for a single query."""
        if not predicted_ids or ground_truth_id not in predicted_ids:
            return 0.0

        # Find rank of ground truth
        rank = predicted_ids.index(ground_truth_id) + 1
        if rank > k:
            return 0.0

        # Calculate DCG
        dcg = 1.0 / np.log2(rank + 1)

        # Calculate IDCG (ideal DCG)
        idcg = 1.0 / np.log2(2)  # Ground truth at rank 1

        return dcg / idcg if idcg > 0 else 0.0

    def _analyze_score_distribution(self, predicted_scores_list: List[List[float]]) -> ScoreDistributionAnalysis:
        """Analyze score distributions across all queries."""
        all_scores = []
        for scores in predicted_scores_list:
            # Filter out any remaining NaN values and ensure they're floats
            valid_scores = [s for s in scores if isinstance(s, (int, float)) and not np.isnan(s)]
            all_scores.extend(valid_scores)

        if not all_scores:
            return ScoreDistributionAnalysis(
                mean_score=0.0,
                median_score=0.0,
                score_std=0.0,
                score_skewness=0.0,
                score_percentiles={},
                score_histogram={}
            )

        # Convert to numpy array for calculations
        all_scores = np.array(all_scores, dtype=float)

        mean_score = float(np.mean(all_scores))
        median_score = float(np.median(all_scores))
        score_std = float(np.std(all_scores))

        # Calculate skewness
        if score_std > 0:
            score_skewness = float(np.mean(((np.array(all_scores) - mean_score) / score_std) ** 3))
        else:
            score_skewness = 0.0

        # Calculate percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        score_percentiles = {}
        for p in percentiles:
            score_percentiles[p] = np.percentile(all_scores, p)

        # Create histogram bins
        hist_bins = np.histogram_bin_edges(all_scores, bins=10)
        hist, _ = np.histogram(all_scores, bins=hist_bins)
        score_histogram = {}
        for i, count in enumerate(hist):
            bin_label = ".2f"
            score_histogram[bin_label] = int(count)

        return ScoreDistributionAnalysis(
            mean_score=mean_score,
            median_score=median_score,
            score_std=score_std,
            score_skewness=score_skewness,
            score_percentiles=score_percentiles,
            score_histogram=score_histogram
        )

    def _analyze_retrieval_consistency(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        predicted_scores_list: List[List[float]]
    ) -> RetrievalConsistencyMetrics:
        """Analyze retrieval consistency across queries."""
        if len(predicted_docs_list) < 2:
            return RetrievalConsistencyMetrics(
                jaccard_similarity=0.0,
                rank_correlation=0.0,
                score_stability=0.0,
                top_k_overlap={}
            )

        # Calculate pairwise similarities
        jaccard_similarities = []
        rank_correlations = []
        score_stabilities = []

        for i in range(len(predicted_docs_list)):
            for j in range(i + 1, len(predicted_docs_list)):
                docs_i = predicted_docs_list[i]
                docs_j = predicted_docs_list[j]
                scores_i = predicted_scores_list[i]
                scores_j = predicted_scores_list[j]

                # Jaccard similarity of document IDs
                ids_i = set()
                for doc in docs_i:
                    if isinstance(doc, dict):
                        ids_i.add(doc.get("id", ""))
                    elif isinstance(doc, str):
                        ids_i.add(doc)
                    else:
                        ids_i.add(str(doc) if doc else "")

                ids_j = set()
                for doc in docs_j:
                    if isinstance(doc, dict):
                        ids_j.add(doc.get("id", ""))
                    elif isinstance(doc, str):
                        ids_j.add(doc)
                    else:
                        ids_j.add(str(doc) if doc else "")
                if ids_i or ids_j:
                    jaccard = len(ids_i & ids_j) / len(ids_i | ids_j)
                    jaccard_similarities.append(jaccard)

                # Rank correlation (FIXED: check for zero standard deviation)
                if len(scores_i) > 1 and len(scores_j) > 1:
                    try:
                        # Convert to numpy arrays for std() check
                        s1 = np.array(scores_i[:min(len(scores_i), len(scores_j))])
                        s2 = np.array(scores_j[:min(len(scores_i), len(scores_j))])

                        # Only calculate correlation if std is not zero for both arrays
                        if s1.std() > 0 and s2.std() > 0:
                            corr = np.corrcoef(s1, s2)[0, 1]
                            if not np.isnan(corr):
                                rank_correlations.append(corr)
                    except Exception:
                        pass

                # Score stability (coefficient of variation)
                if scores_i:
                    cv_i = np.std(scores_i) / np.mean(scores_i) if np.mean(scores_i) > 0 else 0.0
                    score_stabilities.append(cv_i)
                if scores_j:
                    cv_j = np.std(scores_j) / np.mean(scores_j) if np.mean(scores_j) > 0 else 0.0
                    score_stabilities.append(cv_j)

        # Calculate top-K overlap
        top_k_overlap = {}
        for k in self.k_values:
            overlaps = []
            for docs in predicted_docs_list:
                if len(docs) >= k:
                    overlaps.append(1.0)
                else:
                    overlaps.append(0.0)
            top_k_overlap[k] = np.mean(overlaps) if overlaps else 0.0

        return RetrievalConsistencyMetrics(
            jaccard_similarity=float(np.mean(jaccard_similarities)) if jaccard_similarities else 0.0,
            rank_correlation=float(np.mean(rank_correlations)) if rank_correlations else 0.0,
            score_stability=float(np.mean(score_stabilities)) if score_stabilities else 0.0,
            top_k_overlap=top_k_overlap
        )

    def _analyze_false_positives_negatives(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str],
        queries: List[Dict[str, Any]]
    ) -> FalsePositiveNegativeAnalysis:
        """Analyze false positives and negatives."""
        false_positives = 0
        false_negatives = 0
        total_predictions = 0
        total_ground_truths = len(ground_truth_ids)

        precision_by_rank = defaultdict(list)
        false_positive_by_domain = defaultdict(int)
        false_negative_by_domain = defaultdict(int)

        for pred_docs, gt_id, query in zip(predicted_docs_list, ground_truth_ids, queries):
            pred_ids = []
            for doc in pred_docs:
                if isinstance(doc, dict):
                    pred_ids.append(doc.get("id", ""))
                elif isinstance(doc, str):
                    pred_ids.append(doc)
                else:
                    pred_ids.append(str(doc) if doc else "")
            total_predictions += len(pred_ids)

            # Count false positives (predicted but not ground truth)
            for pred_id in pred_ids:
                if pred_id != gt_id:
                    false_positives += 1

            # Count false negatives (ground truth not in predictions)
            if gt_id not in pred_ids:
                false_negatives += 1

            # Precision by rank
            for rank, pred_id in enumerate(pred_ids, 1):
                is_correct = pred_id == gt_id
                precision_by_rank[rank].append(1.0 if is_correct else 0.0)

        # Calculate rates
        false_positive_rate = false_positives / total_predictions if total_predictions > 0 else 0.0
        false_negative_rate = false_negatives / total_ground_truths if total_ground_truths > 0 else 0.0

        # Average precision by rank
        avg_precision_by_rank = {}
        for rank, precisions in precision_by_rank.items():
            avg_precision_by_rank[rank] = np.mean(precisions)

        return FalsePositiveNegativeAnalysis(
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            precision_by_rank=avg_precision_by_rank,
            false_positive_by_domain=dict(false_positive_by_domain),
            false_negative_by_domain=dict(false_negative_by_domain)
        )

    def _segment_performance(
        self,
        queries: List[Dict[str, Any]],
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str]
    ) -> PerformanceSegmentation:
        """Segment queries by performance based on AP scores."""
        high_performance = []
        medium_performance = []
        low_performance = []
        failed = []

        for query, pred_docs, gt_id in zip(queries, predicted_docs_list, ground_truth_ids):
            pred_ids = []
            for doc in pred_docs:
                if isinstance(doc, dict):
                    pred_ids.append(doc.get("id", ""))
                elif isinstance(doc, str):
                    pred_ids.append(doc)
                else:
                    pred_ids.append(str(doc) if doc else "")

            # Calculate AP score (simplified)
            if gt_id in pred_ids:
                rank = pred_ids.index(gt_id) + 1
                ap_score = 1.0 / rank
            else:
                ap_score = 0.0

            query_with_score = {**query, "ap_score": ap_score}

            high_threshold = _performance_config.get('high_performance_threshold', 0.8)
            medium_threshold = _performance_config.get('medium_performance_threshold', 0.4)
            failed_threshold = _performance_config.get('failed_threshold', 0.0)

            if ap_score > high_threshold:
                high_performance.append(query_with_score)
            elif ap_score > medium_threshold:
                medium_performance.append(query_with_score)
            elif ap_score > failed_threshold:
                low_performance.append(query_with_score)
            else:
                failed.append(query_with_score)

        return PerformanceSegmentation(
            high_performance_queries=high_performance,
            medium_performance_queries=medium_performance,
            low_performance_queries=low_performance,
            failed_queries=failed
        )

    def _analyze_confidence(
        self,
        predicted_scores_list: List[List[float]],
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str]
    ) -> ConfidenceAnalysis:
        """Analyze confidence in retrieval results."""
        confidence_intervals = {}
        uncertainty_scores = []
        confidence_by_rank = defaultdict(list)

        # Calculate confidence intervals for scores
        all_scores = []
        for scores in predicted_scores_list:
            all_scores.extend(scores)

        if all_scores:
            for k in self.k_values:
                if len(all_scores) >= k:
                    top_k_scores = sorted(all_scores, reverse=True)[:k]
                    mean_score = np.mean(top_k_scores)
                    std_score = np.std(top_k_scores)
                    confidence_intervals[k] = (mean_score - 1.96 * std_score, mean_score + 1.96 * std_score)

        # Calculate uncertainty scores (coefficient of variation)
        for scores in predicted_scores_list:
            if scores:
                mean_score = np.mean(scores)
                if mean_score > 0:
                    cv = np.std(scores) / mean_score
                    uncertainty_scores.append(cv)

        # Confidence by rank
        for pred_docs, pred_scores, gt_id in zip(predicted_docs_list, predicted_scores_list, ground_truth_ids):
            pred_ids = []
            for doc in pred_docs:
                if isinstance(doc, dict):
                    pred_ids.append(doc.get("id", ""))
                elif isinstance(doc, str):
                    pred_ids.append(doc)
                else:
                    pred_ids.append(str(doc) if doc else "")
            for rank, (pred_id, score) in enumerate(zip(pred_ids, pred_scores), 1):
                confidence_by_rank[rank].append(score)

        avg_confidence_by_rank = {}
        for rank, scores in confidence_by_rank.items():
            avg_confidence_by_rank[rank] = np.mean(scores)

        # Simplified score calibration error
        score_calibration_error = float(np.std(uncertainty_scores)) if uncertainty_scores else 0.0

        return ConfidenceAnalysis(
            confidence_intervals=confidence_intervals,
            uncertainty_scores=uncertainty_scores,
            score_calibration_error=score_calibration_error,
            confidence_by_rank=avg_confidence_by_rank
        )

    def _generate_top_k_accuracy_curves(
        self,
        predicted_docs_list: List[List[Dict[str, Any]]],
        ground_truth_ids: List[str]
    ) -> Dict[int, List[float]]:
        """Generate top-K accuracy curves."""
        accuracy_curves = {}

        max_k = max(self.k_values)
        for k in range(1, max_k + 1):
            accuracies = []
            for pred_docs, gt_id in zip(predicted_docs_list, ground_truth_ids):
                pred_ids = []
                if pred_docs:
                    for doc in pred_docs[:k]:
                        if isinstance(doc, dict):
                            pred_ids.append(doc.get("id", ""))
                        elif isinstance(doc, str):
                            pred_ids.append(doc)
                        else:
                            pred_ids.append(str(doc) if doc else "")
                is_correct = gt_id in pred_ids
                accuracies.append(1.0 if is_correct else 0.0)

            accuracy_curves[k] = accuracies

        return accuracy_curves
