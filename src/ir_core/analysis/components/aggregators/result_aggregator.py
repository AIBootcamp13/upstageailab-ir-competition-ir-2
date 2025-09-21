# src/ir_core/analysis/components/aggregators/result_aggregator.py

"""
Result Aggregator Component

Aggregates results from different analysis components and generates
actionable recommendations for system improvement.
"""

from typing import List

from ...constants import ANALYSIS_THRESHOLDS


class ResultAggregator:
    """
    Aggregates results from different analysis components and generates recommendations.
    """

    @staticmethod
    def generate_recommendations(
        map_score: float,
        retrieval_success_rate: float,
        rewrite_rate: float
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis results.

        Args:
            map_score: Mean Average Precision
            retrieval_success_rate: Success rate
            rewrite_rate: Rewrite rate

        Returns:
            List[str]: Recommendations
        """
        recommendations = []

        if map_score < ANALYSIS_THRESHOLDS["map_score_low"]:
            recommendations.append("MAP score is below 0.5. Consider improving retrieval algorithm or expanding document collection.")

        if retrieval_success_rate < ANALYSIS_THRESHOLDS["retrieval_success_rate_low"]:
            recommendations.append("Retrieval success rate is below 70%. Focus on improving top-10 retrieval accuracy.")

        if rewrite_rate > ANALYSIS_THRESHOLDS["rewrite_rate_high"]:
            recommendations.append("High rewrite rate detected. Verify that query rewriting is improving rather than degrading performance.")

        if rewrite_rate < ANALYSIS_THRESHOLDS["rewrite_rate_low"]:
            recommendations.append("Low rewrite rate. Consider enabling query rewriting to improve retrieval for conversational queries.")

        if not recommendations:
            recommendations.append("Overall performance looks good. Consider fine-tuning hyperparameters for marginal improvements.")

        return recommendations
