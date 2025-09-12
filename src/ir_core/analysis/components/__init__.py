# src/ir_core/analysis/components/__init__.py

"""
Analysis Components Package

This package contains modular analysis components organized by functionality:
- calculators: Metric calculation and computation components
- processors: Query and batch processing components
- analyzers: Error and pattern analysis components
- aggregators: Result aggregation and recommendation components
- types: Data structures and result types
"""

from .calculators.metric_calculator import MetricCalculator, MetricCalculationResult
from .processors.query_processor import QueryBatchProcessor, QueryProcessingResult
from .analyzers.error_analyzer import ErrorAnalyzer, ErrorAnalysisResult
from .aggregators.result_aggregator import ResultAggregator

__all__ = [
    'MetricCalculator',
    'MetricCalculationResult',
    'QueryBatchProcessor',
    'QueryProcessingResult',
    'ErrorAnalyzer',
    'ErrorAnalysisResult',
    'ResultAggregator'
]
