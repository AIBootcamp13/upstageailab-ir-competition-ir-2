# src/ir_core/analysis/components/calculators/__init__.py

"""
Metric Calculators Package

Contains components for calculating various retrieval metrics
with support for parallel processing and batch operations.
"""

from .metric_calculator import MetricCalculator, MetricCalculationResult

__all__ = ['MetricCalculator', 'MetricCalculationResult']
