# src/ir_core/analysis/__init__.py

"""
Enhanced Analysis Framework for Scientific QA Retrieval System

This module provides comprehensive analysis capabilities for evaluating
and improving retrieval performance in scientific question-answering tasks.

Modules:
    - core: Main analysis orchestrator and data structures
    - metrics: Comprehensive metrics calculation
    - query_analyzer: Query analysis and classification
    - retrieval_analyzer: Retrieval quality assessment
    - error_analyzer: Error pattern detection and categorization
    - domain_classifier: Scientific domain classification
    - visualizer: Framework-agnostic visualization utilities
    - components: Modular analysis components (calculators, processors, analyzers, aggregators)
"""

from .core import RetrievalAnalyzer, AnalysisResult
from .metrics import RetrievalMetrics
from .query_analyzer import QueryAnalyzer, QueryFeatures
from .components import (
    MetricCalculator,
    MetricCalculationResult,
    QueryBatchProcessor,
    QueryProcessingResult,
    ErrorAnalyzer,
    ErrorAnalysisResult,
    ResultAggregator
)

__all__ = [
    'RetrievalAnalyzer',
    'AnalysisResult',
    'RetrievalMetrics',
    'QueryAnalyzer',
    'QueryFeatures',
    # New modular components
    'MetricCalculator',
    'MetricCalculationResult',
    'QueryBatchProcessor',
    'QueryProcessingResult',
    'ErrorAnalyzer',
    'ErrorAnalysisResult',
    'ResultAggregator'
]
