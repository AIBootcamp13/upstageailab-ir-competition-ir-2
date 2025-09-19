# src/ir_core/analysis/components/processors/__init__.py

"""
Query Processors Package

Contains components for processing and analyzing query batches
with feature extraction and domain classification.
"""

from .query_processor import QueryBatchProcessor, QueryProcessingResult

__all__ = ['QueryBatchProcessor', 'QueryProcessingResult']
