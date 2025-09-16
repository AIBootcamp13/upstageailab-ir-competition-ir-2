"""
Query Enhancement Module

This module provides various techniques to improve query quality for better retrieval performance
in the Information Retrieval RAG system.

Techniques implemented:
- Query Rewriting & Expansion
- Step-Back Prompting
- Query Decomposition
- Hypothetical Document Embeddings (HyDE)
- Query Translation
"""

from .manager import QueryEnhancementManager
from .strategic_classifier import StrategicQueryClassifier, QueryType
from .rewriter import QueryRewriter
from .step_back import StepBackPrompting
from .decomposer import QueryDecomposer
from .hyde import HyDE
from .translator import QueryTranslator

__all__ = [
    "QueryEnhancementManager",
    "QueryRewriter",
    "StepBackPrompting",
    "QueryDecomposer",
    "HyDE",
    "QueryTranslator",
]