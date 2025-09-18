# src/ir_core/query_enhancement/constants.py

"""
Constants and configuration values for query enhancement module.

This module centralizes all hardcoded values, patterns, and thresholds
to improve maintainability and reduce duplication across the codebase.
"""

from typing import List

# Question words for query analysis (English and Korean)
QUESTION_WORDS: List[str] = [
    'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose',
    '무엇', '어떻게', '왜', '언제', '어디', '누구', '어느', '누구의'
]

# Conversational indicators for query classification
CONVERSATIONAL_INDICATORS: List[str] = [
    '너', '너는', '너의', '안녕', '반가워', '고마워', '미안',
    '기분', '힘들', '좋아', '싫어', '반갑다', '고맙다', '미안하다'
]

# Query type mapping for fallback classification
QUERY_TYPE_MAPPING = {
    "what": "conceptual_vague",
    "how": "conceptual_vague",
    "why": "conceptual_vague",
    "when": "simple_keyword",
    "where": "simple_keyword",
    "calculate": "simple_keyword",
    "general": "conceptual_vague"
}

# Default confidence scores for different techniques
# These scores represent the system's confidence in enhancement quality
# Scale: 0.0 (no confidence) to 1.0 (maximum confidence)
#
# Usage guidelines:
# - 0.8-0.9: High confidence, use for primary retrieval
# - 0.5-0.7: Moderate confidence, consider with monitoring
# - 0.0-0.4: Low confidence, use fallback strategies
DEFAULT_CONFIDENCE_SCORES = {
    'none': 0.5,           # Moderate confidence - no enhancement applied
    'bypass': 0.9,         # High confidence - query passed through unchanged
    'rewriting': 0.8,      # High confidence - generally reliable for most queries
    'step_back': 0.7,      # Good confidence - effective for ambiguous queries
    'decomposition': 0.9,  # Very high confidence - effective for complex queries
    'hyde': 0.9,           # Very high confidence - uses embeddings for retrieval
    'translation': 0.6     # Moderate confidence - depends on language detection accuracy
}

# Priority order for technique selection (lower number = higher priority)
TECHNIQUE_PRIORITY_ORDER = [
    'rewriting',
    'step_back',
    'decomposition',
    'hyde',
    'translation'
]