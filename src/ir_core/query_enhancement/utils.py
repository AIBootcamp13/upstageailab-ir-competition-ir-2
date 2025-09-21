# src/ir_core/query_enhancement/utils.py

"""
Utility functions for query enhancement module.

This module contains common utility functions used across query enhancement components.
"""

from typing import List
from .constants import QUESTION_WORDS, CONVERSATIONAL_INDICATORS


def has_question_words(query: str) -> bool:
    """
    Check if query contains question words.

    Args:
        query: Query to check

    Returns:
        True if question words are present
    """
    query_lower = query.lower()
    return any(word in query_lower for word in QUESTION_WORDS)


def has_conversational_indicators(query: str) -> bool:
    """
    Check if query contains conversational indicators.

    Args:
        query: Query to check

    Returns:
        True if conversational indicators are present
    """
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in CONVERSATIONAL_INDICATORS)


def is_conversational_query(query: str) -> bool:
    """
    Determine if a query is conversational based on indicators.

    Args:
        query: Query to check

    Returns:
        True if query appears to be conversational
    """
    return has_conversational_indicators(query)