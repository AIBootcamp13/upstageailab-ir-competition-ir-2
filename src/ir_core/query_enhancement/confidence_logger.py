# src/ir_core/query_enhancement/confidence_logger.py

"""
Confidence Score Logger with Rich Formatting

This module provides comprehensive logging for confidence scores with
color-coded output and detailed reasoning for debugging and monitoring.
"""

import logging
import sys
import math
from typing import Dict, Any, Optional, List
from datetime import datetime


class ConfidenceLogger:
    """Logger for confidence scores with rich formatting."""

    # ANSI color codes
    COLORS = {
        'high': '\033[92m',      # Green
        'medium': '\033[93m',    # Yellow
        'low': '\033[91m',       # Red
        'error': '\033[91m',     # Red
        'info': '\033[94m',      # Blue
        'warning': '\033[93m',   # Yellow
        'success': '\033[92m',   # Green
        'reset': '\033[0m',      # Reset
        'bold': '\033[1m',       # Bold
        'underline': '\033[4m',  # Underline
    }

    CONFIDENCE_LEVELS = {
        'very_high': {'min': 0.9, 'color': 'high', 'symbol': '🟢'},
        'high': {'min': 0.8, 'color': 'high', 'symbol': '🟢'},
        'medium': {'min': 0.5, 'color': 'medium', 'symbol': '🟡'},
        'low': {'min': 0.1, 'color': 'low', 'symbol': '🔴'},
        'zero': {'min': 0.0, 'color': 'error', 'symbol': '❌'},
    }

    def __init__(self, enable_colors: bool = True, log_level: str = 'INFO', debug_mode: Optional[bool] = None):
        """Initialize the confidence logger."""
        self.enable_colors = enable_colors and self._supports_color()
        self.logger = logging.getLogger('ConfidenceLogger')
        self.logger.setLevel(getattr(logging, log_level))

        # If debug_mode not explicitly set, try to get from settings
        if debug_mode is None:
            try:
                from ..config import settings
                self.debug_mode = getattr(settings, 'debug_confidence_logging', False)
            except (ImportError, AttributeError):
                self.debug_mode = False
        else:
            self.debug_mode = debug_mode

        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _supports_color(self) -> bool:
        """Check if terminal supports color output."""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def _get_confidence_level(self, confidence: float) -> Dict[str, Any]:
        """Get confidence level information."""
        if confidence >= 0.9:
            return self.CONFIDENCE_LEVELS['very_high']
        elif confidence >= 0.8:
            return self.CONFIDENCE_LEVELS['high']
        elif confidence >= 0.5:
            return self.CONFIDENCE_LEVELS['medium']
        elif confidence > 0.0:
            return self.CONFIDENCE_LEVELS['low']
        else:
            return self.CONFIDENCE_LEVELS['zero']

    def _colorize(self, text: str, color: str) -> str:
        """Apply color formatting if enabled."""
        if not self.enable_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _format_confidence_bar(self, confidence: float) -> str:
        """Create a visual confidence bar."""
        bar_length = 20
        filled = int(confidence * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        return f"[{bar}] {confidence:.2f}"

    def log_confidence_score(
        self,
        technique: str,
        confidence: float,
        query: str,
        reasoning: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        retrieval_scores: Optional[Dict[str, float]] = None,
        scientific_keywords: Optional[List[str]] = None,
        query_number: Optional[int] = None,
        total_queries: Optional[int] = None
    ) -> None:
        """Log a confidence score with rich formatting."""

        level_info = self._get_confidence_level(confidence)
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Create the main confidence message
        confidence_bar = self._format_confidence_bar(confidence)
        confidence_msg = f"{level_info['symbol']} {self._colorize(confidence_bar, level_info['color'])}"

        # Create alternative header format (more concise)
        if query_number is not None and total_queries is not None:
            query_info = f"[{query_number}/{total_queries}]"
        else:
            query_info = ""

        header = f"{self._colorize('🔍 QUERY ANALYSIS', 'bold')} {query_info} | {self._colorize(technique.upper(), 'bold')}"
        query_preview = f"{self._colorize('Query:', 'info')} {query[:50]}{'...' if len(query) > 50 else ''}"

        # Add scientific keywords if available
        keywords_display = ""
        if scientific_keywords and len(scientific_keywords) > 0:
            keywords_str = ', '.join(scientific_keywords[:5])  # Limit to 5 keywords
            if len(scientific_keywords) > 5:
                keywords_str += f" +{len(scientific_keywords) - 5} more"
            keywords_display = f"{self._colorize('Keywords:', 'info')} {keywords_str}"

        # Build the message
        message_parts = [
            f"\n{self._colorize('═' * 80, 'bold')}",
            header,
            f"{self._colorize('─' * 80, 'bold')}",
            query_preview,
        ]

        # Add keywords display if available
        if keywords_display:
            message_parts.append(keywords_display)

        message_parts.extend([
            f"{self._colorize('Confidence:', 'info')} {confidence_msg}",
        ])

        # Add reasoning if provided
        if reasoning:
            message_parts.append(f"{self._colorize('Reasoning:', 'info')} {reasoning}")

        # Add context information
        if context:
            context_items = []
            for key, value in context.items():
                if isinstance(value, bool):
                    status = self._colorize('✓', 'success') if value else self._colorize('✗', 'error')
                    context_items.append(f"{key}: {status}")
                elif isinstance(value, (int, float)):
                    context_items.append(f"{key}: {value}")
                elif isinstance(value, list):
                    # Pretty format lists, especially for recommended_techniques
                    if key == 'recommended_techniques' and value:
                        formatted_list = []
                        for item in value:
                            if isinstance(item, dict):
                                technique = item.get('technique', 'unknown')
                                priority = item.get('priority', 'N/A')
                                formatted_list.append(f"{technique}(p{priority})")
                            else:
                                formatted_list.append(str(item))
                        context_items.append(f"{key}: [{', '.join(formatted_list)}]")
                    else:
                        # Truncate long lists
                        list_str = str(value)
                        if len(list_str) > 100:
                            list_str = list_str[:97] + "..."
                        context_items.append(f"{key}: {list_str}")
                elif isinstance(value, dict):
                    # Pretty format dictionaries without truncation
                    if key == 'classification':
                        # Special handling for classification dictionaries
                        primary_type = value.get('primary_type', 'unknown')
                        scores = value.get('scores', {})
                        if scores:
                            # Format scores as key:value pairs
                            score_items = [f"{k}:{v}" for k, v in scores.items()]
                            formatted_scores = f"scores={{{', '.join(score_items)}}}"
                            context_items.append(f"{key}: {primary_type} ({formatted_scores})")
                        else:
                            context_items.append(f"{key}: {primary_type}")
                    elif key == 'recommended_techniques':
                        # Special handling for recommended_techniques dict
                        dict_str = str(value)
                        context_items.append(f"{key}: {dict_str}")
                    else:
                        dict_str = str(value)
                        if len(dict_str) > 100:
                            dict_str = dict_str[:97] + "..."
                        context_items.append(f"{key}: {dict_str}")
                else:
                    # Don't truncate other string values
                    context_items.append(f"{key}: {str(value)}")

            if context_items:
                message_parts.append(f"{self._colorize('Context:', 'info')} {' | '.join(context_items)}")

        # Add retrieval scores if available
        if retrieval_scores:
            if self.debug_mode:
                # Show detailed scores in debug mode
                score_items = []
                for score_type, score_value in retrieval_scores.items():
                    if isinstance(score_value, (int, float)) and not math.isnan(score_value):
                        score_items.append(f"{score_type}: {score_value:.3f}")

                if score_items:
                    message_parts.append(f"{self._colorize('Retrieval Scores:', 'info')} {' | '.join(score_items)}")
            else:
                # Show concise summary in normal mode
                has_scores = any(
                    isinstance(score_value, (int, float)) and not math.isnan(score_value) and score_value > 0
                    for score_value in retrieval_scores.values()
                )
                if has_scores:
                    message_parts.append(f"{self._colorize('Retrieval Scores:', 'info')} Available (use debug mode for details)")

        message_parts.append(f"{self._colorize('═' * 80, 'bold')}\n")

        # Log the message
        full_message = '\n'.join(message_parts)
        self.logger.info(full_message)

    def log_fallback_triggered(
        self,
        original_technique: str,
        fallback_technique: str,
        original_confidence: float,
        query: str,
        reason: str
    ) -> None:
        """Log when a fallback is triggered due to low confidence."""

        timestamp = datetime.now().strftime("%H:%M:%S")

        header = f"{self._colorize('🔄 FALLBACK TRIGGERED', 'warning')} | {timestamp}"
        original_level = self._get_confidence_level(original_confidence)

        message_parts = [
            f"\n{self._colorize('═' * 80, 'warning')}",
            header,
            f"{self._colorize('─' * 80, 'warning')}",
            f"{self._colorize('Original:', 'error')} {original_technique.upper()} {original_level['symbol']} ({original_confidence:.2f})",
            f"{self._colorize('Fallback:', 'success')} {fallback_technique.upper()}",
            f"{self._colorize('Reason:', 'info')} {reason}",
            f"{self._colorize('Query:', 'info')} {query[:50]}{'...' if len(query) > 50 else ''}",
            f"{self._colorize('═' * 80, 'warning')}\n",
        ]

        full_message = '\n'.join(message_parts)
        self.logger.warning(full_message)

    def log_error_confidence(
        self,
        technique: str,
        error: str,
        query: str
    ) -> None:
        """Log confidence score for error conditions."""

        timestamp = datetime.now().strftime("%H:%M:%S")

        header = f"{self._colorize('❌ CONFIDENCE ERROR', 'error')} | {timestamp}"

        message_parts = [
            f"\n{self._colorize('═' * 80, 'error')}",
            header,
            f"{self._colorize('─' * 80, 'error')}",
            f"{self._colorize('Technique:', 'error')} {technique.upper()}",
            f"{self._colorize('Confidence:', 'error')} ❌ [░░░░░░░░░░░░░░░░░░░░] 0.00",
            f"{self._colorize('Error:', 'error')} {error}",
            f"{self._colorize('Query:', 'info')} {query[:50]}{'...' if len(query) > 50 else ''}",
            f"{self._colorize('═' * 80, 'error')}\n",
        ]

        full_message = '\n'.join(message_parts)
        self.logger.error(full_message)


# Global confidence logger instance
_confidence_logger = None

def get_confidence_logger(debug_mode: Optional[bool] = None) -> ConfidenceLogger:
    """Get the global confidence logger instance."""
    global _confidence_logger
    if _confidence_logger is None:
        _confidence_logger = ConfidenceLogger(debug_mode=debug_mode)
    return _confidence_logger

def log_confidence_score(
    technique: str,
    confidence: float,
    query: str,
    reasoning: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    retrieval_scores: Optional[Dict[str, float]] = None,
    scientific_keywords: Optional[List[str]] = None,
    query_number: Optional[int] = None,
    total_queries: Optional[int] = None,
    debug_mode: Optional[bool] = None
) -> None:
    """Convenience function to log confidence scores."""
    logger = get_confidence_logger(debug_mode=debug_mode)
    logger.log_confidence_score(technique, confidence, query, reasoning, context, retrieval_scores, scientific_keywords, query_number, total_queries)

def log_fallback_triggered(
    original_technique: str,
    fallback_technique: str,
    original_confidence: float,
    query: str,
    reason: str
) -> None:
    """Convenience function to log fallback triggers."""
    logger = get_confidence_logger()
    logger.log_fallback_triggered(original_technique, fallback_technique, original_confidence, query, reason)

def log_error_confidence(
    technique: str,
    error: str,
    query: str
) -> None:
    """Convenience function to log confidence errors."""
    logger = get_confidence_logger()
    logger.log_error_confidence(technique, error, query)