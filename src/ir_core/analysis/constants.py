# src/ir_core/analysis/constants.py

"""
Constants and configuration values for the Scientific QA retrieval analysis module.

This module centralizes all hardcoded values, thresholds, and domain-specific
keywords to improve maintainability and reduce duplication across the codebase.

Updated with insights from data profiling:
- Actual source distribution (ko_ai2_arc, ko_mmlu domains)
- Dataset-specific thresholds based on profiling outputs
- Profiling-enhanced retrieval configuration constants

Now loads configuration from YAML file for better maintainability.
"""

from typing import Dict, List, Set, Any
import re
import os
import json
from pathlib import Path

# Import configuration loader
from .config import get_config_loader

# Get configuration instance
_config_loader = get_config_loader()

# Load configuration
try:
    _config = _config_loader.load_config()
except Exception as e:
    print(f"Warning: Could not load configuration file: {e}")
    print("Falling back to default values")
    _config = None


# Dataset source mapping based on profiling (63 unique sources total)
DATASET_SOURCES: Dict[str, List[str]] = _config_loader.get('dataset_sources', {}) if _config else {}

# Source distribution insights from profiling (4272 total documents)
SOURCE_DISTRIBUTION_INSIGHTS: Dict[str, Any] = _config_loader.get('source_distribution', {}) if _config else {}

# Profiling-enhanced retrieval configuration
PROFILING_CONFIG: Dict[str, Any] = _config_loader.get('profiling', {}) if _config else {}

# Scientific domain keywords for classification (Korean and English)
DOMAIN_KEYWORDS: Dict[str, List[str]] = _config_loader.get('domain_classification.keywords', {}) if _config else {}

def _load_scientific_terms_from_artifact() -> List[str]:
    """Attempt to load scientific terms extracted by LLM from profiling artifacts.

    Looks for `${IR_PROFILE_REPORT_DIR or PROFILING_CONFIG['profile_report_dir']}/scientific_terms_extracted.json`.
    Accepts either a list[str] or a mapping {src: list[str]} and flattens/dedupes.
    Returns an empty list if not found or malformed.
    """
    try:
        base_dir = os.environ.get("IR_PROFILE_REPORT_DIR", PROFILING_CONFIG.get("profile_report_dir", "outputs/reports/data_profile/latest"))
        path = Path(base_dir) / "scientific_terms_extracted.json"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        terms: List[str] = []
        if isinstance(data, list):
            terms = [str(t).strip() for t in data if str(t).strip()]
        elif isinstance(data, dict):
            for _src, lst in data.items():
                if isinstance(lst, list):
                    terms.extend([str(t).strip() for t in lst if str(t).strip()])
        # Deduplicate while preserving insertion order
        seen = set()
        deduped = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped
    except Exception:
        # Fail closed: just return empty and rely on static fallback
        return []

# Scientific terms for complexity scoring and feature extraction
# Base fallback list (expanded) covering major scientific domains in Korean
SCIENTIFIC_TERMS_BASE: List[str] = _config_loader.get('scientific_terms.base_terms', []) if _config else []

def _load_persistent_terms_from_data() -> List[str]:
    """Load persistent curated terms from data/scientific_terms.json if present.

    Returns a list if the file exists and contains a non-empty list; otherwise returns [].
    """
    data_path = Path("data") / "scientific_terms.json"
    try:
        if data_path.exists():
            with data_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and any(str(x).strip() for x in data):
                # Normalize and dedupe while preserving order
                seen = set()
                out: List[str] = []
                for x in data:
                    s = str(x).strip()
                    if s and s not in seen:
                        seen.add(s)
                        out.append(s)
                return out
    except Exception:
        pass
    return []

# Resolve scientific terms with the following precedence (UPDATED: always merge for hybrid approach):
# 1) If SCIENTIFIC_TERMS_MODE is set:
#    - 'base_only' -> use base
#    - 'dynamic_only' -> use dynamic artifact only
#    - 'merge' -> ALWAYS merge base + dynamic artifact + data file (hybrid approach)
# 2) If no mode set -> merge base + dynamic artifact + data file (if any), otherwise use base
_mode = os.environ.get("SCIENTIFIC_TERMS_MODE", _config_loader.get('scientific_terms.mode', "merge") if _config else "merge").lower()
if _mode == "base_only":
    SCIENTIFIC_TERMS: List[str] = SCIENTIFIC_TERMS_BASE
elif _mode == "dynamic_only":
    SCIENTIFIC_TERMS = _load_scientific_terms_from_artifact()
else:  # merge mode (default) - ALWAYS merge for hybrid approach
    _dynamic_terms = _load_scientific_terms_from_artifact()
    _data_terms = _load_persistent_terms_from_data()
    _all_additional_terms = _dynamic_terms + _data_terms

    if _all_additional_terms:
        # Hybrid approach: merge base terms with dynamic and data terms
        merged: List[str] = []
        seen: Set[str] = set()
        for t in (SCIENTIFIC_TERMS_BASE + _all_additional_terms):
            if t and t not in seen:
                seen.add(t)
                merged.append(t)
        SCIENTIFIC_TERMS = merged
        print(f"🔄 Hybrid mode: Merged {len(SCIENTIFIC_TERMS_BASE)} base + {len(_dynamic_terms)} dynamic + {len(_data_terms)} data = {len(merged)} total terms")
    else:
        SCIENTIFIC_TERMS = SCIENTIFIC_TERMS_BASE
        print(f"ℹ️ Hybrid mode: No additional terms found, using {len(SCIENTIFIC_TERMS_BASE)} base terms only")

# Query type classification patterns
QUERY_TYPE_PATTERNS: Dict[str, re.Pattern] = {}
if _config:
    query_patterns_config = _config_loader.get('query_type_patterns', {})
    for pattern_name, patterns in query_patterns_config.items():
        if isinstance(patterns, list) and patterns:
            # Join multiple patterns with OR
            combined_pattern = '|'.join(patterns)
            QUERY_TYPE_PATTERNS[pattern_name] = re.compile(combined_pattern, re.IGNORECASE)
        elif isinstance(patterns, str):
            QUERY_TYPE_PATTERNS[pattern_name] = re.compile(patterns, re.IGNORECASE)
else:
    QUERY_TYPE_PATTERNS = {}

# Default K values for precision/recall calculations
DEFAULT_K_VALUES: List[int] = _config_loader.get('metrics.default_k_values', [1, 3, 5, 10]) if _config else [1, 3, 5, 10]

# Analysis thresholds updated with profiling insights
ANALYSIS_THRESHOLDS: Dict[str, float] = _config_loader.get('analysis.thresholds', {}) if _config else {}

# Source-aware chunking recommendations based on profiling
CHUNKING_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = _config_loader.get('chunking.recommendations', {}) if _config else {}

# Parallel processing defaults
PARALLEL_PROCESSING_DEFAULTS: Dict[str, Any] = _config_loader.get('parallel', {}) if _config else {}

# Language complexity assessment
LANGUAGE_COMPLEXITY_INDICATORS: List[str] = _config_loader.get('query_analysis.language_complexity_indicators', []) if _config else []

# Formula detection patterns
FORMULA_PATTERNS: List[str] = _config_loader.get('query_analysis.formula_patterns', []) if _config else []

# Error category definitions for comprehensive analysis
ERROR_CATEGORIES: Dict[str, Dict[str, Any]] = _config_loader.get('error_analysis.categories', {}) if _config else {
    # Query Understanding Failures
    "ambiguous_query": {
        "description": "Query is too vague or has multiple interpretations",
        "category": "query_understanding",
        "severity": "medium",
        "indicators": ["multiple_meanings", "vague_terms", "unclear_intent"]
    },
    "out_of_domain": {
        "description": "Query falls outside the system's knowledge domain",
        "category": "query_understanding",
        "severity": "high",
        "indicators": ["unknown_domain", "no_matching_docs", "domain_mismatch"]
    },
    "complex_multi_concept": {
        "description": "Query contains multiple complex scientific concepts",
        "category": "query_understanding",
        "severity": "high",
        "indicators": ["multiple_domains", "technical_terms", "compound_query"]
    },

    # Retrieval Failures
    "false_positive": {
        "description": "Retrieved irrelevant documents with high confidence",
        "category": "retrieval",
        "severity": "medium",
        "indicators": ["high_score_irrelevant", "wrong_domain_retrieved", "topic_mismatch"]
    },
    "false_negative": {
        "description": "Failed to retrieve relevant documents",
        "category": "retrieval",
        "severity": "high",
        "indicators": ["relevant_doc_missing", "low_score_relevant", "retrieval_failure"]
    },
    "ranking_error": {
        "description": "Correct documents retrieved but poorly ranked",
        "category": "retrieval",
        "severity": "medium",
        "indicators": ["ground_truth_low_rank", "wrong_order", "ranking_inconsistency"]
    },

    # Retrieval Quality Failures
    "source_bias": {
        "description": "Results heavily skewed toward certain sources (e.g., ARC dominance)",
        "category": "retrieval",
        "severity": "medium",
        "indicators": ["arc_over_representation", "mmlu_under_representation", "source_imbalance"]
    },
    "duplicate_contamination": {
        "description": "Retrieved results contain multiple copies of same content",
        "category": "retrieval",
        "severity": "low",
        "indicators": ["exact_duplicates", "near_duplicates", "content_repetition"]
    },
    "keyword_boost_failure": {
        "description": "Source-specific keyword boosting not working as expected",
        "category": "retrieval",
        "severity": "medium",
        "indicators": ["boost_not_applied", "wrong_source_boosted", "boost_ineffective"]
    },

    # System Failures
    "timeout_error": {
        "description": "Query processing exceeded time limits",
        "category": "system",
        "severity": "high",
        "indicators": ["processing_timeout", "response_delay", "system_overload"]
    },
    "parsing_error": {
        "description": "Failed to parse query or document content",
        "category": "system",
        "severity": "medium",
        "indicators": ["malformed_query", "content_extraction_failure", "format_error"]
    },
    "infrastructure_error": {
        "description": "System infrastructure or connectivity issues",
        "category": "system",
        "severity": "critical",
        "indicators": ["connection_failure", "service_unavailable", "resource_exhaustion"]
    }
}

# Enhanced error analysis thresholds with profiling insights
ERROR_ANALYSIS_THRESHOLDS: Dict[str, float] = _config_loader.get('error_analysis.thresholds', {}) if _config else {}

# Pattern detection configurations with profiling insights
PATTERN_DETECTION_CONFIG: Dict[str, Any] = _config_loader.get('pattern_detection', {}) if _config else {}

# Query length normalization factors
QUERY_LENGTH_NORMALIZATION: Dict[str, float] = _config_loader.get('query_analysis.length_normalization', {}) if _config else {}

# Validation set generation domains updated with actual source distribution
VALIDATION_DOMAINS: Dict[str, str] = _config_loader.get('validation.domains', {}) if _config else {}

# Profiling artifacts reference paths (for runtime loading)
PROFILING_ARTIFACTS: Dict[str, str] = _config_loader.get('profiling_artifacts', {}) if _config else {}
