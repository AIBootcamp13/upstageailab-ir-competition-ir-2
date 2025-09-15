# src/ir_core/analysis/constants.py

"""
Constants and configuration values for the Scientific QA retrieval analysis module.

This module centralizes all hardcoded values, thresholds, and domain-specific
keywords to improve maintainability and reduce duplication across the codebase.

Updated with insights from data profiling:
- Actual source distribution (ko_ai2_arc, ko_mmlu domains)
- Dataset-specific thresholds based on profiling outputs
- Profiling-enhanced retrieval configuration constants
"""

from typing import Dict, List, Set, Any
import re
import os
import json
from pathlib import Path


# Dataset source mapping based on profiling (63 unique sources total)
DATASET_SOURCES: Dict[str, List[str]] = {
    "arc_challenge": [
        "ko_ai2_arc__ARC_Challenge__test",
        "ko_ai2_arc__ARC_Challenge__train",
        "ko_ai2_arc__ARC_Challenge__validation"
    ],
    "mmlu_biology": [
        "ko_mmlu__anatomy__test", "ko_mmlu__anatomy__train", "ko_mmlu__anatomy__validation",
        "ko_mmlu__college_biology__test", "ko_mmlu__college_biology__train", "ko_mmlu__college_biology__validation",
        "ko_mmlu__high_school_biology__test", "ko_mmlu__high_school_biology__train", "ko_mmlu__high_school_biology__validation",
        "ko_mmlu__human_aging__test", "ko_mmlu__human_aging__train", "ko_mmlu__human_aging__validation",
        "ko_mmlu__human_sexuality__test", "ko_mmlu__human_sexuality__train", "ko_mmlu__human_sexuality__validation",
        "ko_mmlu__medical_genetics__test", "ko_mmlu__medical_genetics__train", "ko_mmlu__medical_genetics__validation",
        "ko_mmlu__nutrition__test", "ko_mmlu__nutrition__train", "ko_mmlu__nutrition__validation",
        "ko_mmlu__virology__test", "ko_mmlu__virology__train", "ko_mmlu__virology__validation"
    ],
    "mmlu_physics": [
        "ko_mmlu__conceptual_physics__test", "ko_mmlu__conceptual_physics__train", "ko_mmlu__conceptual_physics__validation",
        "ko_mmlu__college_physics__test", "ko_mmlu__college_physics__train", "ko_mmlu__college_physics__validation",
        "ko_mmlu__high_school_physics__test", "ko_mmlu__high_school_physics__train", "ko_mmlu__high_school_physics__validation",
        "ko_mmlu__electrical_engineering__test", "ko_mmlu__electrical_engineering__train", "ko_mmlu__electrical_engineering__validation"
    ],
    "mmlu_chemistry": [
        "ko_mmlu__college_chemistry__test", "ko_mmlu__college_chemistry__train", "ko_mmlu__college_chemistry__validation",
        "ko_mmlu__high_school_chemistry__test", "ko_mmlu__high_school_chemistry__train", "ko_mmlu__high_school_chemistry__validation"
    ],
    "mmlu_computer_science": [
        "ko_mmlu__college_computer_science__test", "ko_mmlu__college_computer_science__train", "ko_mmlu__college_computer_science__validation",
        "ko_mmlu__high_school_computer_science__test", "ko_mmlu__high_school_computer_science__train", "ko_mmlu__high_school_computer_science__validation",
        "ko_mmlu__computer_security__test", "ko_mmlu__computer_security__train", "ko_mmlu__computer_security__validation"
    ],
    "mmlu_astronomy": [
        "ko_mmlu__astronomy__test", "ko_mmlu__astronomy__train", "ko_mmlu__astronomy__validation"
    ],
    "mmlu_medicine": [
        "ko_mmlu__college_medicine__test", "ko_mmlu__college_medicine__train", "ko_mmlu__college_medicine__validation"
    ],
    "mmlu_other": [
        "ko_mmlu__global_facts__test", "ko_mmlu__global_facts__train", "ko_mmlu__global_facts__validation"
    ]
}

# Source distribution insights from profiling (4272 total documents)
SOURCE_DISTRIBUTION_INSIGHTS: Dict[str, Any] = {
    "total_docs": 4272,
    "unique_sources": 63,
    "largest_sources": [
        ("ko_ai2_arc__ARC_Challenge__test", 943),      # 22% of corpus
        ("ko_ai2_arc__ARC_Challenge__train", 866),     # 20% of corpus
        ("ko_ai2_arc__ARC_Challenge__validation", 238), # 6% of corpus
    ],
    "arc_dominance": 0.48,  # ARC represents ~48% of total documents
    "mmlu_fragmentation": 0.52,  # MMLU spread across many small datasets
    "train_test_imbalance": True,  # Most MMLU sources have very few train examples (1-5)
}

# Profiling-enhanced retrieval configuration
PROFILING_CONFIG: Dict[str, Any] = {
    "use_src_boosts": True,           # Enable per-source keyword boosting
    "use_stopword_filtering": False,   # Conservative: disable by default
    "use_duplicate_filtering": True,   # Safe: enable exact duplicate removal
    "use_near_dup_penalty": False,    # Experimental: disable by default
    "profile_report_dir": "outputs/reports/data_profile/latest",

    # TF-IDF keyword extraction settings
    "keywords_top_k": 20,
    "min_df": 2,
    "max_features": 20000,

    # Stopword extraction settings
    "stopwords_top_n": 200,
    "per_src_stopwords_top_n": 50,

    # Near-duplicate detection settings
    "near_dup_hamming_threshold": 3,
    "simhash_bits": 64,
    "lsh_bands": 4,
}

# Scientific domain keywords for classification (Korean and English)
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "physics": [
        "물리", "힘", "에너지", "운동", "속도", "질량", "전자", "원자", "분자", "반응", "화합물",
        "파동", "광자", "중력", "입자", "핵", "방사능", "전기", "자기",
        "physics", "force", "energy", "motion", "speed", "mass", "electron", "atom", "molecule",
        "wave", "photon", "gravity", "particle", "nucleus", "radiation", "electric", "magnetic",
        "newton", "law", "velocity", "acceleration", "momentum"
    ],
    "biology": [
        "생물", "세포", "유전자", "단백질", "RNA", "DNA", "미생물", "생태", "진화",
        "대사", "호흡", "광합성", "가금류", "알", "난백", "난황", "생식", "번식",
        "유기체", "조직", "기관", "계통",
        "biology", "cell", "gene", "protein", "rna", "dna", "microorganism", "ecology", "evolution",
        "metabolism", "respiration", "photosynthesis", "organism", "tissue", "organ", "system",
        "divide", "reproduction", "mitosis", "meiosis"
    ],
    "chemistry": [
        "화학", "원소", "화합물", "반응", "결합", "용액", "산", "염기", "pH",
        "산화", "환원", "촉매", "분자", "원자", "이온", "결정", "용매", "용질",
        "침전", "증류"
    ],
    "astronomy": [
        "천문", "별", "행성", "은하", "우주", "태양", "달", "지구", "블랙홀",
        "혜성", "소행성", "성운", "은하수", "대폭발", "중력파"
    ],
    "geology": [
        "지질", "암석", "광물", "지층", "화산", "지진", "대륙", "판", "퇴적",
        "퇴적물", "지각", "맨틀", "핵", "광상", "지형"
    ],
    "mathematics": [
        "수학", "방정식", "계산", "확률", "통계", "기하", "대수", "미적분",
        "행렬", "벡터", "함수", "그래프", "극한", "적분"
    ],
    "general": [
        "과학", "연구", "실험", "관찰", "측정", "계산", "현상", "원리",
        "분석", "이론"
    ]
}

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
SCIENTIFIC_TERMS_BASE: List[str] = [
    # Core chemistry/physics/biology
    '원자', '분자', '세포', '유전자', '단백질', 'RNA', 'DNA', '화합물', '반응', '에너지', '힘',
    '운동', '속도', '가속도', '질량', '밀도', '전자', '양성자', '중성자', '원소', '결합', '공유결합', '이온결합',
    '용액', '용매', '용질', '농도', '몰농도', '산', '염기', 'pH', '산화', '환원', '촉매', '엔탈피', '엔트로피',
    '압력', '온도', '열', '열용량', '상전이', '기체', '액체', '고체', '평형', '속도론',
    # Physics
    '파동', '주파수', '진폭', '광자', '광속', '굴절', '반사', '간섭', '회절', '중력', '일', '전력', '전압', '전류', '저항',
    '자기장', '전기장', '쿨롱법칙', '로렌츠힘', '운동량', '충격량', '각운동량', '토크', '포텐셜', '보존법칙',
    # Astronomy & earth science
    '블랙홀', '행성', '별', '성단', '성운', '은하', '은하수', '우주', '우주배경복사', '적색편이', '허블법칙',
    '암석', '광물', '지층', '퇴적암', '화성암', '변성암', '화산', '마그마', '용암', '지진', '판구조론', '단층', '습곡',
    # Math & stats (common in MMLU/ARC)
    '방정식', '부등식', '함수', '도함수', '적분', '극한', '행렬', '벡터', '통계', '평균', '분산', '표준편차', '확률',
    '표본', '모수', '가설검정', '회귀',
    # Biology & medicine
    '미토콘드리아', '핵', '리보솜', '세포막', '세포벽', '염색체', '유전형', '표현형', '돌연변이', '분열', '감수분열',
    '번식', '생식', '효소', '대사', '경로', '호흡', '광합성', '삼투', '확산', '항원', '항체', '백신', '병원체',
    '영양소', '탄수화물', '지질', '아미노산', '핵산', '호르몬', '신경전달물질', '뉴런', '시냅스',
]

def _load_persistent_terms_from_conf() -> List[str]:
    """Load persistent curated terms from conf/scientific_terms.json if present.

    Returns a list if the file exists and contains a non-empty list; otherwise returns [].
    """
    conf_path = Path("conf") / "scientific_terms.json"
    try:
        if conf_path.exists():
            with conf_path.open("r", encoding="utf-8") as f:
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
#    - 'merge' -> ALWAYS merge base + dynamic artifact (hybrid approach)
# 2) If no mode set -> merge base + dynamic artifact (if any), otherwise use base
_mode = os.environ.get("SCIENTIFIC_TERMS_MODE", "merge").lower()
if _mode == "base_only":
    SCIENTIFIC_TERMS: List[str] = SCIENTIFIC_TERMS_BASE
elif _mode == "dynamic_only":
    SCIENTIFIC_TERMS = _load_scientific_terms_from_artifact()
else:  # merge mode (default) - ALWAYS merge for hybrid approach
    _dynamic_terms = _load_scientific_terms_from_artifact()
    if _dynamic_terms:
        # Hybrid approach: merge base terms with dynamic terms
        merged: List[str] = []
        seen: Set[str] = set()
        for t in (SCIENTIFIC_TERMS_BASE + _dynamic_terms):
            if t and t not in seen:
                seen.add(t)
                merged.append(t)
        SCIENTIFIC_TERMS = merged
        print(f"🔄 Hybrid mode: Merged {len(SCIENTIFIC_TERMS_BASE)} base + {len(_dynamic_terms)} dynamic = {len(merged)} total terms")
    else:
        SCIENTIFIC_TERMS = SCIENTIFIC_TERMS_BASE
        print(f"ℹ️ Hybrid mode: No dynamic terms found, using {len(SCIENTIFIC_TERMS_BASE)} base terms only")

# Query type classification patterns
QUERY_TYPE_PATTERNS: Dict[str, re.Pattern] = {
    "what": re.compile(r'\b(무엇|뭐|어떤|어떻게|왜|언제|어디|누구|얼마나)\b', re.IGNORECASE),
    "how": re.compile(r'\b(어떻게|방법|과정|절차|원리)\b', re.IGNORECASE),
    "why": re.compile(r'\b(왜|이유|원인|목적)\b', re.IGNORECASE),
    "when": re.compile(r'\b(언제|시기|기간|시간)\b', re.IGNORECASE),
    "where": re.compile(r'\b(어디|장소|위치|지역)\b', re.IGNORECASE),
    "calculate": re.compile(r'\b(계산|구하|값|수치)\b', re.IGNORECASE)
}

# Default K values for precision/recall calculations
DEFAULT_K_VALUES: List[int] = [1, 3, 5, 10]

# Analysis thresholds updated with profiling insights
ANALYSIS_THRESHOLDS: Dict[str, float] = {
    "map_score_low": 0.5,
    "retrieval_success_rate_low": 0.7,
    "rewrite_rate_high": 0.8,
    "rewrite_rate_low": 0.1,
    "low_relevance_threshold": 0.3,
    "query_complexity_change_threshold": 0.1,
    "significant_complexity_change": 0.2,

    # Dataset-specific thresholds based on profiling
    "arc_chunk_size_threshold": 256,      # ARC docs tend to be shorter
    "mmlu_chunk_size_threshold": 768,     # MMLU docs vary more in length
    "duplicate_score_penalty": 0.1,      # Penalty for exact duplicates
    "near_dup_score_penalty": 0.05,      # Lighter penalty for near-duplicates
    "src_boost_weight": 0.1,             # Conservative boosting weight
}

# Source-aware chunking recommendations based on profiling
CHUNKING_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "ko_ai2_arc": {
        "recommended_chunk_size": 256,
        "overlap_ratio": 0.1,
        "reasoning": "ARC challenges are typically shorter, focused questions"
    },
    "ko_mmlu_biology": {
        "recommended_chunk_size": 512,
        "overlap_ratio": 0.15,
        "reasoning": "Biology content varies; moderate chunks work well"
    },
    "ko_mmlu_physics": {
        "recommended_chunk_size": 512,
        "overlap_ratio": 0.15,
        "reasoning": "Physics often includes formulas; moderate overlap helps"
    },
    "ko_mmlu_chemistry": {
        "recommended_chunk_size": 384,
        "overlap_ratio": 0.2,
        "reasoning": "Chemistry includes complex reactions; higher overlap"
    },
    "ko_mmlu_medicine": {
        "recommended_chunk_size": 768,
        "overlap_ratio": 0.1,
        "reasoning": "Medical content tends to be longer and detailed"
    },
    "default": {
        "recommended_chunk_size": 512,
        "overlap_ratio": 0.1,
        "reasoning": "General fallback for unknown sources"
    }
}

# Parallel processing defaults
PARALLEL_PROCESSING_DEFAULTS: Dict[str, int] = {
    "max_workers_analysis": 16,
    "max_workers_query_analysis": 32,
    "max_workers_domain_generation": 6,
    "batch_size_threshold": 10,
    "validation_set_threshold": 5
}

# Language complexity assessment
LANGUAGE_COMPLEXITY_INDICATORS: List[str] = [
    '그리고', '그러나', '때문에', '따라서', '만약'
]

# Formula detection patterns
FORMULA_PATTERNS: List[str] = [
    r'\b[A-Z][a-z]?\d*\b',  # Chemical formulas like H2O, CO2
    r'\d+\s*[+\-*/=]\s*\d+',  # Mathematical expressions
    r'[a-zA-Z]\s*=\s*[^=]+',  # Equations
]

# Error analysis domain checks (simplified keywords for error categorization)
ERROR_ANALYSIS_DOMAIN_CHECKS: Dict[str, List[str]] = {
    "physics": ["물리", "힘", "에너지", "운동", "속도", "질량", "전자", "원자", "분자", "반응", "화합물", "파동", "광자", "중력", "입자", "핵", "방사능", "전기", "자기"],
    "biology": ["생물", "세포", "유전자", "단백질", "RNA", "DNA", "미생물", "생태", "진화", "대사", "호흡", "광합성", "가금류", "알", "난백", "난황", "생식", "번식", "유기체", "조직", "기관", "계통"],
    "chemistry": ["화학", "원소", "화합물", "반응", "결합", "용액", "산", "염기", "pH", "산화", "환원", "촉매", "분자", "원자", "이온", "결정", "용매", "용질", "침전", "증류"],
    "astronomy": ["천문", "별", "행성", "은하", "우주", "태양", "달", "지구", "블랙홀", "혜성", "소행성", "성운", "은하수", "대폭발", "중력파"],
    "geology": ["지질", "암석", "광물", "지층", "화산", "지진", "대륙", "판", "퇴적", "퇴적물", "지각", "맨틀", "핵", "광상", "지형"],
    "mathematics": ["수학", "방정식", "계산", "확률", "통계", "기하", "대수", "미적분", "행렬", "벡터", "함수", "그래프", "극한", "적분"],
    "general": ["과학", "연구", "실험", "관찰", "측정", "계산", "현상", "원리", "분석", "이론"]
}

# Enhanced error analysis thresholds with profiling insights
ERROR_ANALYSIS_THRESHOLDS: Dict[str, float] = {
    "ambiguous_query_threshold": 0.3,  # Low confidence in query understanding
    "out_of_domain_threshold": 0.1,    # Very few domain matches
    "complex_query_threshold": 0.7,    # High complexity score
    "false_positive_threshold": 0.8,   # High score but wrong result
    "false_negative_threshold": 0.2,   # Low score for correct result
    "ranking_error_threshold": 0.5,    # Ground truth not in top positions
    "temporal_degradation_threshold": 0.1,  # Performance drop over time
    "pattern_significance_threshold": 0.05,  # Statistical significance for patterns
    "domain_error_threshold": 0.15,    # Domain-specific error rate threshold

    # Source distribution-aware thresholds
    "arc_bias_threshold": 0.6,         # Alert if >60% results from ARC (bias detection)
    "mmlu_fragmentation_threshold": 0.8, # Alert if results too scattered across MMLU sources
    "source_diversity_threshold": 0.3,  # Minimum source diversity in top results
    "duplicate_contamination_threshold": 0.1,  # Alert if >10% results are duplicates
}

# Error category definitions for comprehensive analysis
ERROR_CATEGORIES: Dict[str, Dict[str, Any]] = {
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

# Pattern detection configurations with profiling insights
PATTERN_DETECTION_CONFIG: Dict[str, Any] = {
    "min_pattern_occurrences": 3,      # Minimum occurrences to consider a pattern
    "correlation_threshold": 0.6,      # Minimum correlation for pattern significance
    "temporal_window_days": 7,         # Time window for temporal analysis
    "domain_error_threshold": 0.15,    # Domain-specific error rate threshold
    "query_length_bins": [10, 20, 30, 50],  # Bins for query length analysis
    "complexity_score_bins": [0.2, 0.4, 0.6, 0.8],  # Bins for complexity analysis

    # Source distribution patterns from profiling
    "expected_arc_ratio": 0.48,        # Expected ARC representation in corpus
    "expected_mmlu_fragmentation": 0.52, # Expected MMLU distribution
    "source_balance_tolerance": 0.1,    # Acceptable deviation from expected ratios
    "duplicate_detection_threshold": 2, # Minimum cluster size for duplicate detection
}

# Query length normalization factors
QUERY_LENGTH_NORMALIZATION: Dict[str, float] = {
    "max_length_score": 100.0,
    "max_term_density": 5.0,
    "avg_word_length_factor": 10.0,
    "clause_count_factor": 3.0
}

# Validation set generation domains updated with actual source distribution
VALIDATION_DOMAINS: Dict[str, str] = {
    "physics": "물리학 (힘, 에너지, 운동, 원자, 입자 등) - ko_mmlu conceptual/college/high_school physics",
    "chemistry": "화학 (화합물, 반응, 원소, 산, 염기 등) - ko_mmlu college/high_school chemistry",
    "biology": "생물학 (세포, 유전자, 단백질, 생명, 진화 등) - ko_mmlu anatomy/biology/genetics/aging/nutrition/virology",
    "astronomy": "천문학 (별, 행성, 은하, 우주, 태양 등) - ko_mmlu astronomy",
    "computer_science": "컴퓨터과학 (알고리즘, 보안, 프로그래밍 등) - ko_mmlu computer_science/security",
    "medicine": "의학 (진단, 치료, 인체, 질병 등) - ko_mmlu college_medicine",
    "general_science": "일반과학 (과학적 사고, 실험, 관찰 등) - ko_ai2_arc ARC_Challenge",
    "global_facts": "일반상식 (사실, 지식, 정보 등) - ko_mmlu global_facts"
}

# Profiling artifacts reference paths (for runtime loading)
PROFILING_ARTIFACTS: Dict[str, str] = {
    "unique_sources": "unique_src.json",
    "source_counts": "src_counts.json",
    "keywords_per_src": "keywords_per_src.json",
    "stopwords_global": "stopwords_global.json",
    "per_src_stopwords": "per_src_stopwords.json",
    "duplicates": "duplicates.json",
    "near_duplicates": "near_duplicates.json",
    "per_src_length_stats": "per_src_length_stats.json",
    "field_presence": "field_presence.json",
    "summary": "summary.json"
}
