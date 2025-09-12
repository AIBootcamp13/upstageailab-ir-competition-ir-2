# src/ir_core/analysis/constants.py

"""
Constants and configuration values for the Scientific QA retrieval analysis module.

This module centralizes all hardcoded values, thresholds, and domain-specific
keywords to improve maintainability and reduce duplication across the codebase.
"""

from typing import Dict, List, Set
import re


# Scientific domain keywords for classification (Korean)
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "physics": [
        "물리", "힘", "에너지", "운동", "속도", "질량", "전자", "원자", "분자", "반응", "화합물",
        "파동", "광자", "중력", "입자", "핵", "방사능", "전기", "자기"
    ],
    "biology": [
        "생물", "세포", "유전자", "단백질", "RNA", "DNA", "미생물", "생태", "진화",
        "대사", "호흡", "광합성", "가금류", "알", "난백", "난황", "생식", "번식",
        "유기체", "조직", "기관", "계통"
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

# Scientific terms for complexity scoring and feature extraction
SCIENTIFIC_TERMS: List[str] = [
    '원자', '분자', '세포', '유전자', '단백질', 'RNA', 'DNA', '화합물', '반응', '에너지', '힘',
    '운동', '속도', '질량', '전자', '양성자', '중성자', '원소', '결합', '용액', '산', '염기', 'pH',
    '파동', '광자', '중력', '블랙홀', '행성', '별', '은하', '우주', '암석', '광물', '지층',
    '화산', '지진', '방정식', '확률', '통계', '미적분', '행렬', '대수', '기하'
]

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

# Analysis thresholds and defaults
ANALYSIS_THRESHOLDS: Dict[str, float] = {
    "map_score_low": 0.5,
    "retrieval_success_rate_low": 0.7,
    "rewrite_rate_high": 0.8,
    "rewrite_rate_low": 0.1,
    "low_relevance_threshold": 0.3,
    "query_complexity_change_threshold": 0.1,
    "significant_complexity_change": 0.2
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
    "physics": ["물리", "힘", "에너지"],
    "biology": ["생물", "세포", "유전자"],
    "chemistry": ["화학", "원소", "화합물"]
}

# Query length normalization factors
QUERY_LENGTH_NORMALIZATION: Dict[str, float] = {
    "max_length_score": 100.0,
    "max_term_density": 5.0,
    "avg_word_length_factor": 10.0,
    "clause_count_factor": 3.0
}

# Validation set generation domains
VALIDATION_DOMAINS: Dict[str, str] = {
    "physics": "물리학 (힘, 에너지, 운동, 원자, 입자 등)",
    "chemistry": "화학 (화합물, 반응, 원소, 산, 염기 등)",
    "biology": "생물학 (세포, 유전자, 단백질, 생명, 진화 등)",
    "astronomy": "천문학 (별, 행성, 은하, 우주, 태양 등)",
    "geology": "지질학 (암석, 광물, 지층, 화산, 지진 등)",
    "mathematics": "수학 (방정식, 계산, 확률, 통계, 기하 등)"
}
