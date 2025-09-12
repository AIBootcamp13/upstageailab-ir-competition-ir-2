# src/ir_core/analysis/query_analyzer.py

"""
Query analysis module for the Scientific QA retrieval system.

This module provides comprehensive analysis of query characteristics,
including complexity scoring, domain classification, query type detection,
and scientific term extraction.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


@dataclass
class QueryFeatures:
    """Detailed features extracted from a query."""
    length: int
    word_count: int
    scientific_terms: List[str]
    complexity_score: float
    query_type: str
    domain: List[str]  # Support multiple domains
    has_numbers: bool
    has_formulas: bool
    language_complexity: float


class QueryAnalyzer:
    """
    Advanced query analyzer for Scientific QA queries.

    Provides comprehensive analysis including complexity scoring,
    domain classification, query type detection, and scientific term extraction.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the query analyzer.

        Args:
            config: Optional configuration for analysis parameters
        """
        self.config = config or DictConfig({})

        # Parallel processing configuration
        self.max_workers = self.config.get('analysis', {}).get('max_workers', None)
        self.enable_parallel = self.config.get('analysis', {}).get('enable_parallel', True)

        # Scientific domain keywords (Korean)
        self.domain_keywords = {
            "physics": ["물리", "힘", "에너지", "운동", "속도", "질량", "전자", "원자", "분자", "반응", "화합물", "파동", "광자", "중력", "입자", "핵", "방사능", "전기", "자기"],
            "biology": ["생물", "세포", "유전자", "단백질", "RNA", "DNA", "미생물", "생태", "진화", "대사", "호흡", "광합성", "가금류", "알", "난백", "난황", "생식", "번식", "유기체", "조직", "기관", "계통"],
            "chemistry": ["화학", "원소", "화합물", "반응", "결합", "용액", "산", "염기", "pH", "산화", "환원", "촉매", "분자", "원자", "이온", "결정", "용매", "용질", "침전", "증류"],
            "astronomy": ["천문", "별", "행성", "은하", "우주", "태양", "달", "지구", "블랙홀", "혜성", "소행성", "성운", "은하수", "대폭발", "중력파"],
            "geology": ["지질", "암석", "광물", "지층", "화산", "지진", "대륙", "판", "퇴적", "퇴적물", "지각", "맨틀", "핵", "광상", "지형"],
            "mathematics": ["수학", "방정식", "계산", "확률", "통계", "기하", "대수", "미적분", "행렬", "벡터", "함수", "그래프", "극한", "적분"],
            "general": ["과학", "연구", "실험", "관찰", "측정", "계산", "현상", "원리", "분석", "이론"]
        }

        # Scientific terms for complexity scoring
        self.scientific_terms = [
            '원자', '분자', '세포', '유전자', '단백질', 'RNA', 'DNA', '화합물', '반응', '에너지', '힘', '운동',
            '속도', '질량', '전자', '양성자', '중성자', '원소', '결합', '용액', '산', '염기', 'pH',
            '파동', '광자', '중력', '블랙홀', '행성', '별', '은하', '우주', '암석', '광물', '지층',
            '화산', '지진', '방정식', '확률', '통계', '미적분', '행렬', '대수', '기하'
        ]

        # Query type patterns
        self.query_type_patterns = {
            "what": re.compile(r'\b(무엇|뭐|어떤|어떻게|왜|언제|어디|누구|얼마나)\b', re.IGNORECASE),
            "how": re.compile(r'\b(어떻게|방법|과정|절차|원리)\b', re.IGNORECASE),
            "why": re.compile(r'\b(왜|이유|원인|목적)\b', re.IGNORECASE),
            "when": re.compile(r'\b(언제|시기|기간|시간)\b', re.IGNORECASE),
            "where": re.compile(r'\b(어디|장소|위치|지역)\b', re.IGNORECASE),
            "calculate": re.compile(r'\b(계산|구하|값|수치)\b', re.IGNORECASE)
        }

    def analyze_query(self, query: str) -> QueryFeatures:
        """
        Analyze a single query and extract comprehensive features.

        Args:
            query: The query string to analyze

        Returns:
            QueryFeatures: Detailed analysis results
        """
        # Basic features
        length = len(query)
        words = query.split()
        word_count = len(words)

        # Scientific term extraction
        scientific_terms = self._extract_scientific_terms(query)

        # Complexity scoring
        complexity_score = self._calculate_complexity(query, scientific_terms)

        # Query type classification
        query_type = self._classify_query_type(query)

        # Domain classification
        domain = self._classify_domain(query)

        # Additional features
        has_numbers = bool(re.search(r'\d', query))
        has_formulas = self._detect_formulas(query)
        language_complexity = self._assess_language_complexity(query)

        return QueryFeatures(
            length=length,
            word_count=word_count,
            scientific_terms=scientific_terms,
            complexity_score=complexity_score,
            query_type=query_type,
            domain=domain,
            has_numbers=has_numbers,
            has_formulas=has_formulas,
            language_complexity=language_complexity
        )

    def analyze_batch(self, queries: List[str], max_workers: Optional[int] = None) -> List[QueryFeatures]:
        """
        Analyze a batch of queries with optional parallel processing.

        Args:
            queries: List of query strings
            max_workers: Maximum number of worker threads. If None, uses min(32, len(queries))

        Returns:
            List[QueryFeatures]: Analysis results for each query
        """
        if not queries:
            return []

        # Use parallel processing for batches larger than threshold
        if len(queries) > 10 and self.enable_parallel and max_workers != 0:  # Allow disabling with max_workers=0
            if max_workers is None:
                max_workers = self.max_workers or min(32, len(queries))

            print(f"🔄 Analyzing {len(queries)} queries using {max_workers} parallel workers...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all analysis tasks
                future_to_query = {executor.submit(self.analyze_query, query): query for query in queries}

                # Collect results in order
                results = []
                for future in as_completed(future_to_query):
                    try:
                        result = future.result()
                        results.append((future_to_query[future], result))
                    except Exception as e:
                        print(f"Error analyzing query: {e}")
                        # Return empty features for failed queries
                        results.append((future_to_query[future], QueryFeatures(
                            length=0, word_count=0, scientific_terms=[], complexity_score=0.0,
                            query_type="unknown", domain=["unknown"], has_numbers=False,
                            has_formulas=False, language_complexity=0.0
                        )))

                # Sort results back to original order
                results.sort(key=lambda x: queries.index(x[0]))
                return [result for _, result in results]
        else:
            # Use sequential processing for small batches
            return [self.analyze_query(query) for query in queries]

    def _extract_scientific_terms(self, query: str) -> List[str]:
        """
        Extract scientific terms from the query.

        Args:
            query: The query string

        Returns:
            List[str]: List of detected scientific terms
        """
        query_lower = query.lower()
        found_terms = []

        for term in self.scientific_terms:
            if term in query_lower:
                found_terms.append(term)

        return found_terms

    def _calculate_complexity(self, query: str, scientific_terms: List[str]) -> float:
        """
        Calculate query complexity score based on multiple factors.

        Args:
            query: The query string
            scientific_terms: List of scientific terms found in the query

        Returns:
            float: Complexity score between 0 and 1
        """
        # Length-based complexity (0-0.3)
        length_score = min(len(query) / 200, 0.3)

        # Scientific term density (0-0.4)
        words = query.split()
        term_density = len(scientific_terms) / len(words) if words else 0
        term_score = min(term_density * 2, 0.4)

        # Language complexity (0-0.3)
        lang_score = self._assess_language_complexity(query) * 0.3

        return length_score + term_score + lang_score

    def _classify_query_type(self, query: str) -> str:
        """
        Classify the query type based on question words and patterns.

        Args:
            query: The query string

        Returns:
            str: Query type (what, how, why, when, where, calculate, or general)
        """
        query_lower = query.lower()

        # Check for question words
        for qtype, pattern in self.query_type_patterns.items():
            if pattern.search(query):
                return qtype

        # Check for calculation indicators
        if any(term in query_lower for term in ['=', '계산', '구하라', '값은']):
            return "calculate"

        # Default to general
        return "general"

    def _classify_domain(self, query: str) -> List[str]:
        """
        Classify the query into scientific domains (supports multiple domains).

        Args:
            query: The query string

        Returns:
            List[str]: List of matched domains
        """
        query_lower = query.lower()
        matched_domains = []

        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                matched_domains.append(domain)

        # Remove duplicates and return
        return list(set(matched_domains)) if matched_domains else ["unknown"]

    def _detect_formulas(self, query: str) -> bool:
        """
        Detect if the query contains mathematical or chemical formulas.

        Args:
            query: The query string

        Returns:
            bool: True if formulas are detected
        """
        # Simple pattern for formulas (can be enhanced)
        formula_patterns = [
            r'\b[A-Z][a-z]?\d*\b',  # Chemical formulas like H2O, CO2
            r'\d+\s*[+\-*/=]\s*\d+',  # Mathematical expressions
            r'[a-zA-Z]\s*=\s*[^=]+',  # Equations
        ]

        for pattern in formula_patterns:
            if re.search(pattern, query):
                return True

        return False

    def _assess_language_complexity(self, query: str) -> float:
        """
        Assess the linguistic complexity of the query.

        Args:
            query: The query string

        Returns:
            float: Language complexity score (0-1)
        """
        words = query.split()

        if not words:
            return 0.0

        # Average word length (longer words suggest complexity)
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_score = min(avg_word_length / 10, 1.0)

        # Sentence structure complexity (presence of clauses)
        clause_indicators = ['그리고', '그러나', '때문에', '따라서', '만약', '이다']
        clause_count = sum(1 for indicator in clause_indicators if indicator in query)
        clause_score = min(clause_count / 3, 1.0)

        return (length_score + clause_score) / 2

    def measure_rewrite_effectiveness(
        self,
        original_query: str,
        rewritten_query: str
    ) -> Dict[str, Any]:
        """
        Measure the effectiveness of query rewriting.

        Args:
            original_query: The original query
            rewritten_query: The rewritten query

        Returns:
            Dict[str, Any]: Effectiveness metrics
        """
        if original_query == rewritten_query:
            return {
                "was_rewritten": False,
                "effectiveness_score": 0.0,
                "changes": []
            }

        # Analyze changes
        orig_features = self.analyze_query(original_query)
        rew_features = self.analyze_query(rewritten_query)

        changes = []

        # Length change
        length_diff = rew_features.length - orig_features.length
        if abs(length_diff) > 10:
            changes.append(f"length_change_{'increase' if length_diff > 0 else 'decrease'}")

        # Complexity change
        complexity_diff = rew_features.complexity_score - orig_features.complexity_score
        if abs(complexity_diff) > 0.1:
            changes.append(f"complexity_{'increase' if complexity_diff > 0 else 'decrease'}")

        # Domain change
        if orig_features.domain != rew_features.domain:
            changes.append("domain_change")

        # Query type change
        if orig_features.query_type != rew_features.query_type:
            changes.append("query_type_change")

        # Scientific terms change
        orig_terms = set(orig_features.scientific_terms)
        rew_terms = set(rew_features.scientific_terms)
        if orig_terms != rew_terms:
            changes.append("scientific_terms_change")

        # Calculate effectiveness score (simplified)
        effectiveness_score = 0.5  # Base score

        if changes:
            effectiveness_score += 0.2  # Some changes are good

        if abs(complexity_diff) > 0.2:
            effectiveness_score += 0.1  # Significant complexity change

        effectiveness_score = min(effectiveness_score, 1.0)

        return {
            "was_rewritten": True,
            "effectiveness_score": effectiveness_score,
            "changes": changes,
            "original_features": orig_features,
            "rewritten_features": rew_features
        }

    def create_domain_validation_set(
        self,
        num_queries_per_domain: int = 10,
        use_llm: bool = True,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a validation set for domain classification using LLM-generated queries with optional parallel processing.

        Args:
            num_queries_per_domain: Number of queries to generate per domain
            use_llm: Whether to use LLM for query generation
            max_workers: Maximum number of worker threads for parallel domain processing

        Returns:
            List[Dict[str, Any]]: Validation set with queries and expected domains
        """
        if not use_llm:
            # Fallback: use predefined queries
            return self._get_predefined_validation_queries()

        # Try Ollama first (local, cost-free)
        try:
            from ..utils.ollama_client import generate_validation_queries_ollama, OllamaClient
            client = OllamaClient()

            if client.check_health():
                print("Using local Ollama for validation set generation...")
                validation_set = []

                # Define domains to test (excluding 'general' and 'unknown')
                test_domains = {
                    "physics": "물리학 (힘, 에너지, 운동, 원자, 입자 등)",
                    "chemistry": "화학 (화합물, 반응, 원소, 산, 염기 등)",
                    "biology": "생물학 (세포, 유전자, 단백질, 생명, 진화 등)",
                    "astronomy": "천문학 (별, 행성, 은하, 우주, 태양 등)",
                    "geology": "지질학 (암석, 광물, 지층, 화산, 지진 등)",
                    "mathematics": "수학 (방정식, 계산, 확률, 통계, 기하 등)"
                }

                # Use parallel processing for domain query generation
                if len(test_domains) > 2 and max_workers != 0:
                    if max_workers is None:
                        max_workers = min(6, len(test_domains))  # One worker per domain

                    print(f"🔄 Generating queries for {len(test_domains)} domains using {max_workers} parallel workers...")

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit domain generation tasks
                        future_to_domain = {}
                        for domain, description in test_domains.items():
                            future = executor.submit(
                                self._generate_domain_queries_ollama,
                                domain, description, num_queries_per_domain, client
                            )
                            future_to_domain[future] = domain

                        # Collect results
                        for future in as_completed(future_to_domain):
                            domain = future_to_domain[future]
                            try:
                                queries = future.result()
                                for query in queries:
                                    validation_set.append({
                                        "query": query,
                                        "expected_domain": [domain],  # Single domain for validation
                                        "source": "ollama_generated"
                                    })
                            except Exception as e:
                                print(f"Error generating queries for {domain}: {e}")
                else:
                    # Sequential processing
                    for domain, description in test_domains.items():
                        print(f"Generating queries for {domain}...")
                        queries = self._generate_domain_queries_ollama(
                            domain, description, num_queries_per_domain, client
                        )
                        for query in queries:
                            validation_set.append({
                                "query": query,
                                "expected_domain": [domain],  # Single domain for validation
                                "source": "ollama_generated"
                            })

                return validation_set
            else:
                print("Ollama not available, falling back to OpenAI...")

        except (ImportError, Exception) as e:
            print(f"Ollama failed ({type(e).__name__}), falling back to OpenAI...")

        # Fallback to OpenAI
        import openai
        from openai import OpenAI

        client = OpenAI()

        validation_set = []

        # Define domains to test (excluding 'general' and 'unknown')
        test_domains = {
            "physics": "물리학 (힘, 에너지, 운동, 원자, 입자 등)",
            "chemistry": "화학 (화합물, 반응, 원소, 산, 염기 등)",
            "biology": "생물학 (세포, 유전자, 단백질, 생명, 진화 등)",
            "astronomy": "천문학 (별, 행성, 은하, 우주, 태양 등)",
            "geology": "지질학 (암석, 광물, 지층, 화산, 지진 등)",
            "mathematics": "수학 (방정식, 계산, 확률, 통계, 기하 등)"
        }

        # Use parallel processing for OpenAI domain generation
        if len(test_domains) > 2 and max_workers != 0:
            if max_workers is None:
                max_workers = min(6, len(test_domains))

            print(f"🔄 Generating queries for {len(test_domains)} domains using {max_workers} parallel workers (OpenAI)...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit domain generation tasks
                future_to_domain = {}
                for domain, description in test_domains.items():
                    future = executor.submit(
                        self._generate_domain_queries_openai,
                        domain, description, num_queries_per_domain, client
                    )
                    future_to_domain[future] = domain

                # Collect results
                for future in as_completed(future_to_domain):
                    domain = future_to_domain[future]
                    try:
                        queries = future.result()
                        for query in queries:
                            validation_set.append({
                                "query": query,
                                "expected_domain": [domain],  # Single domain for validation
                                "source": "openai_generated"
                            })
                    except Exception as e:
                        print(f"Error generating queries for {domain}: {e}")
        else:
            # Sequential processing
            for domain, description in test_domains.items():
                print(f"Generating queries for {domain}...")
                queries = self._generate_domain_queries_openai(
                    domain, description, num_queries_per_domain, client
                )
                for query in queries:
                    validation_set.append({
                        "query": query,
                        "expected_domain": [domain],  # Single domain for validation
                        "source": "openai_generated"
                    })

        return validation_set

    def _generate_domain_queries_ollama(
        self, domain: str, description: str, num_queries: int, client
    ) -> List[str]:
        """Generate queries for a domain using Ollama."""
        from ..utils.ollama_client import generate_validation_queries_ollama
        return generate_validation_queries_ollama(
            domain=domain,
            description=description,
            num_queries=num_queries,
            client=client
        )

    def _generate_domain_queries_openai(
        self, domain: str, description: str, num_queries: int, client
    ) -> List[str]:
        """Generate queries for a domain using OpenAI."""
        prompt = f"""
다음 과학 분야에 대한 한국어 질문을 {num_queries}개 생성해주세요: {description}

요구사항:
1. 각 질문은 해당 분야의 핵심 개념을 다루어야 합니다
2. 질문은 실제 과학 QA 시스템에서 볼 수 있는 형태여야 합니다
3. 각 질문은 10-30자 사이로 적당한 길이여야 합니다
4. 다양한 난이도의 질문을 포함하세요

형식: 각 줄에 하나의 질문만 작성하세요. 다른 텍스트는 포함하지 마세요.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )

        content = response.choices[0].message.content
        if content is None:
            print(f"Warning: No content generated for {domain}")
            return []

        queries = content.strip().split('\n')
        queries = [q.strip() for q in queries if q.strip()]
        return queries[:num_queries]

    def _get_predefined_validation_queries(self) -> List[Dict[str, Any]]:
        """Get predefined validation queries when LLM is not available."""
        return [
            {"query": "물체의 질량과 무게의 차이점은 무엇인가요?", "expected_domain": ["physics"]},
            {"query": "원자의 구조는 어떻게 되어 있나요?", "expected_domain": ["physics", "chemistry"]},
            {"query": "DNA 복제 과정은 어떻게 이루어지나요?", "expected_domain": ["biology"]},
            {"query": "화학 반응에서 촉매의 역할은 무엇인가요?", "expected_domain": ["chemistry"]},
            {"query": "태양계에서 가장 큰 행성은 무엇인가요?", "expected_domain": ["astronomy"]},
            {"query": "지진의 원인은 무엇인가요?", "expected_domain": ["geology"]},
            {"query": "피타고라스 정리는 무엇인가요?", "expected_domain": ["mathematics"]},
            {"query": "세포막의 기능은 무엇인가요?", "expected_domain": ["biology"]},
            {"query": "산과 염기의 차이점은 무엇인가요?", "expected_domain": ["chemistry"]},
            {"query": "빅뱅 이론은 무엇인가요?", "expected_domain": ["astronomy", "physics"]}
        ]

    def evaluate_domain_classification(
        self,
        validation_set: List[Dict[str, Any]],
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate domain classification accuracy against a validation set with optional parallel processing.

        Args:
            validation_set: List of queries with expected domains
            max_workers: Maximum number of worker threads for parallel processing

        Returns:
            Dict[str, Any]: Evaluation metrics and detailed results
        """
        if not validation_set:
            return {
                "total_queries": 0,
                "exact_match_accuracy": 0.0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_f1": 0.0,
                "domain_metrics": {},
                "detailed_results": []
            }

        print(f"🔍 Evaluating domain classification for {len(validation_set)} queries...")

        # Use parallel processing for larger validation sets
        if len(validation_set) > 5 and max_workers != 0:
            if max_workers is None:
                max_workers = min(16, len(validation_set))

            print(f"🔄 Using {max_workers} parallel workers for evaluation...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit evaluation tasks
                future_to_item = {}
                for item in validation_set:
                    future = executor.submit(self._evaluate_single_query, item)
                    future_to_item[future] = item

                # Collect results
                results = []
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        item = future_to_item[future]
                        print(f"Error evaluating query '{item['query']}': {e}")
                        # Add error result
                        results.append({
                            "query": item["query"],
                            "expected": item["expected_domain"],
                            "predicted": ["unknown"],
                            "correct": [],
                            "false_positives": [],
                            "false_negatives": item["expected_domain"],
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "exact_match": False
                        })
        else:
            # Sequential processing for small sets
            results = [self._evaluate_single_query(item) for item in validation_set]

        # Aggregate metrics (same as before)
        total_queries = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        avg_precision = sum(r["precision"] for r in results) / total_queries
        avg_recall = sum(r["recall"] for r in results) / total_queries
        avg_f1 = sum(r["f1"] for r in results) / total_queries

        # Per-domain metrics
        domain_metrics = {}
        for item in validation_set:
            for domain in item["expected_domain"]:
                if domain not in domain_metrics:
                    domain_metrics[domain] = {"total": 0, "correct": 0}

                domain_metrics[domain]["total"] += 1

                # Check if this domain was correctly predicted for this query
                query_result = next(r for r in results if r["query"] == item["query"])
                if domain in query_result["predicted"]:
                    domain_metrics[domain]["correct"] += 1

        for domain in domain_metrics:
            total = domain_metrics[domain]["total"]
            correct = domain_metrics[domain]["correct"]
            domain_metrics[domain]["accuracy"] = correct / total if total > 0 else 0

        return {
            "total_queries": total_queries,
            "exact_match_accuracy": exact_matches / total_queries,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "domain_metrics": domain_metrics,
            "detailed_results": results
        }

    def _evaluate_single_query(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query for domain classification.

        Args:
            item: Query item with expected domains

        Returns:
            Dict[str, Any]: Evaluation result for this query
        """
        query = item["query"]
        expected_domains = set(item["expected_domain"])

        # Get predicted domains
        features = self.analyze_query(query)
        predicted_domains = set(features.domain)

        # Calculate metrics
        correct_predictions = expected_domains.intersection(predicted_domains)
        false_positives = predicted_domains - expected_domains
        false_negatives = expected_domains - predicted_domains

        precision = len(correct_predictions) / len(predicted_domains) if predicted_domains else 0
        recall = len(correct_predictions) / len(expected_domains) if expected_domains else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "query": query,
            "expected": list(expected_domains),
            "predicted": list(predicted_domains),
            "correct": list(correct_predictions),
            "false_positives": list(false_positives),
            "false_negatives": list(false_negatives),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": expected_domains == predicted_domains
        }
