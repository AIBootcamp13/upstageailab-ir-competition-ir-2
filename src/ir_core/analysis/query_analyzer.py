# src/ir_core/analysis/query_analyzer.py

"""
Query analysis module for the Scientific QA retrieval system.

This module provides comprehensive analysis of query characteristics,
including complexity scoring, domain classification, query type detection,
and scientific term extraction.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .constants import (
    DOMAIN_KEYWORDS,
    SCIENTIFIC_TERMS,
    QUERY_TYPE_PATTERNS,
    ANALYSIS_THRESHOLDS,
    PARALLEL_PROCESSING_DEFAULTS,
    LANGUAGE_COMPLEXITY_INDICATORS,
    FORMULA_PATTERNS,
    QUERY_LENGTH_NORMALIZATION,
    VALIDATION_DOMAINS
)
from .query_components import QueryFeatures, BatchQueryProcessor


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
        self.batch_processor = BatchQueryProcessor(config)

        # Parallel processing configuration
        self.max_workers = self.config.get('analysis', {}).get('max_workers', None)
        self.enable_parallel = self.config.get('analysis', {}).get('enable_parallel', True)

    def analyze_query(self, query: str) -> QueryFeatures:
        """
        Analyze a single query and extract comprehensive features.

        Args:
            query: The query string to analyze

        Returns:
            QueryFeatures: Detailed analysis results
        """
        return self.batch_processor.feature_extractor.extract_features(query)

    def analyze_batch(self, queries: List[str], max_workers: Optional[int] = None) -> List[QueryFeatures]:
        """
        Analyze a batch of queries with optional parallel processing.

        Args:
            queries: List of query strings
            max_workers: Maximum number of worker threads. If None, uses min(32, len(queries))

        Returns:
            List[QueryFeatures]: Analysis results for each query
        """
        return self.batch_processor.process_batch(queries, max_workers)


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
        if abs(complexity_diff) > ANALYSIS_THRESHOLDS["query_complexity_change_threshold"]:
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

        if abs(complexity_diff) > ANALYSIS_THRESHOLDS["significant_complexity_change"]:
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
                test_domains = VALIDATION_DOMAINS

                # Use parallel processing for domain query generation
                if len(test_domains) > 2 and max_workers != 0:
                    if max_workers is None:
                        max_workers = PARALLEL_PROCESSING_DEFAULTS["max_workers_domain_generation"]
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
        test_domains = VALIDATION_DOMAINS

        # Use parallel processing for OpenAI domain generation
        if len(test_domains) > 2 and max_workers != 0:
            if max_workers is None:
                max_workers = min(PARALLEL_PROCESSING_DEFAULTS["max_workers_domain_generation"], len(test_domains))

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
                max_workers = PARALLEL_PROCESSING_DEFAULTS["max_workers_analysis"]

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