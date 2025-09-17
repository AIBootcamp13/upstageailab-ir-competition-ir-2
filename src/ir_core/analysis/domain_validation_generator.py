# src/ir_core/analysis/domain_validation_generator.py

"""
Domain validation set generation functionality.

This module provides functionality to generate validation sets
for domain classification using LLM services.
"""

from typing import Dict, List, Any, Optional
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

from .constants import VALIDATION_DOMAINS, PARALLEL_PROCESSING_DEFAULTS
from .parallel_processor import ParallelProcessor


class DomainValidationGenerator:
    """
    Generates validation sets for domain classification using LLMs.
    """

    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize the domain validation generator.

        Args:
            config: Optional configuration
        """
        self.config = config or DictConfig({})
        self.parallel_processor = ParallelProcessor(
            max_workers=self.config.get('analysis', {}).get('max_workers', None),
            enable_parallel=self.config.get('analysis', {}).get('enable_parallel', True)
        )

    def create_domain_validation_set(
        self,
        num_queries_per_domain: int = 10,
        use_llm: bool = True,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a validation set for domain classification using LLM-generated queries.

        Args:
            num_queries_per_domain: Number of queries to generate per domain
            use_llm: Whether to use LLM for query generation
            max_workers: Maximum number of worker threads for parallel domain processing

        Returns:
            List[Dict[str, Any]]: Validation set with queries and expected domains
        """
        if not use_llm:
            return self._get_predefined_validation_queries()

        # Try Ollama first (local, cost-free)
        try:
            from ..utils.ollama_client import generate_validation_queries_ollama, OllamaClient
            client = OllamaClient()

            if client.check_health():
                print("Using local Ollama for validation set generation...")
                return self._generate_with_ollama(client, num_queries_per_domain, max_workers)
            else:
                print("Ollama not available, falling back to OpenAI...")

        except (ImportError, Exception) as e:
            print(f"Ollama failed ({type(e).__name__}), falling back to OpenAI...")

        # Fallback to OpenAI
        return self._generate_with_openai(num_queries_per_domain, max_workers)

    def _generate_with_ollama(
        self,
        client,
        num_queries_per_domain: int,
        max_workers: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Generate validation set using Ollama.

        Args:
            client: Ollama client instance
            num_queries_per_domain: Number of queries per domain
            max_workers: Maximum number of worker threads

        Returns:
            List[Dict[str, Any]]: Generated validation set
        """
        validation_set = []
        test_domains = VALIDATION_DOMAINS

        # Use parallel processing for domain query generation
        if len(test_domains) > 2 and max_workers != 0:
            if max_workers is None:
                max_workers = PARALLEL_PROCESSING_DEFAULTS["max_workers_domain_generation"]

            def generate_domain_queries(domain_item):
                domain, description = domain_item
                queries = self._generate_domain_queries_ollama(
                    domain, description, num_queries_per_domain, client
                )
                return domain, queries

            domain_items = list(test_domains.items())
            results = self.parallel_processor.process_batch(
                domain_items,
                generate_domain_queries,
                batch_threshold=2,
                operation_name="domains"
            )

            for domain, queries in results:
                for query in queries:
                    validation_set.append({
                        "query": query,
                        "expected_domain": [domain],  # Single domain for validation
                        "source": "ollama_generated"
                    })
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

    def _generate_with_openai(
        self,
        num_queries_per_domain: int,
        max_workers: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Generate validation set using OpenAI.

        Args:
            num_queries_per_domain: Number of queries per domain
            max_workers: Maximum number of worker threads

        Returns:
            List[Dict[str, Any]]: Generated validation set
        """
        import openai
        from openai import OpenAI

        client = OpenAI()
        validation_set = []
        test_domains = VALIDATION_DOMAINS

        # Use parallel processing for OpenAI domain generation
        if len(test_domains) > 2 and max_workers != 0:
            if max_workers is None:
                max_workers = min(PARALLEL_PROCESSING_DEFAULTS["max_workers_domain_generation"], len(test_domains))

            def generate_domain_queries(domain_item):
                domain, description = domain_item
                queries = self._generate_domain_queries_openai(
                    domain, description, num_queries_per_domain, client
                )
                return domain, queries

            domain_items = list(test_domains.items())
            results = self.parallel_processor.process_batch(
                domain_items,
                generate_domain_queries,
                batch_threshold=2,
                operation_name="domains"
            )

            for domain, queries in results:
                for query in queries:
                    validation_set.append({
                        "query": query,
                        "expected_domain": [domain],  # Single domain for validation
                        "source": "openai_generated"
                    })
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