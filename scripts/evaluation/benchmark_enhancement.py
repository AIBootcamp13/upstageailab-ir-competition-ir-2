#!/usr/bin/env python3
# scripts/evaluation/benchmark_enhancement.py

"""
Benchmark script for query enhancement techniques.

This script evaluates the performance of different query enhancement techniques
by comparing retrieval quality, response time, and other metrics.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

import pandas as pd
from tqdm import tqdm

# Add src to path
scripts_dir = os.path.dirname(__file__)
repo_dir = os.path.dirname(scripts_dir)
src_dir = os.path.join(repo_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ir_core.query_enhancement.manager import QueryEnhancementManager
from ir_core.retrieval.core import hybrid_retrieve
from ir_core.config import settings


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    query: str
    technique: str
    enhanced_query: str
    retrieval_time: float
    num_results: int
    avg_score: float
    top_score: float
    confidence: float
    error: Optional[str] = None


@dataclass
class TechniqueMetrics:
    """Aggregated metrics for a technique."""
    technique: str
    total_queries: int
    successful_enhancements: int
    avg_retrieval_time: float
    avg_num_results: float
    avg_score: float
    avg_top_score: float
    avg_confidence: float
    error_rate: float


class QueryEnhancementBenchmarker:
    """Benchmarker for query enhancement techniques."""

    def __init__(self, max_workers: int = 4):
        """Initialize the benchmarker."""
        self.max_workers = max_workers
        self.enhancement_manager = QueryEnhancementManager()
        self.logger = logging.getLogger(__name__)

    def load_test_queries(self, file_path: str, limit: Optional[int] = None) -> List[str]:
        """Load test queries from a JSONL file."""
        queries = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    try:
                        data = json.loads(line.strip())
                        if 'query' in data:
                            queries.append(data['query'])
                        elif 'question' in data:
                            queries.append(data['question'])
                    except json.JSONDecodeError:
                        continue

            self.logger.info(f"Loaded {len(queries)} test queries from {file_path}")
            return queries

        except FileNotFoundError:
            self.logger.error(f"Test queries file not found: {file_path}")
            return []

    def benchmark_technique(
        self,
        queries: List[str],
        technique: str,
        progress_callback: Optional[callable] = None
    ) -> List[BenchmarkResult]:
        """Benchmark a specific technique on a set of queries."""

        results = []

        def process_query(query: str) -> BenchmarkResult:
            """Process a single query with the given technique."""
            try:
                # Enhance query
                start_time = time.time()
                enhancement_result = self.enhancement_manager.enhance_query(query, technique=technique)
                enhancement_time = time.time() - start_time

                if not enhancement_result.get('enhanced', False):
                    return BenchmarkResult(
                        query=query,
                        technique=technique,
                        enhanced_query=query,
                        retrieval_time=0.0,
                        num_results=0,
                        avg_score=0.0,
                        top_score=0.0,
                        confidence=0.0,
                        error="Enhancement failed"
                    )

                enhanced_query = enhancement_result.get('enhanced_query', query)

                # Perform retrieval
                retrieval_start = time.time()
                retrieval_results = hybrid_retrieve(query=enhanced_query, rerank_k=10)
                retrieval_time = time.time() - retrieval_start

                # Calculate metrics
                if retrieval_results:
                    scores = [r.get('score', 0.0) for r in retrieval_results]
                    avg_score = sum(scores) / len(scores)
                    top_score = max(scores)
                    num_results = len(retrieval_results)
                else:
                    avg_score = 0.0
                    top_score = 0.0
                    num_results = 0

                confidence = enhancement_result.get('confidence', 0.0)

                return BenchmarkResult(
                    query=query,
                    technique=technique,
                    enhanced_query=enhanced_query,
                    retrieval_time=retrieval_time,
                    num_results=num_results,
                    avg_score=avg_score,
                    top_score=top_score,
                    confidence=confidence
                )

            except Exception as e:
                self.logger.error(f"Error processing query '{query}' with technique '{technique}': {e}")
                return BenchmarkResult(
                    query=query,
                    technique=technique,
                    enhanced_query=query,
                    retrieval_time=0.0,
                    num_results=0,
                    avg_score=0.0,
                    top_score=0.0,
                    confidence=0.0,
                    error=str(e)
                )

        # Process queries with progress tracking
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_query, query) for query in queries]

            for future in tqdm(as_completed(futures), total=len(queries), desc=f"Benchmarking {technique}"):
                result = future.result()
                results.append(result)

                if progress_callback:
                    progress_callback(result)

        return results

    def benchmark_all_techniques(
        self,
        queries: List[str],
        techniques: Optional[List[str]] = None,
        include_baseline: bool = True
    ) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark all available techniques."""

        if techniques is None:
            techniques = self.enhancement_manager.get_available_techniques()

        if include_baseline:
            techniques = ['baseline'] + techniques

        results = {}

        for technique in techniques:
            self.logger.info(f"Benchmarking technique: {technique}")

            if technique == 'baseline':
                # Baseline: no enhancement
                baseline_results = []
                for query in tqdm(queries, desc="Benchmarking baseline"):
                    try:
                        retrieval_start = time.time()
                        retrieval_results = hybrid_retrieve(query=query, rerank_k=10)
                        retrieval_time = time.time() - retrieval_start

                        if retrieval_results:
                            scores = [r.get('score', 0.0) for r in retrieval_results]
                            avg_score = sum(scores) / len(scores)
                            top_score = max(scores)
                            num_results = len(retrieval_results)
                        else:
                            avg_score = 0.0
                            top_score = 0.0
                            num_results = 0

                        baseline_results.append(BenchmarkResult(
                            query=query,
                            technique='baseline',
                            enhanced_query=query,
                            retrieval_time=retrieval_time,
                            num_results=num_results,
                            avg_score=avg_score,
                            top_score=top_score,
                            confidence=1.0  # Baseline always "succeeds"
                        ))
                    except Exception as e:
                        baseline_results.append(BenchmarkResult(
                            query=query,
                            technique='baseline',
                            enhanced_query=query,
                            retrieval_time=0.0,
                            num_results=0,
                            avg_score=0.0,
                            top_score=0.0,
                            confidence=0.0,
                            error=str(e)
                        ))

                results['baseline'] = baseline_results
            else:
                # Enhanced techniques
                technique_results = self.benchmark_technique(queries, technique)
                results[technique] = technique_results

        return results

    def calculate_metrics(self, results: List[BenchmarkResult]) -> TechniqueMetrics:
        """Calculate aggregated metrics for a technique."""

        if not results:
            return TechniqueMetrics(
                technique="empty",
                total_queries=0,
                successful_enhancements=0,
                avg_retrieval_time=0.0,
                avg_num_results=0.0,
                avg_score=0.0,
                avg_top_score=0.0,
                avg_confidence=0.0,
                error_rate=1.0
            )

        total_queries = len(results)
        successful_results = [r for r in results if r.error is None]

        successful_enhancements = len(successful_results)
        error_rate = 1.0 - (successful_enhancements / total_queries)

        if successful_results:
            avg_retrieval_time = sum(r.retrieval_time for r in successful_results) / len(successful_results)
            avg_num_results = sum(r.num_results for r in successful_results) / len(successful_results)
            avg_score = sum(r.avg_score for r in successful_results) / len(successful_results)
            avg_top_score = sum(r.top_score for r in successful_results) / len(successful_results)
            avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
        else:
            avg_retrieval_time = 0.0
            avg_num_results = 0.0
            avg_score = 0.0
            avg_top_score = 0.0
            avg_confidence = 0.0

        return TechniqueMetrics(
            technique=results[0].technique if results else "unknown",
            total_queries=total_queries,
            successful_enhancements=successful_enhancements,
            avg_retrieval_time=avg_retrieval_time,
            avg_num_results=avg_num_results,
            avg_score=avg_score,
            avg_top_score=avg_top_score,
            avg_confidence=avg_confidence,
            error_rate=error_rate
        )

    def save_results(self, results: Dict[str, List[BenchmarkResult]], output_dir: str):
        """Save benchmark results to files."""

        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        for technique, technique_results in results.items():
            filename = f"{technique}_results.jsonl"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                for result in technique_results:
                    f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')

        # Save aggregated metrics
        metrics = {}
        for technique, technique_results in results.items():
            metrics[technique] = asdict(self.calculate_metrics(technique_results))

        metrics_filepath = os.path.join(output_dir, "metrics.json")
        with open(metrics_filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Save comparison table
        comparison_data = []
        for technique, metric_data in metrics.items():
            comparison_data.append({
                'Technique': technique,
                'Success Rate': f"{(1 - metric_data['error_rate']) * 100:.1f}%",
                'Avg Retrieval Time': f"{metric_data['avg_retrieval_time']:.3f}s",
                'Avg Results': f"{metric_data['avg_num_results']:.1f}",
                'Avg Score': f"{metric_data['avg_score']:.3f}",
                'Top Score': f"{metric_data['avg_top_score']:.3f}",
                'Confidence': f"{metric_data['avg_confidence']:.3f}"
            })

        df = pd.DataFrame(comparison_data)
        csv_filepath = os.path.join(output_dir, "comparison.csv")
        df.to_csv(csv_filepath, index=False)

        self.logger.info(f"Results saved to {output_dir}")


def main():
    """Main entry point for the benchmarking script."""
    parser = argparse.ArgumentParser(description="Benchmark query enhancement techniques")
    parser.add_argument(
        "--queries-file",
        type=str,
        default="data/eval.jsonl",
        help="Path to JSONL file containing test queries"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reports/enhancement_benchmark",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--techniques",
        type=str,
        nargs="+",
        help="Specific techniques to benchmark (default: all available)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50,
        help="Maximum number of queries to test"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        default=True,
        help="Include baseline (no enhancement) in benchmark"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create benchmarker
    benchmarker = QueryEnhancementBenchmarker(max_workers=args.max_workers)

    # Load test queries
    queries = benchmarker.load_test_queries(args.queries_file, limit=args.max_queries)
    if not queries:
        print(f"No queries found in {args.queries_file}")
        return 1

    print(f"Starting benchmark with {len(queries)} queries...")

    # Run benchmark
    results = benchmarker.benchmark_all_techniques(
        queries=queries,
        techniques=args.techniques,
        include_baseline=args.include_baseline
    )

    # Save results
    benchmarker.save_results(results, args.output_dir)

    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 50)

    for technique, technique_results in results.items():
        metrics = benchmarker.calculate_metrics(technique_results)
        print(f"\n{technique.upper()}:")
        print(f"  Success Rate: {(1 - metrics.error_rate) * 100:.1f}%")
        print(f"  Avg Retrieval Time: {metrics.avg_retrieval_time:.3f}s")
        print(f"  Avg Results: {metrics.avg_num_results:.1f}")
        print(f"  Avg Score: {metrics.avg_score:.3f}")
        print(f"  Top Score: {metrics.avg_top_score:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())