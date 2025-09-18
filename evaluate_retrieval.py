#!/usr/bin/env python3
"""
Comprehensive Retrieval Evaluation Script

This script evaluates the retrieval performance of the RAG system using the eval dataset.
It measures various metrics including:
- Retrieval accuracy (top-k hit rates)
- Score distributions
- Performance comparison between different retrieval methods
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ir_core.retrieval.core import hybrid_retrieve, sparse_retrieve
from ir_core.config import settings


@dataclass
class RetrievalResult:
    """Container for a single retrieval result"""
    eval_id: int
    query: str
    retrieved_docs: List[Dict[str, Any]]
    execution_time: float
    method: str


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    total_queries: int
    avg_execution_time: float
    top1_hit_rate: float
    top3_hit_rate: float
    top5_hit_rate: float
    avg_retrieved_docs: float
    score_distributions: Dict[str, List[float]]


class RetrievalEvaluator:
    """Comprehensive evaluator for retrieval performance"""

    def __init__(self, eval_file: str = "data/eval.jsonl"):
        self.eval_file = Path(eval_file)
        self.eval_data = self._load_eval_data()

    def _load_eval_data(self) -> List[Dict[str, Any]]:
        """Load evaluation data from JSONL file"""
        eval_data = []
        with open(self.eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line))
        return eval_data

    def _extract_query_from_eval(self, eval_item: Dict[str, Any]) -> str:
        """Extract the last user query from eval item"""
        messages = eval_item.get('msg', [])
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if user_messages:
            return user_messages[-1]['content']
        return ""

    def evaluate_method(self, method: str = "hybrid", top_k: int = 5) -> List[RetrievalResult]:
        """Evaluate a specific retrieval method"""
        results = []

        print(f"Evaluating {method} retrieval on {len(self.eval_data)} queries...")

        for i, eval_item in enumerate(self.eval_data):
            eval_id = eval_item['eval_id']
            query = self._extract_query_from_eval(eval_item)

            if not query:
                continue

            start_time = time.time()

            try:
                if method == "hybrid":
                    retrieved_docs = hybrid_retrieve(query, rerank_k=top_k)
                elif method == "sparse":
                    retrieved_docs = sparse_retrieve(query, size=top_k)
                else:
                    raise ValueError(f"Unknown method: {method}")
            except Exception as e:
                print(f"Error retrieving for eval_id {eval_id}: {e}")
                retrieved_docs = []

            execution_time = time.time() - start_time

            result = RetrievalResult(
                eval_id=eval_id,
                query=query,
                retrieved_docs=retrieved_docs,
                execution_time=execution_time,
                method=method
            )
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(self.eval_data)} queries")

        return results

    def calculate_metrics(self, results: List[RetrievalResult]) -> EvaluationMetrics:
        """Calculate comprehensive metrics from results"""
        if not results:
            return EvaluationMetrics(0, 0, 0, 0, 0, 0, {})

        total_queries = len(results)
        execution_times = [r.execution_time for r in results]

        # For now, we'll use a simple heuristic for "hits" - documents with score > 0
        # In a real evaluation, you'd need ground truth relevance labels
        top1_hits = sum(1 for r in results if r.retrieved_docs and len(r.retrieved_docs) > 0)
        top3_hits = sum(1 for r in results if r.retrieved_docs and len(r.retrieved_docs) >= 3)
        top5_hits = sum(1 for r in results if r.retrieved_docs and len(r.retrieved_docs) >= 5)

        avg_docs = np.mean([len(r.retrieved_docs) for r in results])

        # Collect score distributions
        score_distributions = defaultdict(list)
        for result in results:
            for doc in result.retrieved_docs:
                if 'score' in doc:
                    score_distributions['final_scores'].append(doc['score'])
                if 'rrf_score' in doc:
                    score_distributions['rrf_scores'].append(doc['rrf_score'])
                if '_score' in doc:
                    score_distributions['es_scores'].append(doc['_score'])
                if 'sparse_score' in doc:
                    score_distributions['sparse_scores'].append(doc['sparse_score'])
                if 'dense_score' in doc:
                    score_distributions['dense_scores'].append(doc['dense_score'])

        return EvaluationMetrics(
            total_queries=total_queries,
            avg_execution_time=float(np.mean(execution_times)),
            top1_hit_rate=top1_hits / total_queries,
            top3_hit_rate=top3_hits / total_queries,
            top5_hit_rate=top5_hits / total_queries,
            avg_retrieved_docs=float(avg_docs),
            score_distributions=dict(score_distributions)
        )

    def print_report(self, metrics: EvaluationMetrics, method: str):
        """Print a comprehensive evaluation report"""
        print(f"\n{'='*60}")
        print(f"RETRIEVAL EVALUATION REPORT - {method.upper()}")
        print(f"{'='*60}")

        print(f"Total Queries: {metrics.total_queries}")
        print(".3f")
        print(".1%")
        print(".1%")
        print(".1%")
        print(".1f")

        print(f"\nSCORE DISTRIBUTIONS:")
        for score_type, scores in metrics.score_distributions.items():
            if scores:
                print(f"  {score_type}:")
                print(f"    Mean: {np.mean(scores):.4f}")
                print(f"    Std:  {np.std(scores):.4f}")
                print(f"    Min:  {np.min(scores):.4f}")
                print(f"    Max:  {np.max(scores):.4f}")
                print(f"    Count: {len(scores)}")

    def save_results(self, results: List[RetrievalResult], output_file: str):
        """Save detailed results to JSON file"""
        output_data = []
        for result in results:
            result_dict = {
                'eval_id': result.eval_id,
                'query': result.query,
                'method': result.method,
                'execution_time': result.execution_time,
                'num_retrieved': len(result.retrieved_docs),
                'retrieved_docs': [
                    {
                        'docid': doc.get('_source', {}).get('docid', doc.get('_id', '')),
                        'score': doc.get('score', 0),
                        'rrf_score': doc.get('rrf_score', 0),
                        'es_score': doc.get('_score', 0),
                        'sparse_score': doc.get('sparse_score', 0),
                        'dense_score': doc.get('dense_score', 0),
                        'content_preview': doc.get('_source', {}).get('content', '')[:200] + '...'
                    }
                    for doc in result.retrieved_docs
                ]
            }
            output_data.append(result_dict)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {output_file}")


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument("--method", choices=["hybrid", "sparse"], default="hybrid",
                       help="Retrieval method to evaluate")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of documents to retrieve")
    parser.add_argument("--output", type=str,
                       help="Output file for detailed results")
    parser.add_argument("--eval-file", type=str, default="data/eval.jsonl",
                       help="Path to evaluation data file")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RetrievalEvaluator(args.eval_file)

    # Run evaluation
    results = evaluator.evaluate_method(args.method, args.top_k)

    # Calculate metrics
    metrics = evaluator.calculate_metrics(results)

    # Print report
    evaluator.print_report(metrics, args.method)

    # Save results if requested
    if args.output:
        evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()