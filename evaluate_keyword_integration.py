#!/usr/bin/env python3
"""
Comparative Evaluation Script for Keyword-Enhanced Retrieval

This script compares the performance of:
1. Baseline RRF retrieval (original system)
2. Keyword-enhanced RRF retrieval (with curated keywords)

It uses the existing evaluation framework and provides detailed metrics comparison.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ir_core.retrieval.core import hybrid_retrieve
from ir_core.config import settings


class RetrievalEvaluator:
    """Evaluator for comparing retrieval performance"""

    def __init__(self, eval_data_path: str = "data/eval.jsonl"):
        self.eval_data_path = Path(eval_data_path)
        self.eval_data = self._load_eval_data()

    def _load_eval_data(self) -> List[Dict[str, Any]]:
        """Load evaluation data from JSONL file"""
        if not self.eval_data_path.exists():
            print(f"Warning: Evaluation data not found at {self.eval_data_path}")
            return []

        eval_data = []
        with open(self.eval_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    eval_data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")

        print(f"Loaded {len(eval_data)} evaluation queries")
        return eval_data

    def evaluate_retrieval(self, retrieval_function, name: str, max_queries: int = None) -> Dict[str, Any]:
        """Evaluate a retrieval function on the test set"""

        if not self.eval_data:
            return {"error": "No evaluation data available"}

        results = []
        total_time = 0.0

        eval_subset = self.eval_data[:max_queries] if max_queries else self.eval_data

        print(f"Evaluating {name} on {len(eval_subset)} queries...")

        for i, query_data in enumerate(eval_subset):
            query = query_data.get("query", "")
            if not query:
                continue

            print(f"Processing query {i+1}/{len(eval_subset)}: {query[:50]}...")

            # Time the retrieval
            start_time = time.time()
            try:
                hits = retrieval_function(query, bm25_k=20, rerank_k=10)
                retrieval_time = time.time() - start_time

                # Extract relevant information
                result = {
                    "query": query,
                    "retrieval_time": retrieval_time,
                    "num_results": len(hits),
                    "hits": []
                }

                # Process hits
                for hit in hits[:10]:  # Only keep top 10
                    hit_info = {
                        "docid": hit.get("_source", {}).get("docid", hit.get("_id", "")),
                        "score": hit.get("_score", 0.0),
                        "title": hit.get("_source", {}).get("title", "")[:100],  # Truncate long titles
                    }
                    result["hits"].append(hit_info)

                results.append(result)
                total_time += retrieval_time

            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "retrieval_time": 0.0,
                    "num_results": 0,
                    "hits": []
                })

        # Calculate aggregate metrics
        successful_queries = [r for r in results if "error" not in r]
        avg_time = total_time / len(successful_queries) if successful_queries else 0.0

        metrics = {
            "evaluator": name,
            "total_queries": len(eval_subset),
            "successful_queries": len(successful_queries),
            "failed_queries": len(eval_subset) - len(successful_queries),
            "average_retrieval_time": avg_time,
            "total_retrieval_time": total_time,
            "results": results
        }

        return metrics

    def calculate_hit_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate hit rate (percentage of queries with at least one result)"""
        if not results:
            return 0.0

        successful_results = [r for r in results if r.get("num_results", 0) > 0]
        return len(successful_results) / len(results)

    def print_report(self, metrics: Dict[str, Any]):
        """Print a formatted evaluation report"""
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {metrics.get('evaluator', 'Unknown')}")
        print(f"{'='*60}")

        print(f"Total Queries: {metrics.get('total_queries', 0)}")
        print(f"Successful: {metrics.get('successful_queries', 0)}")
        print(f"Failed: {metrics.get('failed_queries', 0)}")

        if metrics.get('successful_queries', 0) > 0:
            hit_rate = self.calculate_hit_rate(metrics.get('results', []))
            print(".1%")
            print(".3f")
            print(".3f")

        print(f"{'='*60}\n")


def baseline_retrieval(query: str, **kwargs) -> List[Dict[str, Any]]:
    """Baseline retrieval function (original RRF without keyword enhancement)"""
    # Temporarily disable curated keywords integration for baseline
    import ir_core.retrieval.core as core_module

    # Store original function
    original_build_query = core_module.build_flexible_match_query

    # Create baseline version that doesn't use curated keywords
    def baseline_build_flexible_match_query(query: str, size: int) -> dict:
        """Baseline version without curated keywords"""
        # Extract dynamic keywords from the query using LLM only
        dynamic_keywords = core_module._extract_keywords_from_query(query)

        # Build multi-field query with appropriate boosts (no curated keywords)
        bool_query = {
            "should": [
                # Highest boost for keywords field - direct keyword matches are very strong signals
                {"match": {"keywords": {"query": ' '.join(dynamic_keywords), "boost": 4.0}}},
                # High boost for hypothetical questions - these are phrased like user queries
                {"match": {"hypothetical_questions": {"query": query, "boost": 3.0}}},
                # Medium boost for summary - concise version of document content
                {"match": {"summary": {"query": query, "boost": 2.0}}},
                # Default boost for full content - comprehensive but less specific
                {"match": {"content": {"query": query, "boost": 1.0}}}
            ],
            "minimum_should_match": 1  # At least one field must match
        }

        # Add phrase matching for better precision on Korean queries
        if any('\uac00' <= char <= '\ud7a3' for char in query):
            bool_query["should"].append({
                "match_phrase": {
                    "content": {
                        "query": query,
                        "slop": 2,  # Allow some reordering
                        "boost": 1.5
                    }
                }
            }
        )

        return {
            "query": {
                "bool": bool_query
            },
            "size": size
        }

    # Temporarily replace the function
    core_module.build_flexible_match_query = baseline_build_flexible_match_query

    try:
        # Run retrieval with baseline function
        results = core_module.hybrid_retrieve(query, **kwargs)
        return results
    finally:
        # Restore original function
        core_module.build_flexible_match_query = original_build_query


def enhanced_retrieval(query: str, **kwargs) -> List[Dict[str, Any]]:
    """Enhanced retrieval function (with curated keywords)"""
    return hybrid_retrieve(query, **kwargs)


def compare_systems(max_queries: int = 50):
    """Compare baseline vs enhanced retrieval systems"""

    print("🔍 Starting Comparative Evaluation")
    print("=" * 60)

    evaluator = RetrievalEvaluator()

    # Evaluate baseline system
    print("\n📊 Evaluating BASELINE system (original RRF)...")
    baseline_metrics = evaluator.evaluate_retrieval(
        baseline_retrieval,
        "Baseline RRF",
        max_queries=max_queries
    )
    evaluator.print_report(baseline_metrics)

    # Evaluate enhanced system
    print("\n📊 Evaluating ENHANCED system (with curated keywords)...")
    enhanced_metrics = evaluator.evaluate_retrieval(
        enhanced_retrieval,
        "Enhanced RRF + Curated Keywords",
        max_queries=max_queries
    )
    evaluator.print_report(enhanced_metrics)

    # Compare results
    print("\n🔄 COMPARISON SUMMARY")
    print("=" * 60)

    baseline_time = baseline_metrics.get('average_retrieval_time', 0)
    enhanced_time = enhanced_metrics.get('average_retrieval_time', 0)

    baseline_hit_rate = evaluator.calculate_hit_rate(baseline_metrics.get('results', []))
    enhanced_hit_rate = evaluator.calculate_hit_rate(enhanced_metrics.get('results', []))

    print(".3f")
    print(".3f")
    print(".3f")

    if enhanced_time > 0 and baseline_time > 0:
        time_change = ((enhanced_time - baseline_time) / baseline_time) * 100
        print(".1f")

    # Save detailed results
    comparison_results = {
        "baseline": baseline_metrics,
        "enhanced": enhanced_metrics,
        "comparison": {
            "baseline_avg_time": baseline_time,
            "enhanced_avg_time": enhanced_time,
            "baseline_hit_rate": baseline_hit_rate,
            "enhanced_hit_rate": enhanced_hit_rate,
            "time_difference_percent": ((enhanced_time - baseline_time) / baseline_time) * 100 if baseline_time > 0 else 0
        }
    }

    output_file = Path("outputs/keyword_integration_comparison.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Detailed results saved to: {output_file}")
    print("✅ Comparative evaluation completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare retrieval systems with/without curated keywords")
    parser.add_argument("--max-queries", type=int, default=50, help="Maximum number of queries to evaluate")
    parser.add_argument("--eval-file", type=str, default="data/eval.jsonl", help="Path to evaluation data")

    args = parser.parse_args()

    # Update evaluator path if specified
    if args.eval_file != "data/eval.jsonl":
        evaluator = RetrievalEvaluator(args.eval_file)

    try:
        compare_systems(max_queries=args.max_queries)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()