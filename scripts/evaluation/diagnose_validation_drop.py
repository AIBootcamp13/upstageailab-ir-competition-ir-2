#!/usr/bin/env python3
# scripts/evaluation/diagnose_validation_drop.py

"""
Diagnostic script to analyze validation performance drop after constants.py updates.

This script compares feature extraction before and after the changes,
analyzes data distributions, and provides recommendations.
"""

import os
import sys
import json
from typing import Dict, List, Any
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ir_core.analysis.query_analyzer import QueryAnalyzer
from ir_core.analysis.constants import (
    SCIENTIFIC_TERMS, DOMAIN_KEYWORDS,
    SCIENTIFIC_TERMS_BASE
)
from ir_core.utils import read_jsonl


class ValidationDropDiagnoser:
    """Diagnoses causes of validation performance drop."""

    def __init__(self):
        self.analyzer = QueryAnalyzer()

    def load_validation_data(self, validation_path: str) -> List[Dict[str, Any]]:
        """Load validation data."""
        return list(read_jsonl(validation_path))

    def extract_validation_queries(self, validation_data: List[Dict[str, Any]]) -> List[str]:
        """Extract queries from validation data."""
        return [item.get("msg", [{}])[0].get("content", "") for item in validation_data]

    def analyze_feature_changes(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze how feature extraction changed."""
        print("🔍 Analyzing feature extraction changes...")

        results = []
        for query in queries:
            features = self.analyzer.analyze_query(query)
            results.append({
                'query': query,
                'scientific_terms': features.scientific_terms,
                'domain': features.domain,
                'complexity_score': features.complexity_score,
                'query_type': features.query_type
            })

        return {
            'total_queries': len(queries),
            'results': results,
            'domain_distribution': self._analyze_domain_distribution(results),
            'scientific_term_stats': self._analyze_scientific_terms(results),
            'complexity_stats': self._analyze_complexity(results)
        }

    def _analyze_domain_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Analyze domain distribution in validation set."""
        domain_counts = Counter()
        for result in results:
            for domain in result['domain']:
                domain_counts[domain] += 1
        return dict(domain_counts)

    def _analyze_scientific_terms(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze scientific term usage."""
        all_terms = []
        queries_with_terms = 0

        for result in results:
            terms = result['scientific_terms']
            if terms:
                queries_with_terms += 1
                all_terms.extend(terms)

        term_counts = Counter(all_terms)

        return {
            'total_unique_terms': len(term_counts),
            'queries_with_terms': queries_with_terms,
            'total_queries': len(results),
            'coverage_rate': queries_with_terms / len(results) if results else 0,
            'top_terms': dict(term_counts.most_common(10))
        }

    def _analyze_complexity(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze complexity score distribution."""
        complexities = [r['complexity_score'] for r in results]

        return {
            'mean': sum(complexities) / len(complexities) if complexities else 0,
            'min': min(complexities) if complexities else 0,
            'max': max(complexities) if complexities else 0,
            'distribution': self._create_histogram_bins(complexities)
        }

    def _create_histogram_bins(self, values: List[float], bins: int = 10) -> Dict[str, int]:
        """Create histogram bins for values."""
        if not values:
            return {}

        min_val, max_val = min(values), max(values)
        bin_size = (max_val - min_val) / bins

        histogram = defaultdict(int)
        for val in values:
            bin_idx = int((val - min_val) / bin_size)
            if bin_idx == bins:  # Handle edge case
                bin_idx = bins - 1
            bin_range = ".2f"
            histogram[bin_range] += 1

        return dict(histogram)

    def compare_with_base_terms(self, queries: List[str]) -> Dict[str, Any]:
        """Compare current terms vs base terms."""
        print("⚖️ Comparing current vs base scientific terms...")

        current_terms = set(SCIENTIFIC_TERMS)
        base_terms = set(SCIENTIFIC_TERMS_BASE)

        added_terms = current_terms - base_terms
        removed_terms = base_terms - current_terms
        common_terms = current_terms & base_terms

        # Test coverage on validation queries
        current_coverage = self._calculate_term_coverage(queries, list(current_terms))
        base_coverage = self._calculate_term_coverage(queries, list(base_terms))

        return {
            'current_total': len(current_terms),
            'base_total': len(base_terms),
            'added_terms': len(added_terms),
            'removed_terms': len(removed_terms),
            'common_terms': len(common_terms),
            'current_coverage': current_coverage,
            'base_coverage': base_coverage,
            'coverage_difference': current_coverage - base_coverage
        }

    def _calculate_term_coverage(self, queries: List[str], terms: List[str]) -> float:
        """Calculate what fraction of queries contain at least one term."""
        covered = 0
        for query in queries:
            query_lower = query.lower()
            if any(term in query_lower for term in terms):
                covered += 1
        return covered / len(queries) if queries else 0

    def analyze_domain_keywords(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze domain keyword effectiveness."""
        print("🏷️ Analyzing domain keyword effectiveness...")

        domain_stats = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = 0
            for query in queries:
                query_lower = query.lower()
                if any(keyword in query_lower for keyword in keywords):
                    matches += 1

            coverage = matches / len(queries) if queries else 0
            domain_stats[domain] = {
                'matches': matches,
                'coverage': coverage,
                'keyword_count': len(keywords)
            }

        return domain_stats

    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check scientific term coverage
        term_stats = analysis_results.get('scientific_term_stats', {})
        coverage_rate = term_stats.get('coverage_rate', 0)

        if coverage_rate < 0.3:
            recommendations.append(
                f"⚠️ LOW SCIENTIFIC TERM COVERAGE: Only {coverage_rate:.1%} of validation queries contain scientific terms. "
                "Consider reverting to base terms or expanding the term list."
            )

        # Check domain coverage
        domain_dist = analysis_results.get('domain_distribution', {})
        unknown_count = domain_dist.get('unknown', 0)
        total_queries = analysis_results.get('total_queries', 1)

        if unknown_count / total_queries > 0.4:
            recommendations.append(
                f"⚠️ HIGH UNKNOWN DOMAINS: {unknown_count/total_queries:.1%} of queries classified as 'unknown' domain. "
                "Domain keywords may be too restrictive."
            )

        # Check term comparison
        term_comparison = analysis_results.get('term_comparison', {})
        coverage_diff = term_comparison.get('coverage_difference', 0)

        if coverage_diff < -0.1:  # Coverage dropped by more than 10%
            recommendations.append(
                f"⚠️ COVERAGE DROP: Scientific term coverage decreased by {abs(coverage_diff):.1%} compared to base terms. "
                "Consider reverting profiling-based changes."
            )

        # General recommendations
        recommendations.extend([
            "🔧 CONSIDER REVERTING: If performance drop is significant, revert constants.py to pre-profiling state",
            "📊 CREATE NEW VALIDATION SET: Generate validation queries that match the profiled document distribution",
            "🎯 FINE-TUNE KEYWORDS: Adjust keywords based on validation set analysis rather than training data only",
            "⚖️ BALANCE APPROACH: Use hybrid approach - keep some profiling insights but maintain validation compatibility"
        ])

        return recommendations

    def run_full_diagnosis(self, validation_path: str) -> Dict[str, Any]:
        """Run complete diagnosis."""
        print("🚀 Starting validation performance drop diagnosis...")

        # Load data
        validation_data = self.load_validation_data(validation_path)
        queries = self.extract_validation_queries(validation_data)

        print(f"📋 Loaded {len(queries)} validation queries")

        # Run analyses
        feature_analysis = self.analyze_feature_changes(queries)
        term_comparison = self.compare_with_base_terms(queries)
        domain_analysis = self.analyze_domain_keywords(queries)

        # Combine results
        results = {
            'feature_analysis': feature_analysis,
            'term_comparison': term_comparison,
            'domain_analysis': domain_analysis
        }

        # Generate recommendations
        recommendations = self.generate_recommendations({
            **feature_analysis,
            'term_comparison': term_comparison
        })

        results['recommendations'] = recommendations

        return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose validation performance drop")
    parser.add_argument(
        "--validation-path",
        default="data/validation.jsonl",
        help="Path to validation data"
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/validation_diagnosis.json",
        help="Output path for diagnosis results"
    )

    args = parser.parse_args()

    # Create diagnoser
    diagnoser = ValidationDropDiagnoser()

    # Run diagnosis
    results = diagnoser.run_full_diagnosis(args.validation_path)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("📊 DIAGNOSIS SUMMARY")
    print("="*60)

    feature_analysis = results['feature_analysis']
    term_comparison = results['term_comparison']

    print(f"Total validation queries: {feature_analysis['total_queries']}")
    print(f"Scientific term coverage: {feature_analysis['scientific_term_stats']['coverage_rate']:.1%}")
    print(f"Unknown domain queries: {feature_analysis['domain_distribution'].get('unknown', 0)}")

    print(f"\nScientific terms comparison:")
    print(f"  Current terms: {term_comparison['current_total']}")
    print(f"  Base terms: {term_comparison['base_total']}")
    print(f"  Coverage difference: {term_comparison.get('coverage_difference', 0):.1%}")

    print(f"\n🔧 RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec}")

    print(f"\n📄 Full results saved to: {args.output}")


if __name__ == "__main__":
    main()