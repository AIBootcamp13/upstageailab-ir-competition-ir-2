#!/usr/bin/env python3
# scripts/evaluation/create_balanced_validation.py

"""
Create a balanced validation set that matches the document distribution.

This script analyzes the document distribution and creates validation queries
that better represent the actual data distribution.
"""

import os
import sys
import json
import random
from typing import Dict, List, Any
from pathlib import Path
from collections import Counter, defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ir_core.utils import read_jsonl


class BalancedValidationCreator:
    """Creates balanced validation sets based on document distribution."""

    def __init__(self):
        self.documents = []
        self.source_distribution = {}
        self.domain_distribution = {}

    def load_documents(self, documents_path: str):
        """Load and analyze document distribution."""
        print(f"📚 Loading documents from {documents_path}")
        self.documents = list(read_jsonl(documents_path))

        # Analyze source distribution
        sources = [doc.get('src', 'unknown') for doc in self.documents]
        self.source_distribution = Counter(sources)

        print(f"📊 Loaded {len(self.documents)} documents")
        print(f"🔍 Found {len(self.source_distribution)} unique sources")

        # Show top sources
        print("\n📈 Top 10 sources by document count:")
        for src, count in self.source_distribution.most_common(10):
            print(f"  {src}: {count} docs ({count/len(self.documents)*100:.1f}%)")

    def analyze_domain_distribution(self):
        """Analyze domain distribution based on source mapping."""
        from ir_core.analysis.constants import DATASET_SOURCES

        domain_counts = defaultdict(int)

        for src, count in self.source_distribution.items():
            # Find which domain this source belongs to
            for domain, sources in DATASET_SOURCES.items():
                if src in sources:
                    domain_counts[domain] += count
                    break
            else:
                domain_counts['unknown'] += count

        self.domain_distribution = dict(domain_counts)

        print("\n🏷️ Domain distribution:")
        total_docs = sum(domain_counts.values())
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {count} docs ({count/total_docs*100:.1f}%)")

        return self.domain_distribution

    def generate_validation_queries(self, num_queries: int = 100) -> List[Dict[str, Any]]:
        """Generate balanced validation queries."""
        print(f"\n🎯 Generating {num_queries} balanced validation queries...")

        # Calculate target queries per domain
        domain_targets = {}
        total_docs = sum(self.domain_distribution.values())

        for domain, doc_count in self.domain_distribution.items():
            # Proportional allocation with minimum of 5 queries per domain
            target = max(5, int(num_queries * (doc_count / total_docs)))
            domain_targets[domain] = target

        print("🎯 Target queries per domain:")
        for domain, target in domain_targets.items():
            print(f"  {domain}: {target}")

        # Get documents per domain
        domain_docs = defaultdict(list)
        from ir_core.analysis.constants import DATASET_SOURCES

        for doc in self.documents:
            src = doc.get('src', 'unknown')
            domain_found = False

            for domain, sources in DATASET_SOURCES.items():
                if src in sources:
                    domain_docs[domain].append(doc)
                    domain_found = True
                    break

            if not domain_found:
                domain_docs['unknown'].append(doc)

        # Generate queries for each domain
        validation_queries = []

        for domain, target_count in domain_targets.items():
            docs = domain_docs.get(domain, [])
            if not docs:
                print(f"⚠️ No documents found for domain {domain}")
                continue

            # Sample documents for this domain
            sampled_docs = random.sample(docs, min(target_count, len(docs)))

            for doc in sampled_docs:
                # Generate query from document content
                query = self._generate_query_from_document(doc, domain)
                if query:
                    validation_queries.append({
                        'eval_id': f'balanced_{domain}_{len(validation_queries):03d}',
                        'msg': [{'role': 'user', 'content': query}],
                        'ground_truth_doc_id': doc.get('docid', ''),
                        'domain': domain,
                        'source': doc.get('src', 'unknown')
                    })

        # Shuffle to avoid domain clustering
        random.shuffle(validation_queries)

        print(f"✅ Generated {len(validation_queries)} validation queries")
        return validation_queries

    def _generate_query_from_document(self, doc: Dict[str, Any], domain: str) -> str:
        """Generate a query from document content."""
        content = doc.get('content', '')

        # Create excerpt from content
        excerpt = content[:100] + "..." if len(content) > 100 else content

        # Simple query generation based on domain
        query_templates = {
            'arc_challenge': [
                "다음 현상에 대해 설명해 주세요: {excerpt}",
                "이 현상의 원인은 무엇인가요? {excerpt}",
                "다음 상황에서 어떤 결과가 예상되나요? {excerpt}"
            ],
            'mmlu_biology': [
                "다음 생물학적 현상에 대해 알려주세요: {excerpt}",
                "이 생물의 특징은 무엇인가요? {excerpt}",
                "생물학에서 {excerpt} 의 역할은 무엇인가요?"
            ],
            'mmlu_physics': [
                "물리학에서 {excerpt} 의 원리는 무엇인가요?",
                "다음 물리 현상을 설명해 주세요: {excerpt}",
                "이 물리적 성질의 특징은 무엇인가요? {excerpt}"
            ],
            'mmlu_chemistry': [
                "화학에서 {excerpt} 반응의 특징은 무엇인가요?",
                "다음 화합물의 성질에 대해 알려주세요: {excerpt}",
                "화학적 현상 {excerpt} 에 대해 설명해 주세요"
            ],
            'mmlu_medicine': [
                "의학에서 {excerpt} 의 중요성은 무엇인가요?",
                "이 건강 상태의 특징은 무엇인가요? {excerpt}",
                "의학적 현상 {excerpt} 에 대해 알려주세요"
            ],
            'unknown': [
                "다음에 대해 설명해 주세요: {excerpt}",
                "이 현상에 대해 알려주세요: {excerpt}",
                "다음 상황을 설명해 주세요: {excerpt}"
            ]
        }

        # Map domain to template category
        if 'arc' in domain:
            templates = query_templates['arc_challenge']
        elif 'biology' in domain:
            templates = query_templates['mmlu_biology']
        elif 'physics' in domain:
            templates = query_templates['mmlu_physics']
        elif 'chemistry' in domain:
            templates = query_templates['mmlu_chemistry']
        elif 'medicine' in domain:
            templates = query_templates['mmlu_medicine']
        else:
            templates = query_templates['unknown']

        # Select random template
        template = random.choice(templates)

        # Generate query
        try:
            query = template.format(excerpt=excerpt)
            return query
        except (KeyError, ValueError):
            # Fallback for formatting errors
            return f"다음에 대해 설명해 주세요: {excerpt}"

    def save_validation_set(self, queries: List[Dict[str, Any]], output_path: str):
        """Save validation set to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for query in queries:
                json.dump(query, f, ensure_ascii=False)
                f.write('\n')

        print(f"💾 Saved {len(queries)} queries to {output_path}")

    def analyze_validation_balance(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the balance of the generated validation set."""
        domain_counts = Counter()
        source_counts = Counter()

        for query in queries:
            domain = query.get('domain', 'unknown')
            source = query.get('source', 'unknown')
            domain_counts[domain] += 1
            source_counts[source] += 1

        return {
            'total_queries': len(queries),
            'domain_distribution': dict(domain_counts),
            'source_distribution': dict(source_counts),
            'domain_balance_score': self._calculate_balance_score(domain_counts)
        }

    def _calculate_balance_score(self, distribution: Counter) -> float:
        """Calculate balance score (lower is more balanced)."""
        if not distribution:
            return 1.0

        values = list(distribution.values())
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5

        # Coefficient of variation (lower is more balanced)
        return std_dev / mean if mean > 0 else 1.0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Create balanced validation set")
    parser.add_argument(
        "--documents-path",
        default="data/documents.jsonl",
        help="Path to documents file"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of validation queries to generate"
    )
    parser.add_argument(
        "--output-path",
        default="data/validation_balanced.jsonl",
        help="Output path for balanced validation set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create balanced validation set
    creator = BalancedValidationCreator()
    creator.load_documents(args.documents_path)
    creator.analyze_domain_distribution()

    queries = creator.generate_validation_queries(args.num_queries)
    creator.save_validation_set(queries, args.output_path)

    # Analyze balance
    balance_analysis = creator.analyze_validation_balance(queries)

    print(f"\n📊 Balance Analysis:")
    print(f"Total queries: {balance_analysis['total_queries']}")
    print(f"Domain balance score: {balance_analysis['domain_balance_score']:.3f} (lower is better)")

    print(f"\n🏷️ Domain distribution in new validation set:")
    for domain, count in balance_analysis['domain_distribution'].items():
        pct = count / balance_analysis['total_queries'] * 100
        print(f"  {domain}: {count} queries ({pct:.1f}%)")


if __name__ == "__main__":
    main()