#!/usr/bin/env python3
"""
Scientific Keywords Preprocessing Script

This script processes the curated scientific keywords from scientific-keywords-extra.md,
removes duplicates, normalizes terms, and organizes them by scientific domains.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict


class ScientificKeywordsProcessor:
    """Processor for scientific keywords from markdown files"""

    def __init__(self):
        self.keywords = set()
        self.normalized_keywords = set()
        self.keyword_mapping = {}  # normalized -> original
        self.domain_keywords = defaultdict(set)

    def parse_markdown_file(self, file_path: str) -> List[str]:
        """Parse markdown file and extract keywords"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by sections (marked by headers)
        sections = re.split(r'^#+\s+', content, flags=re.MULTILINE)

        all_keywords = []

        for section in sections:
            if not section.strip():
                continue

            # Extract keywords from numbered lists
            # Pattern matches: 1. **keyword** or 1. **keyword** (description)
            keyword_pattern = r'^\d+\.\s*\*\*([^*]+)\*\*(?:\s*\([^)]*\))?'
            matches = re.findall(keyword_pattern, section, re.MULTILINE)

            for match in matches:
                keyword = match.strip()
                if keyword:
                    all_keywords.append(keyword)

        return all_keywords

    def normalize_keyword(self, keyword: str) -> str:
        """Normalize a keyword for deduplication"""
        # Remove parentheses and their contents
        normalized = re.sub(r'\s*\([^)]*\)', '', keyword)
        # Convert to lowercase
        normalized = normalized.lower()
        # Remove special characters except spaces and hyphens
        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        # Strip whitespace
        normalized = normalized.strip()

        return normalized

    def categorize_keyword(self, keyword: str) -> str:
        """Categorize keyword into scientific domain"""
        keyword_lower = keyword.lower()

        # Define domain keywords with more comprehensive coverage
        domains = {
            'biology': [
                'dna', 'rna', 'protein', 'cell', 'gene', 'chromosome', 'enzyme',
                'virus', 'bacteria', 'organism', 'evolution', 'mutation', 'species',
                'ecosystem', 'photosynthesis', 'respiration', 'metabolism', 'hormone',
                'neuron', 'brain', 'immune', 'antibody', 'vaccine', 'cancer',
                '생물', '세포', '유전자', '단백질', '바이러스', '면역', '호르몬',
                '신경', '뇌', '혈액', '심장', '폐', '간', '신장', '근육', '뼈',
                '피부', '눈', '귀', '코', '혀', '생식', '번식', '진화', '종',
                '생태계', '생물 다양성', '식물', '동물', '미생물', '균', '박테리아'
            ],
            'chemistry': [
                'acid', 'base', 'ph', 'molecule', 'atom', 'element', 'compound',
                'reaction', 'catalyst', 'solvent', 'solution', 'crystal', 'polymer',
                'organic', 'inorganic', 'electrolysis', 'oxidation', 'reduction',
                '화학', '산', '염기', '용액', '용매', '결합', '반응', '산화', '환원',
                '분자', '원자', '원소', '화합물', '결정', '용해', '침전', '증류',
                '삼투압', '농도', 'ph', '산성', '알칼리성', '중성', '염', '산소',
                '수소', '탄소', '질소', '산화물', '염화물', '수화물'
            ],
            'physics': [
                'force', 'energy', 'power', 'velocity', 'acceleration', 'mass',
                'gravity', 'magnet', 'electric', 'current', 'voltage', 'resistance',
                'wave', 'frequency', 'wavelength', 'light', 'sound', 'heat', 'temperature',
                '물리', '힘', '에너지', '속도', '가속도', '질량', '중력', '자기',
                '전기', '전류', '전압', '저항', '파동', '주파수', '파장', '빛',
                '소리', '열', '온도', '압력', '밀도', '운동', '정지', '마찰',
                '전자기', '광학', '음향', '열역학', '역학', '전자', '양자'
            ],
            'earth_science': [
                'earthquake', 'volcano', 'weather', 'climate', 'ocean', 'atmosphere',
                'mineral', 'rock', 'soil', 'water', 'cycle', 'erosion', 'fossil',
                '지구', '지질', '지진', '화산', '날씨', '기후', '바다', '대기',
                '광물', '암석', '토양', '물', '주기', '침식', '화석', '대륙',
                '판', '지각', '맨틀', '핵', '행성', '달', '태양', '별', '은하',
                '우주', '천체', '운석', '혜성', '소행성'
            ],
            'mathematics': [
                'probability', 'statistics', 'function', 'equation', 'graph',
                'variable', 'distribution', 'calculation', 'measurement',
                '수학', '확률', '통계', '함수', '방정식', '그래프', '변수',
                '분포', '계산', '측정', '기하', '대수', '미적분', '벡터',
                '행렬', '집합', '논리', '증명'
            ],
            'computer_science': [
                'algorithm', 'network', 'database', 'security', 'encryption',
                'programming', 'memory', 'processor', 'software', 'hardware',
                '컴퓨터', '알고리즘', '네트워크', '데이터베이스', '보안', '암호화',
                '프로그래밍', '메모리', '프로세서', '소프트웨어', '하드웨어',
                '코드', '데이터', '정보', '시스템', '운영체제', '인터넷'
            ],
            'medicine': [
                'disease', 'treatment', 'diagnosis', 'symptom', 'therapy',
                'surgery', 'drug', 'vaccine', 'health', 'medical',
                '질병', '치료', '진단', '증상', '치료법', '수술', '약', '백신',
                '건강', '의학', '병원', '의사', '환자', '약물', '면역', '감염'
            ],
            'environmental_science': [
                'environment', 'pollution', 'conservation', 'sustainability',
                'recycling', 'waste', 'climate change', 'global warming',
                '환경', '오염', '보존', '지속가능', '재활용', '폐기물', '기후변화',
                '지구온난화', '생태계', '자원', '에너지', '탄소', '온실가스'
            ]
        }

        # Check each domain with more flexible matching
        best_match = 'general_science'
        max_matches = 0

        for domain, domain_keywords in domains.items():
            matches = sum(1 for domain_kw in domain_keywords if domain_kw in keyword_lower)
            if matches > max_matches:
                max_matches = matches
                best_match = domain

        # If we have at least one match, use that domain
        if max_matches > 0:
            return best_match

        return 'general_science'

    def process_keywords(self, file_path: str) -> Dict:
        """Main processing function"""
        print(f"Processing keywords from {file_path}...")

        # Parse keywords from file
        raw_keywords = self.parse_markdown_file(file_path)
        print(f"Found {len(raw_keywords)} raw keywords")

        # Process each keyword
        for keyword in raw_keywords:
            normalized = self.normalize_keyword(keyword)

            # Skip empty or very short keywords
            if len(normalized) < 2:
                continue

            # Store original and normalized versions
            if normalized not in self.normalized_keywords:
                self.normalized_keywords.add(normalized)
                self.keyword_mapping[normalized] = keyword

                # Categorize
                domain = self.categorize_keyword(keyword)
                self.domain_keywords[domain].add(keyword)

        print(f"After deduplication: {len(self.normalized_keywords)} unique keywords")

        # Prepare results
        results = {
            'summary': {
                'total_raw_keywords': len(raw_keywords),
                'unique_normalized_keywords': len(self.normalized_keywords),
                'domains': {}
            },
            'domains': {},
            'all_keywords': sorted(list(self.keyword_mapping.values())),
            'normalized_mapping': self.keyword_mapping
        }

        # Organize by domain
        for domain, keywords in self.domain_keywords.items():
            domain_list = sorted(list(keywords))
            results['domains'][domain] = domain_list
            results['summary']['domains'][domain] = len(domain_list)

        return results

    def save_results(self, results: Dict, output_file: str):
        """Save processed results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {output_file}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Process scientific keywords")
    parser.add_argument("--input", type=str,
                       default="src/ir_core/retrieval/scientific-keywords-extra.md",
                       help="Input markdown file")
    parser.add_argument("--output", type=str,
                       default="data/processed_scientific_keywords.json",
                       help="Output JSON file")

    args = parser.parse_args()

    # Process keywords
    processor = ScientificKeywordsProcessor()
    results = processor.process_keywords(args.input)

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total raw keywords: {results['summary']['total_raw_keywords']}")
    print(f"Unique keywords: {results['summary']['unique_normalized_keywords']}")
    print("\nKeywords by domain:")
    for domain, count in results['summary']['domains'].items():
        print(f"  {domain}: {count} keywords")

    # Save results
    processor.save_results(results, args.output)


if __name__ == "__main__":
    main()