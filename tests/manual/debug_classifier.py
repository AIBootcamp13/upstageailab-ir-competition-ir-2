#!/usr/bin/env python3
# scripts/debug_classifier.py

import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ir_core.query_enhancement.strategic_classifier import StrategicQueryClassifier

def debug_classification():
    classifier = StrategicQueryClassifier()

    test_queries = [
        "너 잘하는게 뭐야?",
        "우울한데 신나는 얘기 좀 해줄래?",
        "플랑크톤의 역할에 대해 알려줘.",
        "기생과 공생의 차이에 대해 알려줘."
    ]

    # Debug AI-directed patterns
    ai_directed_patterns = [
        re.compile(r'\b(너는|너의|너|니가|너에게|너한테)\b.*\b(뭐|무엇|어때|어떻게|왜|잘)\b', re.IGNORECASE),
        re.compile(r'\b(너|니가)\b.*\b(좋아|싫어|할 수|가능)\b', re.IGNORECASE),
        re.compile(r'\b(너|당신)\b.*\b(해|해주|해줄래|해주세요)\b', re.IGNORECASE)
    ]

    for query in test_queries:
        print(f"\n🔍 Debugging: {query}")

        # Check AI-directed patterns
        ai_matched = False
        for i, pattern in enumerate(ai_directed_patterns):
            if pattern.search(query):
                print(f"   AI Pattern {i} matched!")
                ai_matched = True
                break
        if not ai_matched:
            print("   No AI pattern matched")

        result = classifier.classify_query(query)

        print(f"   Primary Type: {result['primary_type']}")
        print(f"   Scores: {result['scores']}")
        print(f"   Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    debug_classification()