#!/usr/bin/env python3
# scripts/test_strategic_enhancement.py

"""
Test script to demonstrate the strategic query enhancement system.
"""

import sys
import os
import re
from enum import Enum
from typing import Dict, Any, List

# Copy the StrategicQueryClassifier code here for standalone testing
class QueryType(Enum):
    """Enumeration of strategic query types."""
    CONCEPTUAL = "conceptual"           # Abstract/conceptual questions
    CONVERSATIONAL = "conversational"    # Chit-chat, non-informational
    MULTI_TURN = "multi_turn"           # Context-dependent questions
    AMBIGUOUS = "ambiguous"             # Vague or broad questions
    COMPLEX = "complex"                 # Multi-part or compound questions
    SPECIFIC = "specific"               # Clear, specific questions
    OUT_OF_DOMAIN = "out_of_domain"     # Questions outside scientific scope


class StrategicQueryClassifier:
    """Strategic classifier for standalone testing."""

    def __init__(self):
        # Conceptual query patterns (abstract, relationship-based)
        self.conceptual_patterns = [
            re.compile(r'\b(역할|영향|관계|차이|비교|의의|의미|기능)\b', re.IGNORECASE),
            re.compile(r'\b(왜|이유|원인|결과|영향을)\b', re.IGNORECASE),
            re.compile(r'\b(어떻게|과정|메커니즘|방식)\b', re.IGNORECASE),
            re.compile(r'\b(개념|정의|설명|이해)\b', re.IGNORECASE)
        ]

        # Conversational/chit-chat patterns
        self.conversational_patterns = [
            # AI-directed questions
            re.compile(r'\b(너는|너의|자네|당신은|너|니가|너에게|너한테)\b', re.IGNORECASE),
            re.compile(r'\b(어때|어떻게 지내|잘 지내|괜찮아|좋아)\b', re.IGNORECASE),
            re.compile(r'\b(너 잘|너 못|너의 능력|너의 기능)\b', re.IGNORECASE),

            # Emotional expressions and requests
            re.compile(r'\b(좋아|싫어|재미있|지루해|행복해|슬퍼|화나|우울해|기분)\b', re.IGNORECASE),
            re.compile(r'\b(신나는|재미있는|웃긴|슬픈|감동적인)\b', re.IGNORECASE),
            re.compile(r'\b(얘기해|말해|이야기해|해주|해줄래|해주세요)\b', re.IGNORECASE),

            # Social interactions
            re.compile(r'\b(안녕|인사|반가워|잘 있었어|오랜만|보고 싶었어)\b', re.IGNORECASE),
            re.compile(r'\b(도와|부탁|제발|부디|미안|고마워|감사)\b', re.IGNORECASE),
            re.compile(r'\b(추천|소개|말해|얘기|이야기|대화)\b', re.IGNORECASE),

            # Commands without scientific intent
            re.compile(r'\b(그만|멈춰|종료|끝내|다시 시작)\b', re.IGNORECASE),
            re.compile(r'\b(테스트|확인|체크|점검)\b', re.IGNORECASE)
        ]

        # Out-of-domain indicators
        self.out_of_domain_keywords = [
            '가격', '비용', '구매', '판매', '쇼핑', '레시피', '요리',
            '날씨', '교통', '지도', '길찾기', '스포츠', '연예인',
            '게임', '영화', '음악', '패션', '뷰티', '건강', '의료'
        ]

    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify a query into strategic categories."""
        # Initialize scores
        scores = {QueryType.CONCEPTUAL: 0, QueryType.CONVERSATIONAL: 0,
                 QueryType.AMBIGUOUS: 0, QueryType.COMPLEX: 0, QueryType.SPECIFIC: 0,
                 QueryType.OUT_OF_DOMAIN: 0}

        query_lower = query.lower()
        word_count = len(query.split())

        # Check for out-of-domain keywords
        for keyword in self.out_of_domain_keywords:
            if keyword in query_lower:
                scores[QueryType.OUT_OF_DOMAIN] += 2

        # Score conceptual patterns
        for pattern in self.conceptual_patterns:
            if pattern.search(query):
                scores[QueryType.CONCEPTUAL] += 1

        # Score conversational patterns
        for pattern in self.conversational_patterns:
            if pattern.search(query):
                scores[QueryType.CONVERSATIONAL] += 1

        # Length-based scoring
        if word_count <= 5:
            scores[QueryType.CONCEPTUAL] += 1
        elif word_count > 15:
            scores[QueryType.COMPLEX] += 1

        # Special handling for AI-directed questions
        ai_directed_patterns = [
            re.compile(r'\b너\b.*\b(잘|못|좋아|싫어|할|하는|하는게|무엇|뭐)\b', re.IGNORECASE),
            re.compile(r'\b너는\b.*\b(뭐|무엇|어때|어떻게)\b', re.IGNORECASE),
            re.compile(r'\b너\b.*\b(해|해주|해줄래|해주세요)\b', re.IGNORECASE)
        ]

        for pattern in ai_directed_patterns:
            if pattern.search(query):
                scores[QueryType.CONVERSATIONAL] += 3  # Strong boost for AI-directed questions
                # If this is clearly an AI-directed question, reduce conceptual score
                if scores[QueryType.CONCEPTUAL] > 0:
                    scores[QueryType.CONCEPTUAL] -= 1
                break

        # Tie-breaker: if conversational and conceptual scores are equal and > 0,
        # prioritize conversational for AI-directed questions
        if (scores[QueryType.CONVERSATIONAL] == scores[QueryType.CONCEPTUAL] and
            scores[QueryType.CONVERSATIONAL] > 0):
            # Check if it's AI-directed
            if re.search(r'\b너\b', query, re.IGNORECASE):
                scores[QueryType.CONVERSATIONAL] += 1

        # Determine primary query type
        primary_type = max(scores.keys(), key=lambda k: scores[k])

        # If no strong indicators, classify as specific
        if scores[primary_type] == 0:
            primary_type = QueryType.SPECIFIC
            scores[QueryType.SPECIFIC] = 1

        # Generate technique recommendations
        techniques = self._recommend_techniques(primary_type, scores, query)

        return {
            'primary_type': primary_type.value,
            'scores': {k.value: v for k, v in scores.items()},
            'recommended_techniques': techniques,
            'confidence': scores[primary_type] / max(1, sum(scores.values())),
            'query_length': word_count,
            'analysis': f"Query classified as {primary_type.value}"
        }

    def _recommend_techniques(self, primary_type: QueryType, scores: Dict[QueryType, int], query: str) -> List[Dict[str, Any]]:
        """Recommend enhancement techniques based on query classification."""
        recommendations = []

        if primary_type == QueryType.CONCEPTUAL:
            recommendations.append({
                'technique': 'hyde',
                'priority': 1,
                'reason': 'Conceptual queries benefit most from hypothetical document embeddings'
            })
        elif primary_type == QueryType.CONVERSATIONAL:
            recommendations.append({
                'technique': 'bypass',
                'priority': 1,
                'reason': 'Conversational queries should bypass retrieval'
            })
        elif primary_type == QueryType.COMPLEX:
            recommendations.append({
                'technique': 'decomposition',
                'priority': 1,
                'reason': 'Complex queries should be decomposed'
            })
        else:
            recommendations.append({
                'technique': 'rewriting',
                'priority': 1,
                'reason': 'Standard query rewriting'
            })

        return recommendations


def test_strategic_classification():
    """Test the strategic classifier with various query types."""
    print("🧠 Testing Strategic Query Classification")
    print("=" * 50)

    classifier = StrategicQueryClassifier()

    test_queries = [
        # Conceptual queries
        "플랑크톤의 역할에 대해 알려줘.",
        "기생과 공생의 차이에 대해 알려줘.",

        # Conversational queries
        "너 잘하는게 뭐야?",
        "우울한데 신나는 얘기 좀 해줄래?",

        # Complex queries
        "인공지능이 과학 연구에서 어떤 역할을 하고 있으며, 앞으로 어떻게 발전할 것인가?",
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        classification = classifier.classify_query(query)
        print(f"   Type: {classification['primary_type']}")
        print(f"   Confidence: {classification['confidence']:.2f}")
        print(f"   Techniques: {[t['technique'] for t in classification['recommended_techniques']]}")


if __name__ == "__main__":
    try:
        test_strategic_classification()
        print("\n✅ Strategic classification test completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()