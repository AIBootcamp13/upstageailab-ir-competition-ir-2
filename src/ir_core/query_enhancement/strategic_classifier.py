# src/ir_core/query_enhancement/strategic_classifier.py

from typing import Dict, Any, List, Optional
import re
from enum import Enum


class QueryType(Enum):
    """Enumeration of strategic query types."""
    CONCEPTUAL = "conceptual"           # Abstract/conceptual questions
    CONVERSATIONAL = "conversational"    # Chit-chat, non-informational
    MULTI_TURN = "multi_turn"           # Context-dependent questions
    AMBIGUOUS = "ambiguous"             # Vague or broad questions
    COMPLEX = "complex"                 # Multi-part or compound questions
    SPECIFIC = "specific"               # Clear, specific questions
    SIMPLE = "simple"                   # Very clear, direct questions that need no enhancement
    OUT_OF_DOMAIN = "out_of_domain"     # Questions outside scientific scope


class StrategicQueryClassifier:
    """
    Strategic classifier that categorizes queries based on their characteristics
    and determines the optimal enhancement technique(s) to apply.
    """

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
            re.compile(r'\b(안녕|인사|반가워|반갑다|잘 있었어|오랜만|보고 싶었어)\b', re.IGNORECASE),
            re.compile(r'\b(도와|부탁|제발|부디|미안|고마워|감사)\b', re.IGNORECASE),
            re.compile(r'\b(추천|소개|말해|얘기|이야기|대화)\b', re.IGNORECASE),

            # Commands without scientific intent
            re.compile(r'\b(그만|멈춰|종료|끝내|다시 시작)\b', re.IGNORECASE),
            re.compile(r'\b(테스트|확인|체크|점검)\b', re.IGNORECASE)
        ]

        # Multi-turn context indicators
        self.multi_turn_indicators = [
            re.compile(r'\b(이전|전에|지난번|방금|위에서|아까)\b', re.IGNORECASE),
            re.compile(r'\b(계속|다음|또한|추가로|더불어)\b', re.IGNORECASE),
            re.compile(r'\b(그리고|그런데|하지만|그러나)\b', re.IGNORECASE),
            re.compile(r'\b(예를 들어|예를 들면|구체적으로)\b', re.IGNORECASE)
        ]

        # Ambiguous/broad question patterns
        self.ambiguous_indicators = [
            re.compile(r'\b(어떤|무엇이|어디에|언제쯤)\b', re.IGNORECASE),
            re.compile(r'\b(일반적|보통|평균|대부분|많은)\b', re.IGNORECASE),
            re.compile(r'\b(새로운|최근|현재|미래|과거)\b', re.IGNORECASE)
        ]

        # Complex query indicators
        self.complex_indicators = [
            re.compile(r'\b(그리고|또는|vs|대|비교|차이|대신)\b', re.IGNORECASE),
            re.compile(r'\b(어떻게|왜|무엇을|어디서|언제)\b', re.IGNORECASE),
            re.compile(r'\b(한편|반면|그러나|하지만)\b', re.IGNORECASE)
        ]

        # Out-of-domain indicators
        self.out_of_domain_keywords = [
            '가격', '비용', '구매', '판매', '쇼핑', '레시피', '요리',
            '날씨', '교통', '지도', '길찾기', '스포츠', '연예인',
            '게임', '영화', '음악', '패션', '뷰티', '건강', '의료'
        ]

    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify a query into strategic categories and recommend techniques.

        Args:
            query: The query to classify

        Returns:
            Dictionary with classification results and technique recommendations
        """
        # Initialize scores for each category
        scores = {
            QueryType.CONCEPTUAL: 0,
            QueryType.CONVERSATIONAL: 0,
            QueryType.MULTI_TURN: 0,
            QueryType.AMBIGUOUS: 0,
            QueryType.COMPLEX: 0,
            QueryType.SPECIFIC: 0,
            QueryType.SIMPLE: 0,
            QueryType.OUT_OF_DOMAIN: 0
        }

        # Analyze query characteristics
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
        conversational_matches = 0
        for pattern in self.conversational_patterns:
            if pattern.search(query):
                conversational_matches += 1
                scores[QueryType.CONVERSATIONAL] += 1

        # Boost conversational score for short queries that are clearly greetings
        if word_count <= 3 and conversational_matches > 0:
            # Check if it's a clear greeting/social interaction
            greeting_words = ['안녕', '반갑', '인사', 'hello', 'hi', 'hey']
            if any(word in query for word in greeting_words):
                scores[QueryType.CONVERSATIONAL] += 3  # Strong boost for clear greetings

        # Score multi-turn indicators
        for indicator in self.multi_turn_indicators:
            if indicator.search(query):
                scores[QueryType.MULTI_TURN] += 1

        # Score ambiguous indicators
        for indicator in self.ambiguous_indicators:
            if indicator.search(query):
                scores[QueryType.AMBIGUOUS] += 1

        # Score complex indicators
        complex_score = 0
        for indicator in self.complex_indicators:
            if indicator.search(query):
                complex_score += 1

        # Additional complexity heuristics
        if word_count > 15:
            complex_score += 1
        if complex_score >= 2:
            scores[QueryType.COMPLEX] += complex_score

        # Length-based scoring
        if word_count <= 5:
            scores[QueryType.CONCEPTUAL] += 1  # Short queries often conceptual
        elif word_count > 20:
            scores[QueryType.COMPLEX] += 1    # Long queries often complex
        elif 6 <= word_count <= 15:
            scores[QueryType.SIMPLE] += 1     # Medium-length queries often simple and direct

        # Score SIMPLE queries: clear, direct questions with specific terms
        simple_indicators = [
            # Specific scientific terms that indicate clear intent
            re.compile(r'\b(DNA|RNA|세포|유전자|단백질|효소|호르몬)\b', re.IGNORECASE),
            re.compile(r'\b(원자|분자|화학|물리|생물|지구과학|천문학)\b', re.IGNORECASE),
            re.compile(r'\b(계산|측정|실험|연구|분석|관찰)\b', re.IGNORECASE),
            # Direct question patterns
            re.compile(r'\b(무엇인가|어떤가|얼마나|어디인가)\b.*\?', re.IGNORECASE),
            re.compile(r'\b(예|아니오)\b.*\?', re.IGNORECASE),
        ]

        for indicator in simple_indicators:
            if indicator.search(query):
                scores[QueryType.SIMPLE] += 2  # Give higher weight to simple indicators

        # Boost SIMPLE score for queries that are direct and specific
        if (word_count >= 6 and word_count <= 15 and
            scores[QueryType.CONCEPTUAL] == 0 and
            scores[QueryType.AMBIGUOUS] == 0):
            scores[QueryType.SIMPLE] += 3  # Strong boost for medium-length direct queries

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
            'analysis': self._generate_analysis(primary_type, scores)
        }

    def _recommend_techniques(self, primary_type: QueryType, scores: Dict[QueryType, int], query: str) -> List[Dict[str, Any]]:
        """
        Recommend enhancement techniques based on query classification.

        Args:
            primary_type: The primary query type
            scores: Scores for all query types
            query: Original query

        Returns:
            List of recommended techniques with priorities
        """
        recommendations = []

        if primary_type == QueryType.CONCEPTUAL:
            # Conceptual queries: HyDE is most effective
            recommendations.append({
                'technique': 'hyde',
                'priority': 1,
                'reason': 'Conceptual queries benefit most from hypothetical document embeddings'
            })
            if scores[QueryType.AMBIGUOUS] > 0:
                recommendations.append({
                    'technique': 'step_back',
                    'priority': 2,
                    'reason': 'Step-back prompting helps clarify ambiguous conceptual questions'
                })

        elif primary_type == QueryType.CONVERSATIONAL:
            # Conversational queries: Skip enhancement entirely
            recommendations.append({
                'technique': 'bypass',
                'priority': 1,
                'reason': 'Conversational queries should bypass retrieval and use direct response'
            })

        elif primary_type == QueryType.MULTI_TURN:
            # Multi-turn queries: Rewriting is essential
            recommendations.append({
                'technique': 'rewriting',
                'priority': 1,
                'reason': 'Multi-turn queries need context consolidation through rewriting'
            })
            if scores[QueryType.CONCEPTUAL] > 0:
                recommendations.append({
                    'technique': 'hyde',
                    'priority': 2,
                    'reason': 'Conceptual multi-turn queries benefit from HyDE'
                })

        elif primary_type == QueryType.AMBIGUOUS:
            # Ambiguous queries: Step-back then HyDE
            recommendations.append({
                'technique': 'step_back',
                'priority': 1,
                'reason': 'Step-back prompting clarifies ambiguous queries'
            })
            recommendations.append({
                'technique': 'hyde',
                'priority': 2,
                'reason': 'HyDE helps with semantic matching for clarified queries'
            })

        elif primary_type == QueryType.COMPLEX:
            # Complex queries: Decomposition first, then apply other techniques
            recommendations.append({
                'technique': 'decomposition',
                'priority': 1,
                'reason': 'Complex queries should be decomposed into simpler sub-queries'
            })
            if scores[QueryType.CONCEPTUAL] > 0:
                recommendations.append({
                    'technique': 'hyde',
                    'priority': 2,
                    'reason': 'Apply HyDE to conceptual sub-queries'
                })

        elif primary_type == QueryType.SIMPLE:
            # Simple queries: Bypass enhancement entirely
            recommendations.append({
                'technique': 'bypass',
                'priority': 1,
                'reason': 'Simple, direct queries should bypass enhancement and go directly to retrieval'
            })

        elif primary_type == QueryType.OUT_OF_DOMAIN:
            # Out-of-domain: Skip retrieval
            recommendations.append({
                'technique': 'bypass',
                'priority': 1,
                'reason': 'Out-of-domain queries should bypass scientific retrieval'
            })

        else:  # SPECIFIC or default
            # Specific queries: Standard rewriting
            recommendations.append({
                'technique': 'rewriting',
                'priority': 1,
                'reason': 'Specific queries benefit from standard query rewriting'
            })

        return recommendations

    def _generate_analysis(self, primary_type: QueryType, scores: Dict[QueryType, int]) -> str:
        """
        Generate a human-readable analysis of the query classification.

        Args:
            primary_type: Primary query type
            scores: Classification scores

        Returns:
            Analysis description
        """
        type_descriptions = {
            QueryType.CONCEPTUAL: "개념적/추상적 질문",
            QueryType.CONVERSATIONAL: "대화형/비정보성 질문",
            QueryType.MULTI_TURN: "다중 대화 질문",
            QueryType.AMBIGUOUS: "모호하거나 범위가 넓은 질문",
            QueryType.COMPLEX: "복합 질문",
            QueryType.SPECIFIC: "구체적 질문",
            QueryType.SIMPLE: "단순 직접적 질문",
            QueryType.OUT_OF_DOMAIN: "범위 외 질문"
        }

        analysis = f"질문 유형: {type_descriptions[primary_type]}"

        # Add secondary characteristics
        secondary_types = [t for t in scores if scores[t] > 0 and t != primary_type]
        if secondary_types:
            secondary_desc = [type_descriptions[t] for t in secondary_types[:2]]  # Top 2
            analysis += f" (추가 특징: {', '.join(secondary_desc)})"

        return analysis

    def should_bypass_retrieval(self, classification: Dict[str, Any]) -> bool:
        """
        Determine if retrieval should be bypassed based on classification.

        Args:
            classification: Query classification results

        Returns:
            True if retrieval should be bypassed
        """
        primary_type = classification['primary_type']
        techniques = classification['recommended_techniques']

        # Bypass for conversational, simple, or out-of-domain queries
        return (primary_type in ['conversational', 'simple', 'out_of_domain'] or
                any(t['technique'] == 'bypass' for t in techniques))