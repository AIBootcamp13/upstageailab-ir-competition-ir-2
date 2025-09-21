# src/ir_core/query_enhancement/prompts.py

"""
Prompt templates for query enhancement module.

This module contains all prompt templates used by the query enhancement system.
"""

from typing import Optional

# Query intent classification prompt
QUERY_INTENT_CLASSIFICATION_PROMPT = """당신은 사용자 질문의 의도를 분석하여 가장 적절한 처리 방식을 결정하는 AI 분류 전문가입니다.
다음 카테고리 중 하나로만 분류하여 그 키워드를 출력해야 합니다.

[카테고리]
- conversational: 사용자가 감정을 표현하거나, 인사하거나, AI 자체에 대해 묻는 등 정보 검색 목적이 아닌 일반 대화. (예: "고마워", "너는 누구니?", "오늘 기분 어때?", "요새 너무 힘들다")
- conversational_follow_up: 이전 대화의 맥락 없이는 의미가 불분명한 후속 질문. (예: "그럼 그건 왜 그런데?", "다른 예시는 없어?")
- simple_keyword: 명확한 키워드(고유명사, 전문 용어)를 포함하고 있어 추가적인 향상 없이도 검색이 가능한 직접적인 질문. (예: "엽록체의 역할은?", "라마 3.1 모델에 대해 알려줘.")
- conceptual_vague: "방법", "원리", "영향", "이유", "차이점" 등 키워드가 아닌 개념에 대해 묻는 광범위하거나 추상적인 질문. (예: "공생과 기생의 차이점은?", "지구 온난화가 해양 생태계에 미치는 영향은?", "나무가 생태계에서 하는 역할에 대해 설명해줘.")

[대화 기록]
{conversation_history}

[사용자 최근 질문]
{query}

[분류]
출력은 카테고리 키워드만 출력하세요. 설명을 추가하지 마세요."""


def format_query_intent_prompt(query: str, conversation_history: Optional[str] = None) -> str:
    """
    Format the query intent classification prompt with query and history.

    Args:
        query: The query to classify
        conversation_history: Previous conversation history

    Returns:
        Formatted prompt string
    """
    if conversation_history:
        history_text = conversation_history
    else:
        history_text = "없음"

    return QUERY_INTENT_CLASSIFICATION_PROMPT.format(
        conversation_history=history_text,
        query=query
    )