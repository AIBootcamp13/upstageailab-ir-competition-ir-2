# src/ir_core/tools/retrieval_tool.py

# src/ir_core/tools/retrieval_tool.py
import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from ..retrieval import hybrid_retrieve
from ..config import settings

class ScientificSearchArgs(BaseModel):
    query: str
    top_k: int = 5
    use_profiling_insights: Optional[bool] = None  # None means use settings default

def scientific_search(query: str, top_k: int = 5, use_profiling_insights: Optional[bool] = None) -> List[Dict[str, Any]]:
    """
    특정 쿼리와 관련된 과학 문서를 검색합니다.
    사용자가 과학적 주제, 개념 또는 사실에 대해 질문할 때 이 도구를 사용합니다.

    Args:
        query: 검색 쿼리
        top_k: 반환할 상위 문서 수
        use_profiling_insights: 프로파일링 인사이트를 사용할지 여부 (None이면 settings.yaml의 값 사용)
    """
    # Get profiling insights setting from configuration
    insights_config = getattr(settings, 'profiling_insights', {})
    settings_enabled = insights_config.get('enabled', True)

    # If use_profiling_insights is None, use the value from settings
    # If it's explicitly set by LLM but settings say disabled, respect settings
    if use_profiling_insights is None:
        use_profiling_insights = settings_enabled
    elif not settings_enabled:
        # Settings override LLM choice - if disabled in settings, force to False
        use_profiling_insights = False

    # Ensure it's a boolean
    use_profiling_insights = bool(use_profiling_insights)

    print(f"Executing scientific_search with query: '{query}' (profiling_insights: {use_profiling_insights})")
    retrieved_hits = hybrid_retrieve(
        query=query,
        rerank_k=top_k,
        use_profiling_insights=use_profiling_insights
    )

    formatted_results = []
    for hit in retrieved_hits:
        # Access the result directly (not nested under "hit" key)
        source_doc = hit.get("_source", {})
        doc_id = source_doc.get("docid") or hit.get("_id")
        content = source_doc.get("content", "No content available.")
        score = hit.get("score", 0.0)

        # Extract preserved scores
        rrf_score = hit.get("rrf_score", 0.0)
        sparse_score = hit.get("sparse_score", 0.0)
        dense_score = hit.get("dense_score", 0.0)
        es_score = hit.get("_score", 0.0)

        # Handle NaN values by converting them to 0.0
        if score is not None and not math.isnan(score):
            score = float(score)
        else:
            score = 0.0

        if doc_id:
            formatted_results.append({
                "id": doc_id,
                "content": content,
                "score": score,
                "rrf_score": rrf_score,
                "sparse_score": sparse_score,
                "dense_score": dense_score,
                "es_score": es_score
            })

    # ID를 기준으로 중복을 제거하면서 순서를 보존합니다.
    seen = {}
    order = []
    for item in formatted_results:
        _id = item["id"]
        if _id not in seen:
            seen[_id] = item
            order.append(_id)
        else:
            # 점수가 더 높은 항목을 유지합니다.
            existing_score = seen[_id].get("score", 0.0)
            new_score = item.get("score", 0.0)

            # Handle NaN values in comparison
            if math.isnan(existing_score):
                existing_score = 0.0
            if math.isnan(new_score):
                new_score = 0.0

            if new_score > existing_score:
                seen[_id] = item

    return [seen[k] for k in order]

# --- 함수 시그니처 변경됨 ---
# 이제 이 함수는 'prompt_description'을 인자로 받습니다.
def get_tool_definition(prompt_description: str):
    """
    OpenAI의 함수 호출(function calling)에 필요한 scientific_search 도구의 JSON 스키마 정의를 반환합니다.
    """
    return {
        "type": "function",
        "function": {
            "name": "scientific_search",
            # --- 프롬프트가 동적으로 삽입됨 ---
            "description": prompt_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색에 사용될 완전한 형태의 독립형 질의문입니다. 반드시 한국어로 작성하세요.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "검색할 문서 수 (예: 3 또는 5)",
                        "default": 5,
                    },
                    "use_profiling_insights": {
                        "type": ["boolean", "null"],
                        "description": "프로파일링 인사이트를 사용하여 검색 최적화 여부 (null이면 settings.yaml 값 사용)",
                        "default": None,
                    }
                },
                "required": ["query"],
            },
        }
    }