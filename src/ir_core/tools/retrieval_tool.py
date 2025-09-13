# src/ir_core/tools/retrieval_tool.py

from typing import List, Dict, Any
from pydantic import BaseModel
from ..retrieval import hybrid_retrieve

class ScientificSearchArgs(BaseModel):
    query: str
    top_k: int = 5

def scientific_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    특정 쿼리와 관련된 과학 문서를 검색합니다.
    사용자가 과학적 주제, 개념 또는 사실에 대해 질문할 때 이 도구를 사용합니다.
    """
    print(f"Executing scientific_search with query: '{query}'")
    retrieved_hits = hybrid_retrieve(query=query, rerank_k=top_k)

    formatted_results = []
    for hit in retrieved_hits:
        inner = hit.get("hit", {})
        source_doc = inner.get("_source", {})
        doc_id = source_doc.get("docid") or inner.get("_id")
        content = source_doc.get("content", "No content available.")
        score = hit.get("score")
        if doc_id:
            formatted_results.append({
                "id": doc_id,
                "content": content,
                "score": float(score) if score is not None else None
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
            existing_score = seen[_id].get("score")
            new_score = item.get("score")
            if existing_score is None or (new_score is not None and new_score > existing_score):
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
                    }
                },
                "required": ["query"],
            },
        }
    }