# src/ir_core/tools/retrieval_tool.py
from typing import List, Dict, Any
from ..retrieval import hybrid_retrieve
from ..config import settings # --- Phase 1: 중앙 설정 임포트 ---

# --- Phase 1: 기본 top_k 값을 중앙 설정에서 가져오도록 수정 ---
def scientific_search(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Searches for scientific documents related to a specific query.

    Use this tool when a user asks a question about a scientific topic,
    concept, or fact.

    Args:
        query: The user's question or search term.
        top_k: The number of documents to return.

    Returns:
        A list of the top_k most relevant document chunks, including their ID and content.
    """
    # top_k가 명시적으로 주어지지 않으면 설정 파일의 기본값을 사용
    final_top_k = top_k if top_k is not None else settings.PIPELINE_DEFAULT_TOP_K

    print(f"Executing scientific_search with query: '{query}' and top_k: {final_top_k}")

    # Use the existing hybrid_retrieve function as the tool's core logic
    retrieved_hits = hybrid_retrieve(query=query, rerank_k=final_top_k)

    formatted_results = []
    for hit in retrieved_hits:
        inner = hit.get("hit", {})
        source_doc = inner.get("_source", {}) if isinstance(inner, dict) else {}
        doc_id = source_doc.get("docid") or inner.get("_id")
        content = source_doc.get("content", "No content available.")
        score = hit.get("score") if isinstance(hit, dict) else None
        if doc_id:
            formatted_results.append({"id": doc_id, "content": content, "score": float(score) if score is not None else None})

    seen = {}
    order = []
    for item in formatted_results:
        _id = item["id"]
        if _id not in seen:
            seen[_id] = item
            order.append(_id)
        else:
            existing = seen[_id]
            existing_score = existing.get("score")
            new_score = item.get("score")
            try:
                if existing_score is None:
                    seen[_id] = item
                elif new_score is not None and new_score > existing_score:
                    seen[_id] = item
            except Exception:
                pass

    deduped = [seen[k] for k in order]
    return deduped

def get_tool_definition():
    """
    Returns the JSON schema definition for the scientific_search tool,
    which is required for models like OpenAI's function calling.
    """
    return {
        "type": "function",
        "function": {
            "name": "scientific_search",
            "description": (
                "과학적 사실, 개념, 현상에 대한 질문에 답하기 위해 관련 문서를 검색하는 도구입니다. "
                "사용자의 질문이 과학과 관련 있을 때만 사용해야 합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "검색에 사용할 명확하고 완전한 형태의 한국어 질의문. "
                            "예: '달이 항상 같은 면만 보이는 이유'"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"반환할 문서의 수. (기본값: {settings.PIPELINE_DEFAULT_TOP_K})",
                        "default": settings.PIPELINE_DEFAULT_TOP_K,
                    }
                },
                "required": ["query"],
            },
        }
    }
