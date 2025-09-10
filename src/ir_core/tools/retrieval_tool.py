# src/ir_core/tools/retrieval_tool.py
from typing import List, Dict, Any
from ..retrieval import hybrid_retrieve

def scientific_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
    print(f"Executing scientific_search with query: '{query}'")

    # Use the existing hybrid_retrieve function as the tool's core logic
    retrieved_hits = hybrid_retrieve(query=query, rerank_k=top_k)

    # --- FIX ---
    # The previous version only returned the content. This updated version
    # returns both the document ID and the content, which is necessary for
    # the evaluation script to generate a correct submission file.
    formatted_results = []
    for hit in retrieved_hits:
        # hit has shape: {"hit": <es hit>, "cosine": ..., "score": ...}
        inner = hit.get("hit", {})
        source_doc = inner.get("_source", {}) if isinstance(inner, dict) else {}
        # Prefer stable document id in _source['docid'] if available
        doc_id = source_doc.get("docid") or inner.get("_id")
        content = source_doc.get("content", "No content available.")
        score = hit.get("score") if isinstance(hit, dict) else None
        if doc_id:
            formatted_results.append({"id": doc_id, "content": content, "score": float(score) if score is not None else None})

    # Deduplicate by id while preserving order. If duplicates exist, keep
    # the entry with the highest score but preserve the order of first
    # appearance among unique ids.
    seen = {}
    order = []
    for item in formatted_results:
        _id = item["id"]
        if _id not in seen:
            seen[_id] = item
            order.append(_id)
        else:
            # keep the one with the higher score
            existing = seen[_id]
            existing_score = existing.get("score")
            new_score = item.get("score")
            # If new_score is higher (or existing score is None), replace
            try:
                if existing_score is None:
                    seen[_id] = item
                elif new_score is not None and new_score > existing_score:
                    seen[_id] = item
            except Exception:
                # If comparison fails for any reason, prefer existing
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
                "검색 도구입니다. 도구를 호출할 때 반드시 '독립형(standalone) 질의문'을 한국어로만 작성하세요. "
                "질의문에는 추가 문맥, 지시문, 혹은 번역 지시를 포함하지 마시고 오직 검색어(질의)만 포함해야 합니다. "
                "절대로 영어로 작성하지 마세요. 모델이 도구를 호출할 때 이 규칙을 준수해야 합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "검색에 사용될 완전한 형태의 독립형 질의문입니다. 반드시 한국어로 작성하세요. "
                            "예시: '나무의 분류에 대한 방법' — 문장부호와 불필요한 설명 없이 질의만 적어주세요."
                        ),
                        "examples": ["나무의 분류에 대한 방법", "헬륨이 다른 원소와 반응하지 않는 이유"]
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

