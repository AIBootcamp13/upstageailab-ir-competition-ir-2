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
        A list of the top_k most relevant document chunks.
    """
    print(f"Executing scientific_search with query: '{query}'")

    # Use the existing hybrid_retrieve function as the tool's core logic
    retrieved_hits = hybrid_retrieve(query=query, rerank_k=top_k)

    # Format the results for the LLM
    formatted_results = []
    for hit in retrieved_hits:
        source_doc = hit.get("hit", {}).get("_source", {})
        content = source_doc.get("content", "No content available.")
        formatted_results.append({"content": content})

    return formatted_results

def get_tool_definition():
    """
    Returns the JSON schema definition for the scientific_search tool,
    which is required for models like OpenAI's function calling.
    """
    return {
        "type": "function",
        "function": {
            "name": "scientific_search",
            "description": "Searches for and retrieves scientific documents based on a user's query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific question or topic to search for. Should be a self-contained query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "The number of documents to retrieve.",
                        "default": 5,
                    }
                },
                "required": ["query"],
            },
        }
    }
