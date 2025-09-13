# src/ir_core/orchestration/pipeline.py

from typing import List, Dict, Any, cast
import os
import json
import openai
import requests
from openai.types.chat import ChatCompletionMessageToolCall

from ..generation.base import BaseGenerator
from ..tools.dispatcher import default_dispatcher, ToolDispatcher
from ..tools.retrieval_tool import get_tool_definition
from .rewriter_openai import BaseQueryRewriter # QueryRewriter를 임포트합니다.

class RAGPipeline:
    """
    QueryRewriter를 포함하여 전체 RAG 프로세스를 조율합니다.
    """
    def __init__(
        self,
        generator: BaseGenerator,
        query_rewriter: BaseQueryRewriter, # QueryRewriter 인스턴스를 받도록 __init__ 수정
        tool_prompt_description: str,
        tool_calling_model: str,
        dispatcher: ToolDispatcher = default_dispatcher
    ):
        """
        RAG 파이프라인을 초기화합니다.
        """
        self.generator = generator
        self.query_rewriter = query_rewriter # rewriter를 인스턴스 변수로 저장
        self.dispatcher = dispatcher
        self.tool_prompt_description = tool_prompt_description
        self.tool_calling_model = tool_calling_model
        # Determine if using OpenAI or Ollama for tool calling
        if ":" in tool_calling_model:  # Assume Ollama model (e.g., qwen2:7b)
            self.use_ollama = True
            self.ollama_url = "http://localhost:11434/api/chat"
        else:  # Assume OpenAI
            self.use_ollama = False
            self.client = openai.OpenAI()

    def run_retrieval_only(self, query: str) -> List[Dict[str, Any]]:
        """
        평가 목적으로 파이프라인의 검색 부분만 실행합니다.
        """
        # --- 1단계: 쿼리 재작성 ---
        rewritten_query = self.query_rewriter.rewrite_query(query)
        print(f"원본 쿼리: '{query}' -> 재작성된 쿼리: '{rewritten_query}'")

        tools = [cast(Any, get_tool_definition(self.tool_prompt_description))]

        try:
            if self.use_ollama:
                # Use Ollama for tool calling
                tools = [get_tool_definition(self.tool_prompt_description)]
                payload = {
                    "model": self.tool_calling_model,
                    "messages": [{"role": "user", "content": rewritten_query}],
                    "tools": tools,
                    "stream": False
                }
                response = requests.post(self.ollama_url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                response_message = result.get("message", {})
                tool_calls = response_message.get("tool_calls", [])
                if tool_calls:
                    tool_call = tool_calls[0]
                    tool_args_json = json.dumps(tool_call["function"]["arguments"])
                    tool_name = tool_call["function"]["name"]
                else:
                    tool_name = None
                    tool_args_json = "{}"
                    tool_args_json = "{}"
            else:
                # Use OpenAI for tool calling
                tools = [cast(Any, get_tool_definition(self.tool_prompt_description))]
                response = self.client.chat.completions.create(
                    model=self.tool_calling_model,
                    messages=[{"role": "user", "content": rewritten_query}],
                    tools=tools,
                    tool_choice="auto"
                )
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                if tool_calls:
                    tool_call = cast(ChatCompletionMessageToolCall, tool_calls[0])
                    tool_args_json = tool_call.function.arguments or "{}"
                    tool_name = tool_call.function.name
                else:
                    tool_name = None
                    tool_args_json = "{}"

            if tool_name:
                tool_result = self.dispatcher.execute_tool(
                    tool_name=tool_name,
                    tool_args_json=tool_args_json
                )
                return [{"standalone_query": rewritten_query, "docs": tool_result}]
            else:
                return [{"standalone_query": rewritten_query, "docs": []}]
        except Exception as e:
            print(f"'{query}' 쿼리에 대한 검색 중 오류 발생: {e}")
            return [{"standalone_query": rewritten_query, "docs": []}]

    def run(self, query: str) -> str:
        """
        주어진 사용자 쿼리에 대해 전체 RAG 파이프라인을 실행합니다.
        """
        retrieved_output = self.run_retrieval_only(query)

        # 검색 결과와 함께 재작성된 쿼리를 추출합니다.
        docs = retrieved_output[0].get("docs", [])
        standalone_query = retrieved_output[0].get("standalone_query", query)

        if docs:
            context_docs_content = [item.get('content', '') for item in docs]
            # 최종 답변 생성 시에도 명확성을 위해 재작성된 쿼리를 사용합니다.
            final_answer = self.generator.generate(query=standalone_query, context_docs=context_docs_content)
        else:
            final_answer = self.generator.generate(query=query, context_docs=[])

        return final_answer