# src/ir_core/orchestration/pipeline.py
from typing import List, Dict, Any, Optional, cast
import os
import json
import openai
from openai.types.chat import ChatCompletionMessageToolCall

from ..config import settings
from ..generation.base import BaseGenerator
from ..tools.dispatcher import default_dispatcher, ToolDispatcher
from ..tools.retrieval_tool import get_tool_definition
# --- Phase 2: 새로 추가된 모듈 임포트 ---
from .query_rewriter import rewrite_query

class RAGPipeline:
    """
    전체 RAG 프로세스를 오케스트레이션합니다.
    """
    def __init__(self, generator: BaseGenerator, dispatcher: ToolDispatcher = default_dispatcher):
        self.generator = generator
        self.dispatcher = dispatcher
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY environment variable not set. Tool calling will fail.")
        self.client = openai.OpenAI()

    def run_retrieval_only(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        평가를 위해 파이프라인의 검색 부분만 실행합니다.
        대화 맥락을 이해하여 독립형 질의를 생성하고, 이를 기반으로 검색을 수행합니다.
        """
        # --- Phase 2: 질의 재구성 단계 추가 ---
        print("Step 1: Rewriting query from conversation history...")
        standalone_query = rewrite_query(messages, self.client)

        if not standalone_query or standalone_query == "__NO_QUERY__":
            print("Step 2: Non-scientific query detected. Skipping retrieval.")
            # 잡담이거나 검색이 불필요한 경우, 빈 standalone_query와 빈 문서를 반환
            return [{"standalone_query": "", "docs": []}]

        print(f"Step 2: Standalone query is '{standalone_query}'. Executing retrieval tool...")
        # --- Phase 2: 재구성된 질의로 도구 강제 호출 ---
        # 이제 LLM에게 도구 사용 여부를 '묻는' 대신, 재구성된 질의로 '직접' 도구를 호출합니다.
        # 이렇게 함으로써 LLM의 판단에 따른 변동성을 줄이고 안정적인 검색을 보장합니다.
        tool_name = "scientific_search"
        tool_args = {"query": standalone_query, "top_k": settings.PIPELINE_DEFAULT_TOP_K}
        tool_args_json = json.dumps(tool_args, ensure_ascii=False)

        tool_result = self.dispatcher.execute_tool(
            tool_name=tool_name,
            tool_args_json=tool_args_json
        )

        if isinstance(tool_result, list):
            return [{"standalone_query": standalone_query, "docs": tool_result}]
        else:
            # 도구 실행 실패 시
            print(f"Tool execution failed. Result: {tool_result}")
            return [{"standalone_query": standalone_query, "docs": []}]

    def run(self, messages: List[Dict[str, Any]]) -> str:
        """
        사용자 대화 기록에 대해 전체 RAG 파이프라인을 실행합니다.
        """
        print(f"\n--- Running RAG Pipeline for conversation ---")

        try:
            # Step 1 & 2: 대화 맥락을 분석하고 필요한 문서를 검색합니다.
            retrieval_results = self.run_retrieval_only(messages)

            retrieved_docs = []
            query_for_generation = messages[-1].get("content", "")

            if retrieval_results:
                result_item = retrieval_results[0]
                retrieved_docs = result_item.get("docs", [])
                # 답변 생성 시에는 재구성된 쿼리를 사용하는 것이 더 명확할 수 있습니다.
                if result_item.get("standalone_query"):
                    query_for_generation = result_item["standalone_query"]

            # Step 3: 검색된 문서(있거나 없거나)를 바탕으로 최종 답변을 생성합니다.
            if retrieved_docs:
                print(f"Step 3: Found {len(retrieved_docs)} documents. Generating answer with context...")
                context_docs = [item.get('content', '') for item in retrieved_docs]
                final_answer = self.generator.generate(query=query_for_generation, context_docs=context_docs)
            else:
                print("Step 3: No documents found or needed. Generating conversational answer...")
                # 검색 결과가 없을 때는 잡담용 프롬프트를 사용하여 답변 생성
                final_answer = self.generator.generate(
                    query=query_for_generation,
                    context_docs=[],
                    prompt_template_path="prompts/conversational_v1.jinja2" # 잡담용 프롬프트 지정
                )

            print("--- Pipeline finished ---")
            return final_answer

        except Exception as e:
            print(f"An error occurred during the RAG pipeline execution: {e}")
            return "Error: Could not complete the request due to an internal error."
