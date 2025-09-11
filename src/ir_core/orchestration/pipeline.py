# src/ir_core/orchestration/pipeline.py

from typing import List, Dict, Any, cast
import os
import json
import openai
from openai.types.chat import ChatCompletionMessageToolCall

from ..generation.base import BaseGenerator
from ..tools.dispatcher import default_dispatcher, ToolDispatcher
from ..tools.retrieval_tool import get_tool_definition # get_tool_definition 임포트는 유지

class RAGPipeline:
    """
    전체 RAG 프로세스를 조율합니다.
    """
    # --- __init__ 메소드 변경됨 ---
    # 이제 'tool_prompt_description'을 인자로 받아 self.tool_prompt_description에 저장합니다.
    def __init__(
        self,
        generator: BaseGenerator,
        tool_prompt_description: str,
        dispatcher: ToolDispatcher = default_dispatcher
    ):
        """
        RAG 파이프라인을 초기화합니다.

        Args:
            generator: 생성기 클래스의 인스턴스 (예: OpenAIGenerator).
            tool_prompt_description: 도구 설명에 사용될 프롬프트 문자열.
            dispatcher: 도구 실행을 위한 ToolDispatcher 인스턴스.
        """
        self.generator = generator
        self.dispatcher = dispatcher
        self.tool_prompt_description = tool_prompt_description # 도구 설명을 인스턴스 변수로 저장
        self.client = openai.OpenAI()

    def run_retrieval_only(self, query: str) -> List[Dict[str, Any]]:
        """
        평가 목적으로 파이프라인의 검색 부분만 실행합니다.
        """
        # --- 도구 정의 호출 변경됨 ---
        # 저장된 도구 설명을 get_tool_definition 함수에 전달합니다.
        tools = [cast(Any, get_tool_definition(self.tool_prompt_description))]

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # Note: This could also be made configurable
                messages=[{"role": "user", "content": query}],
                tools=tools,
                tool_choice="auto"
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                tool_call = cast(ChatCompletionMessageToolCall, tool_calls[0])
                tool_args_json = tool_call.function.arguments or "{}"
                tool_args = json.loads(tool_args_json)
                standalone_query = tool_args.get("query", query)

                tool_result = self.dispatcher.execute_tool(
                    tool_name=tool_call.function.name,
                    tool_args_json=tool_args_json
                )

                return [{"standalone_query": standalone_query, "docs": tool_result}]
            else:
                return []
        except Exception as e:
            print(f"'{query}' 쿼리에 대한 검색 중 오류 발생: {e}")
            return []

    def run(self, query: str) -> str:
        """
        주어진 사용자 쿼리에 대해 전체 RAG 파이프라인을 실행합니다.
        """
        retrieved_output = self.run_retrieval_only(query)

        if retrieved_output:
            docs = retrieved_output[0].get("docs", [])
            context_docs_content = [item.get('content', '') for item in docs]
            final_answer = self.generator.generate(query=query, context_docs=context_docs_content)
        else:
            final_answer = self.generator.generate(query=query, context_docs=[])

        return final_answer