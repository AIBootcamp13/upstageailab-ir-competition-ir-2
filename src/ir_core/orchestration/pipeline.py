# src/ir_core/orchestration/pipeline.py
from typing import List, Dict, Any, cast, Optional
import os
import openai
from openai.types.chat import ChatCompletionMessageToolCall

# 이제 쿼리 분석기를 임포트합니다.
from .query_analyzer import default_query_analyzer, QueryAnalysisResult
from ..generation.base import BaseGenerator
from ..tools.dispatcher import default_dispatcher, ToolDispatcher
from ..tools.retrieval_tool import get_tool_definition

class RAGPipeline:
    """
    쿼리 분석 모듈을 포함하여 전체 RAG 프로세스를 조율합니다.

    이 파이프라인은 먼저 QueryAnalyzer를 사용하여 사용자의 의도를 파악하고,
    과학적 질문일 경우 독립적인 쿼리를 생성합니다. 그 후에 도구를 사용하거나
    직접 답변을 생성하는 흐름을 제어합니다.
    """
    def __init__(self, generator: BaseGenerator, dispatcher: ToolDispatcher = default_dispatcher):
        self.generator = generator
        self.dispatcher = dispatcher
        self.client = openai.OpenAI()
        self.query_analyzer = default_query_analyzer

    def run_retrieval_only(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        평가를 위해 파이프라인의 검색 부분만 실행합니다.

        1. 쿼리 분석기를 호출하여 의도와 독립 쿼리를 얻습니다.
        2. 의도가 'scientific_question'인 경우에만 도구 호출을 통해 문서를 검색합니다.
        3. 'chit_chat'인 경우 빈 결과를 반환합니다.

        Args:
            conversation_history: 전체 대화 기록.

        Returns:
            검색된 문서 딕셔너리 리스트 또는 빈 리스트.
        """
        # 1. 대화 분석
        analysis_result = self.query_analyzer.analyze(conversation_history)

        if analysis_result.intent != "scientific_question":
            # 과학적 질문이 아니면 검색을 수행하지 않고 빈 결과를 반환합니다.
            return []

        # 2. 독립 쿼리를 사용하여 도구 호출
        standalone_query = analysis_result.query
        tools = [cast(Any, get_tool_definition())]

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": standalone_query}],
                tools=tools,
                tool_choice="auto"
            )
            tool_calls = response.choices[0].message.tool_calls

            if tool_calls:
                tool_call = cast(ChatCompletionMessageToolCall, tool_calls[0])
                tool_result = self.dispatcher.execute_tool(
                    tool_name=tool_call.function.name,
                    tool_args_json=tool_call.function.arguments or "{}"
                )

                # 도구 결과에 분석된 독립 쿼리를 추가하여 반환합니다.
                if isinstance(tool_result, list):
                    return [{"standalone_query": standalone_query, "docs": tool_result}]

            # 도구 호출이 없거나 실패한 경우
            return []
        except Exception as e:
            print(f"검색 중 오류 발생 (쿼리: '{standalone_query}'): {e}")
            return []

    def run(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        전체 RAG 파이프라인을 실행합니다.
        """
        print(f"\n--- RAG 파이프라인 실행 (마지막 메시지: '{conversation_history[-1]['content']}') ---")

        # 1. 쿼리 분석 및 검색 실행
        retrieval_output = self.run_retrieval_only(conversation_history)

        # 2. 결과에 따라 최종 답변 생성
        if retrieval_output:
            # 문서가 성공적으로 검색된 경우
            docs = retrieval_output[0].get("docs", [])
            standalone_query = retrieval_output[0].get("standalone_query")
            context_docs = [doc.get('content', '') for doc in docs]

            print(f"분석된 쿼리 '{standalone_query}'로 {len(docs)}개의 문서를 찾았습니다. 답변을 생성합니다.")
            final_answer = self.generator.generate(
                query=standalone_query,
                context_docs=context_docs
            )
        else:
            # 검색이 필요 없거나(chit_chat) 실패한 경우, 대화형으로 답변
            print("과학적 질문이 아니거나 검색 결과가 없습니다. 대화형으로 답변합니다.")
            # 마지막 사용자 메시지를 사용하여 대화형 답변 생성
            last_user_message = conversation_history[-1]['content']
            final_answer = self.generator.generate(
                query=last_user_message,
                context_docs=[],
                prompt_template_path="prompts/conversational_v1.jinja2" # 대화형 프롬프트 사용
            )

        print("--- 파이프라인 종료 ---")
        return final_answer
