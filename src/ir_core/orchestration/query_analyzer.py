# src/ir_core/orchestration/query_analyzer.py
"""
대화형 쿼리 분석을 전담하는 모듈입니다.
이 모듈은 LLM을 사용하여 대화의 의도를 분류하고, 검색에 적합한
독립형 쿼리를 생성하는 로직을 포함합니다.
"""
import os
import json
from typing import List, Dict, Literal
import openai
from pydantic import BaseModel, Field
from jinja2 import Environment, FileSystemLoader

# 분석 결과의 데이터 구조를 정의하는 Pydantic 모델
class QueryAnalysisResult(BaseModel):
    """쿼리 분석 결과를 담는 데이터 모델입니다."""
    intent: Literal["scientific_question", "chit_chat"] = Field(
        ...,
        description="분석된 사용자의 의도. 'scientific_question' 또는 'chit_chat' 중 하나입니다."
    )
    query: str = Field(
        ...,
        description="의도가 'scientific_question'일 경우 생성된 독립형 검색 쿼리입니다."
    )

class QueryAnalyzer:
    """
    LLM을 사용하여 대화 내역을 분석하고 의도 분류 및 쿼리 재작성을 수행합니다.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        self.client = openai.OpenAI()
        self.model_name = model_name

        # Jinja2 템플릿 로더 설정
        self.jinja_env = Environment(
            loader=FileSystemLoader(os.path.join(os.getcwd(), "prompts")),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def analyze(self, conversation_history: List[Dict[str, str]]) -> QueryAnalysisResult:
        """
        주어진 대화 내역을 분석하여 QueryAnalysisResult를 반환합니다.

        Args:
            conversation_history: 사용자와 어시스턴트 간의 대화 기록.

        Returns:
            분석 결과 (의도 및 재작성된 쿼리).
        """
        try:
            # 프롬프트 템플릿을 로드하고 렌더링합니다.
            template = self.jinja_env.get_template("query_analyzer.jinja2")
            prompt = template.render(conversation_history=conversation_history)

            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # 일관된 결과를 위해 온도를 0으로 설정
                response_format={"type": "json_object"}, # JSON 출력 모드 사용
            )

            # 응답 파싱 및 Pydantic 모델로 변환
            response_content = response.choices[0].message.content
            analysis_data = json.loads(response_content)

            return QueryAnalysisResult(**analysis_data)

        except openai.APIError as e:
            print(f"OpenAI API 오류 발생: {e}")
            # API 오류 시, 보수적으로 chit_chat으로 처리하여 불필요한 검색을 방지합니다.
            return QueryAnalysisResult(intent="chit_chat", query="")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"분석 결과 파싱 오류: {e}")
            return QueryAnalysisResult(intent="chit_chat", query="")

# 싱글톤 인스턴스를 생성하여 애플리케이션 전체에서 재사용
default_query_analyzer = QueryAnalyzer()
