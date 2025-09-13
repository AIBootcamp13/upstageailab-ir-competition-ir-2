# src/ir_core/orchestration/rewriter.py

import os
from typing import List, Dict, Optional, Any, cast
from abc import ABC, abstractmethod

import openai
import jinja2

class BaseQueryRewriter(ABC):
    """
    Base class for query rewriters.
    """
    @abstractmethod
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a query.

        Args:
            query: The original query

        Returns:
            The rewritten query
        """
        pass

    def _is_query_too_different(self, original: str, rewritten: str) -> bool:
        """
        재작성된 쿼리가 원본과 너무 다른지 확인합니다.
        과학 용어의 보존을 우선으로 합니다.
        """
        # 과학 관련 키워드가 유지되는지 확인
        scientific_keywords = [
            '알파', '베타', '감마', 'RNA', 'DNA', '단백질', '원자', '분자', '세포', '유전자',
            '태양계', '은하계', '블랙홀', '중력', '전자', '양성자', '중성자', '원소', '화합물',
            '반응', '합성', '분해', '산화', '환원', '결합', '분자량', '밀도', '압력', '온도',
            '난백', '가금류', '알', '세포', '조직', '기관', '계통', '생물', '생명', '진화',
            '유전', '변이', '선택', '적응', '생태계', '환경', '에너지', '물질', '힘', '운동',
            '물리', '화학', '생물학', '지구과학', '천문학', '물', '공기', '불', '흙', '금속',
            '비금속', '산', '염기', '염', '용액', '용매', '용질', '침전', '증발', '응축'
        ]

        original_keywords = [kw for kw in scientific_keywords if kw in original]
        rewritten_keywords = [kw for kw in scientific_keywords if kw in rewritten]

        # 원본에 있던 과학 키워드의 50% 이상이 유지되어야 함
        if original_keywords and len(set(rewritten_keywords) & set(original_keywords)) / len(original_keywords) < 0.5:
            return True

        # 길이 차이가 너무 크면 의심
        len_ratio = len(rewritten) / len(original)
        if len_ratio < 0.5 or len_ratio > 2.0:
            return True

        # 공통 단어의 비율이 너무 낮으면 의심
        original_words = set(original.split())
        rewritten_words = set(rewritten.split())
        common_words = original_words & rewritten_words
        if len(common_words) / len(original_words) < 0.3:
            return True

        return False

class QueryRewriter(BaseQueryRewriter):
    """
    사용자의 대화형 질문을 검색에 최적화된 독립형 질문으로 재작성합니다.
    """
    def __init__(
        self,
        model_name: str,
        prompt_template_path: str,
        client: Optional[openai.OpenAI] = None,
        max_tokens: int = 150,
        temperature: float = 0.0,
    ):
        """
        QueryRewriter를 초기화합니다.

        Args:
            model_name (str): 재작성에 사용할 LLM 모델 이름.
            prompt_template_path (str): 재작성에 사용할 Jinja2 프롬프트 템플릿 경로.
            client (Optional[openai.OpenAI], optional): 미리 설정된 OpenAI 클라이언트.
            max_tokens (int): 최대 토큰 수.
            temperature (float): 온도 설정.
        """
        self.model_name = model_name
        self.client = client or openai.OpenAI()
        self.prompt_template_path = prompt_template_path
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Jinja2 환경을 설정하여 프롬프트 템플릿을 로드합니다.
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.getcwd()),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        try:
            self.prompt_template = self.jinja_env.get_template(self.prompt_template_path)
        except jinja2.TemplateNotFound:
            raise FileNotFoundError(f"'{self.prompt_template_path}'에서 재작성 프롬프트 템플릿을 찾을 수 없습니다.")

    def rewrite_query(self, query: str) -> str:
        """
        단일 사용자 질문(쿼리)을 재작성합니다.
        현재는 간단한 대화 내역을 구성하여 프롬프트에 전달합니다.

        Args:
            query (str): 사용자의 마지막 질문.

        Returns:
            str: 재작성된 독립형 질문.
        """
        # 프롬프트는 대화 내역(conversation_history)을 JSON 형식으로 기대합니다.
        conversation_history = [{"role": "user", "content": query}]

        rendered_prompt = self.prompt_template.render(conversation_history=conversation_history)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": rendered_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            rewritten_query = response.choices[0].message.content or query
            # LLM이 추가적인 따옴표를 출력하는 경우를 대비해 제거합니다.
            rewritten_query = rewritten_query.strip().strip('"')

            # 기본적인 검증: 재작성된 쿼리가 너무 짧거나 원본과 너무 다르면 원본 사용
            if len(rewritten_query) < 5 or self._is_query_too_different(query, rewritten_query):
                print(f"재작성된 쿼리가 부적절하여 원본 쿼리를 사용합니다: '{rewritten_query}'")
                return query

            return rewritten_query
        except Exception as e:
            print(f"쿼리 재작성 중 오류 발생: {e}")
            # 실패 시 원본 쿼리를 그대로 반환합니다.
            return query