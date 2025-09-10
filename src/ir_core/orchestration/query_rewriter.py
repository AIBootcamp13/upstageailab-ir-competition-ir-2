# src/ir_core/orchestration/query_rewriter.py
import os
from typing import List, Dict, Any, Optional
import openai
import jinja2
from ..config import settings

# --- Phase 2: 질의 재구성 모듈 구현 ---

# Jinja2 환경은 한 번만 설정
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.getcwd()),
    trim_blocks=True,
    lstrip_blocks=True,
)

def _render_prompt(messages: List[Dict[str, str]]) -> str:
    """Jinja2 템플릿을 사용하여 질의 재구성 프롬프트를 렌더링합니다."""
    try:
        template = jinja_env.get_template(settings.PROMPT_REPHRASE_QUERY)
        return template.render(messages=messages)
    except jinja2.TemplateNotFound:
        raise FileNotFoundError(
            f"질의 재구성 프롬프트 템플릿을 찾을 수 없습니다: '{settings.PROMPT_REPHRASE_QUERY}'"
        )

def rewrite_query(
    messages: List[Dict[str, Any]],
    client: Optional[openai.OpenAI] = None
) -> str:
    """
    대화 기록을 바탕으로 독립형 검색 질의(standalone query)를 생성합니다.

    Args:
        messages: 사용자와 어시스턴트 간의 대화 기록 리스트.
        client: (선택) OpenAI 클라이언트 인스턴스.

    Returns:
        재구성된 독립형 질의 문자열. 과학적 질문이 아니면 "__NO_QUERY__"를 반환.
    """
    client = client or openai.OpenAI()

    try:
        prompt = _render_prompt(messages)

        response = client.chat.completions.create(
            model=settings.PIPELINE_REWRITER_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # 재현 가능하고 일관된 출력을 위해 온도를 0으로 설정
            max_tokens=200,
        )

        rewritten_query = response.choices[0].message.content.strip()
        print(f"Rewritten Query: '{rewritten_query}'") # 디버깅용 로그
        return rewritten_query

    except FileNotFoundError as e:
        print(f"Error: {e}")
        # 프롬프트 파일을 찾지 못하면 마지막 사용자 메시지를 그대로 반환
        return messages[-1].get("content", "")
    except Exception as e:
        print(f"An error occurred during query rewriting: {e}")
        # 에러 발생 시, 안전하게 마지막 사용자 메시지를 반환
        return messages[-1].get("content", "")
