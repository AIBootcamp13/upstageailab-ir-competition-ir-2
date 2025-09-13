# src/ir_core/generation/openai.py

import os
from typing import List, Optional, Any, cast
import openai
import jinja2
from .base import BaseGenerator

class OpenAIGenerator(BaseGenerator):
    """
    OpenAI 모델을 위한 BaseGenerator의 구체적인 구현체입니다.
    이 클래스는 OpenAI API와 상호 작용하여 답변을 생성하며,
    Jinja2 템플릿을 사용하여 프롬프트를 구성합니다.
    """
    def __init__(
        self,
        model_name: str,
        prompt_template_path: str,
        persona_path: Optional[str] = None,
        client: Optional[openai.OpenAI] = None,
    ):
        """
        OpenAI 생성기를 초기화합니다.

        Args:
            model_name (str): 사용할 OpenAI 모델의 이름.
            prompt_template_path (str): 답변 생성에 사용할 Jinja2 프롬프트 템플릿 파일 경로.
            persona_path (str, optional): 시스템 메시지로 사용할 페르소나 파일 경로. Defaults to None.
            client (Optional[openai.OpenAI], optional): 미리 설정된 OpenAI 클라이언트 인스턴스. Defaults to None.
        """
        self.model_name = model_name
        self.client = client or openai.OpenAI()
        self.prompt_template_path = prompt_template_path

        # Jinja2 환경을 설정하여 프로젝트 루트에서 템플릿을 로드합니다.
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.getcwd()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # --- 페르소나 로딩 로직 (수정됨) ---
        # 이제 명시적으로 제공된 persona_path에서 시스템 메시지를 로드합니다.
        self.default_persona = ""
        if persona_path and os.path.exists(persona_path):
            try:
                with open(persona_path, "r", encoding="utf-8") as pf:
                    self.default_persona = pf.read()
                print(f"'{persona_path}'에서 페르소나를 성공적으로 로드했습니다.")
            except Exception as e:
                print(f"경고: '{persona_path}'에서 페르소나를 로드하는 중 오류 발생: {e}")
        else:
            print("페르소나 경로가 제공되지 않았거나 파일이 존재하지 않습니다. 기본 시스템 메시지를 사용합니다.")


    def _render_prompt(self, query: str, context_docs: List[str], template_path: str) -> str:
        """지정된 경로의 Jinja2 프롬프트 템플릿을 로드하고 렌더링합니다."""
        try:
            template = self.jinja_env.get_template(template_path)
            return template.render(query=query, context_docs=context_docs)
        except jinja2.TemplateNotFound:
            raise FileNotFoundError(
                f"'{template_path}'에서 프롬프트 템플릿을 찾을 수 없습니다. "
                f"프로젝트 루트 기준 경로가 올바른지 확인하세요."
            )

    def generate(
        self,
        query: str,
        context_docs: List[str],
        prompt_template_path: Optional[str] = None,
    ) -> str:
        """
        템플릿화된 프롬프트를 사용하여 OpenAI Chat Completions API로 답변을 생성합니다.
        """
        # 호출 시 특정 템플릿 경로가 제공되면 그것을 사용하고, 아니면 인스턴스의 기본값을 사용합니다.
        template_to_use = prompt_template_path or self.prompt_template_path

        try:
            full_prompt = self._render_prompt(query, context_docs, template_to_use)
            # ===============================================================
            # print("==================== FINAL PROMPT TO LLM ====================")
            # print(full_prompt)
            # print("=============================================================")
            # ===============================================================
        except FileNotFoundError as e:
            print(e)
            return "오류: 프롬프트 템플릿을 찾을 수 없어 답변을 생성할 수 없습니다."

        try:
            messages = []
            if self.default_persona:
                messages.append({"role": "system", "content": self.default_persona})
            messages.append({"role": "user", "content": full_prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(Any, messages),
                temperature=0.2,
            )

            return response.choices[0].message.content or "모델이 답변을 생성하지 않았습니다."
        except Exception as e:
            print(f"OpenAI API 호출 중 오류 발생: {e}")
            return "오류: 모델로부터 답변을 생성하는 데 실패했습니다."