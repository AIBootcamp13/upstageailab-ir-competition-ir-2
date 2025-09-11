import os
from typing import List, Optional, Any, cast
import openai
import jinja2
from .base import BaseGenerator

class OpenAIGenerator(BaseGenerator):
    """
    A concrete implementation of the BaseGenerator for OpenAI models.

    This class interfaces with the OpenAI API to generate answers.
    It uses a Jinja2 template to construct the prompt.
    """
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        prompt_template_path: Optional[str] = None,
        client: Optional[openai.OpenAI] = None,
    ):
        """
        Initializes the OpenAI generator.

        Args:
            model_name: The name of the OpenAI model to use.
            prompt_template_path: Path to the Jinja2 prompt template file.
            client: An optional pre-configured OpenAI client instance.
        """
        from ..config import settings

        self.model_name = model_name
        self.client = client or openai.OpenAI()
        # Prefer the prompt template path from settings if not provided
        self.prompt_template_path = prompt_template_path or settings.PROMPT_TEMPLATE_PATH

        # Set up Jinja2 environment to load templates from the project root
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.getcwd()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Load default system persona message from env or file or settings
        self.default_persona = os.getenv("GENERATOR_SYSTEM_MESSAGE", "")
        # If env var not set, try reading from configured file
        if not self.default_persona:
            try:
                persona_file = getattr(settings, "GENERATOR_SYSTEM_MESSAGE_FILE", None)
                if persona_file and os.path.exists(persona_file):
                    with open(persona_file, "r", encoding="utf-8") as pf:
                        self.default_persona = pf.read()
            except Exception:
                self.default_persona = ""

        # Final fallback to settings.GENERATOR_SYSTEM_MESSAGE
        if not self.default_persona:
            self.default_persona = getattr(settings, "GENERATOR_SYSTEM_MESSAGE", "")

    def _render_prompt(self, query: str, context_docs: List[str], template_path: str) -> str:
        """지정된 템플릿 경로를 사용하여 프롬프트를 로드하고 렌더링합니다."""
        try:
            template = self.jinja_env.get_template(template_path)
            return template.render(query=query, context_docs=context_docs)
        except jinja2.TemplateNotFound:
            raise FileNotFoundError(
                f"프롬프트 템플릿을 '{template_path}'에서 찾을 수 없습니다."
            )


    def generate(
        self,
        query: str,
        context_docs: List[str],
        prompt_template_path: Optional[str] = None,
    ) -> str:
        """
        OpenAI Chat Completions API를 사용하여 답변을 생성합니다.
        이제 특정 프롬프트 템플릿을 동적으로 선택할 수 있습니다.
        """
        # 호출 시 특정 템플릿 경로가 제공되지 않으면, 인스턴스의 기본 경로를 사용합니다.
        template_path = prompt_template_path or self.prompt_template_path

        try:
            # 사용할 템플릿 경로를 명시적으로 전달합니다.
            full_prompt = self._render_prompt(query, context_docs, template_path)
        except FileNotFoundError as e:
            print(e)
            return "오류: 프롬프트 템플릿이 없어 답변을 생성할 수 없습니다."

        try:
            messages = [
                {"role": "system", "content": self.default_persona},
                {"role": "user", "content": full_prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(Any, messages),
                temperature=0.2,
            )
            return response.choices[0].message.content or "모델이 유효한 답변을 반환하지 않았습니다."

        except Exception as e:
            print(f"OpenAI API 호출 중 오류 발생: {e}")
            return "오류: 모델로부터 답변을 생성할 수 없습니다."

