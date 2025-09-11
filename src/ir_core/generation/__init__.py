# src/ir_core/generation/__init__.py

from typing import TYPE_CHECKING
from .base import BaseGenerator
from .openai import OpenAIGenerator
from .ollama import OllamaGenerator

# TYPE_CHECKING 블록은 순환 참조 오류 없이 타입 힌트를 제공하기 위해 사용됩니다.
if TYPE_CHECKING:
    from omegaconf import DictConfig

def get_generator(cfg: "DictConfig") -> BaseGenerator:
    """
    Hydra 설정(cfg)을 기반으로 적절한 생성기 인스턴스를 생성하고 반환하는 팩토리 함수입니다.

    Args:
        cfg (DictConfig): Hydra에 의해 관리되는 전체 설정 객체.

    Raises:
        ValueError: cfg.pipeline.generator_type이 알 수 없는 값일 경우 발생합니다.

    Returns:
        BaseGenerator: 설정에 따라 초기화된 OpenAIGenerator 또는 OllamaGenerator 인스턴스.
    """
    # 설정에서 생성기 유형을 읽어옵니다.
    generator_type = cfg.pipeline.generator_type.lower()

    if generator_type == "openai":
        # OpenAI 생성기를 초기화할 때, 설정 파일의 상세 경로들을 전달합니다.
        return OpenAIGenerator(
            model_name=cfg.pipeline.generator_model_name,
            prompt_template_path=cfg.prompts.generation_qa,
            persona_path=cfg.prompts.persona
        )
    elif generator_type == "ollama":
        # Ollama 생성기 또한 일관성을 위해 설정 객체로부터 초기화되도록 수정합니다.
        return OllamaGenerator(
            model_name=cfg.pipeline.generator_model_name,
            prompt_template_path=cfg.prompts.generation_qa
        )
    else:
        raise ValueError(f"알 수 없는 생성기 유형입니다: '{generator_type}'")

# 이 패키지의 공개 API를 명시적으로 정의합니다.
__all__ = ["get_generator", "BaseGenerator"]