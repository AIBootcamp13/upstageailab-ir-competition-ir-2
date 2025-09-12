# scripts/execution/run_rag.py

import os
import sys

# --- 새로운 임포트 (New Imports) ---
import hydra
from omegaconf import DictConfig

from src.scripts_utils import add_src_to_path


# --- 유틸리티 임포트 (Utility Imports) ---
add_src_to_path()
# get_generator는 run_pipeline 함수 내부에서 임포트하여 순환 참조를 방지합니다.


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_pipeline(cfg: DictConfig) -> None:
    """
    Hydra 설정을 사용하여 주어진 쿼리에 대해 전체 RAG 파이프라인을 초기화하고 실행합니다.

    Args:
        cfg (DictConfig): Hydra에 의해 관리되는 설정 객체.
                           'query'는 커맨드라인에서 오버라이드해야 합니다.
    """
    from ir_core.generation import get_generator
    from ir_core.orchestration.pipeline import RAGPipeline

    # Hydra 설정에서 쿼리를 가져옵니다.
    query = cfg.get("query")
    if not query:
        print(
            "오류: 쿼리가 제공되지 않았습니다. 커맨드라인에서 'query=\"질문 내용\"' 형식으로 전달하세요."
        )
        return

    print("--- RAG 시스템 초기화 ---")

    # 1. 설정(cfg)을 전달하여 생성기를 초기화합니다. (오류 수정)
    try:
        generator = get_generator(cfg)
        print(
            f"'{cfg.pipeline.generator_type}' 유형의 생성기가 성공적으로 초기화되었습니다."
        )
    except ValueError as e:
        print(f"생성기 초기화 오류: {e}")
        return

    # 2. RAG 파이프라인을 초기화하고, 설정에서 도구 설명을 읽어옵니다.
    try:
        with open(cfg.prompts.tool_description, "r", encoding="utf-8") as f:
            tool_desc = f.read()
    except FileNotFoundError:
        print(
            f"오류: '{cfg.prompts.tool_description}'에서 도구 설명 파일을 찾을 수 없습니다."
        )
        return

    pipeline = RAGPipeline(generator=generator, tool_prompt_description=tool_desc)

    # 3. 파이프라인을 실행하고 최종 답변을 출력합니다.
    print(f"\n--- 쿼리 실행: '{query}' ---")
    final_answer = pipeline.run(query)

    print("\n===================================")
    print("최종 답변:")
    print(final_answer)
    print("===================================")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    run_pipeline()
