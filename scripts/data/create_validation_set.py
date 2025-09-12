# scripts/create_validation_set.py

import os
import sys
import json
import random
import asyncio
from tqdm.asyncio import tqdm_asyncio

import hydra
import openai
import jinja2  # Jinja2 템플릿 렌더링을 위해 임포트
from omegaconf import DictConfig


def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


# --- 비동기 질문 생성 함수 (수정됨) ---
# 이제 프롬프트 템플릿 문자열을 인자로 받습니다.
async def generate_question_for_document(
    client: openai.AsyncOpenAI, document_content: str, model: str, prompt_template: str
) -> str | None:
    """주어진 문서와 프롬프트 템플릿으로 비동기적으로 LLM을 사용하여 단일 질문을 생성합니다."""
    # Jinja2를 사용하여 문서 내용을 프롬프트 템플릿에 삽입합니다.
    rendered_prompt = jinja2.Template(prompt_template).render(
        document_content=document_content
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": rendered_prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        if response.choices:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  - 모델={model} 질문 생성 중 오류 발생: {e}")
    return None


@hydra.main(config_path="../conf", config_name="config", version_base=None)
async def run(cfg: DictConfig) -> None:
    """문서 샘플로부터 비동기적으로 검증 데이터셋을 생성합니다."""
    _add_src_to_path()
    from ir_core.utils import read_jsonl

    # --- 설정에서 값 불러오기 (수정됨) ---
    input_file = cfg.data.documents_path
    output_file = cfg.data.validation_path
    sample_size = cfg.create_validation_set.sample_size
    model_name = cfg.create_validation_set.llm_model
    prompt_path = (
        cfg.create_validation_set.prompt_path
    )  # 설정에서 프롬프트 경로를 읽어옵니다.

    print(f"'{input_file}' 파일로부터 검증 데이터셋 생성을 시작합니다...")
    print(f"'{prompt_path}' 프롬프트를 사용하여 {sample_size}개의 질문을 생성합니다.")

    if not os.getenv("OPENAI_API_KEY"):
        print("\n오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    # --- 프롬프트 파일 로드 (추가됨) ---
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template_content = f.read()
    except FileNotFoundError:
        print(f"오류: '{prompt_path}'에서 프롬프트 파일을 찾을 수 없습니다.")
        return

    client = openai.AsyncOpenAI()
    all_docs = list(read_jsonl(input_file))

    sample_docs = random.sample(all_docs, min(sample_size, len(all_docs)))

    # --- 비동기 작업 생성 (수정됨) ---
    # 이제 로드된 프롬프트 내용을 각 작업에 전달합니다.
    tasks = [
        generate_question_for_document(
            client, doc.get("content", ""), model_name, prompt_template_content
        )
        for doc in sample_docs
    ]

    questions = await tqdm_asyncio.gather(*tasks)

    validation_set = []
    for doc, question in zip(sample_docs, questions):
        if question:
            doc_id = doc.get("docid")
            validation_set.append(
                {
                    "eval_id": f"val_{doc_id}",
                    "msg": [{"role": "user", "content": question}],
                    "ground_truth_doc_id": doc_id,
                }
            )

    print(f"\n성공적으로 {len(validation_set)}개의 질문-문서 쌍을 생성했습니다.")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in validation_set:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"검증 데이터셋이 '{output_file}'에 저장되었습니다.")


if __name__ == "__main__":
    run()
