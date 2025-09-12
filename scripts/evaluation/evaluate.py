# scripts/evaluate.py

import os
import sys
import json
from tqdm import tqdm

# --- 새로운 임포트 (New Imports) ---
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# ------------------------------------

# OmegaConf가 ${env:VAR_NAME} 구문을 해석할 수 있도록 'env' 리졸버를 등록합니다.
OmegaConf.register_new_resolver("env", os.getenv)


# Add the src directory to the path to allow for project imports
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


# --- 유틸리티 임포트 (Utility Imports) ---
_add_src_to_path()
from ir_core.utils.wandb import generate_run_name


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    """
    Hydra 설정을 사용하여 RAG 파이프라인에 대한 평가를 실행하고,
    대회 제출 파일을 생성한 후 WandB에 아티팩트로 기록합니다.

    Args:
        cfg (DictConfig): Hydra에 의해 관리되는 설정 객체.
    """
    # --- WandB 초기화 (WandB Initialization) ---
    # generate_run_name 유틸리티를 재사용하지만, 접두사를 'eval'로 설정합니다.
    # OmegaConf.set_struct를 사용하여 cfg 객체를 임시로 수정 가능하게 만듭니다.
    OmegaConf.set_struct(cfg, False)
    cfg.wandb.run_name_prefix = cfg.wandb.run_name.evaluate
    run_name = generate_run_name(cfg)
    OmegaConf.set_struct(cfg, True)  # 다시 읽기 전용으로 변경

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        job_type="evaluation",  # 작업 유형을 'evaluation'으로 지정
    )
    print(f"평가 실행 시작: {cfg.data.evaluation_path}")
    print(f"적용된 설정:\n{OmegaConf.to_yaml(cfg)}")

    # Lazy imports after path is set
    from ir_core.orchestration.pipeline import RAGPipeline
    from ir_core.generation import get_generator
    from ir_core.utils import read_jsonl

    # 설정에서 지정된 경로의 도구 설명 프롬프트를 읽어옵니다.
    try:
        with open(cfg.prompts.tool_description, "r", encoding="utf-8") as f:
            tool_desc = f.read()
    except FileNotFoundError:
        print(
            f"오류: '{cfg.prompts.tool_description}'에서 도구 설명 파일을 찾을 수 없습니다."
        )
        wandb.finish()
        return

    # 1. QueryRewriter 인스턴스를 생성합니다.
    from ir_core.orchestration.rewriter import QueryRewriter

    query_rewriter = QueryRewriter(
        model_name=cfg.pipeline.rewriter_model,
        prompt_template_path=cfg.prompts.rephrase_query,
    )

    # 2. RAG 파이프라인을 초기화할 때 rewriter 인스턴스를 전달합니다.
    generator = get_generator(cfg)
    pipeline = RAGPipeline(
        generator=generator,
        query_rewriter=query_rewriter,
        tool_prompt_description=tool_desc,
    )

    # 출력 디렉토리가 존재하는지 확인
    output_path = cfg.data.output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 만약 이전에 생성된 제출 파일이 있다면 삭제하여 덮어쓰기를 방지합니다.
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"기존 제출 파일 삭제: {output_path}")

    # 2. 평가 파일의 각 항목 처리
    eval_items = list(read_jsonl(cfg.data.evaluation_path))

    # 샘플 제한 로직 적용
    if cfg.limit:
        print(f"평가를 처음 {cfg.limit}개 항목으로 제한합니다.")
        eval_items = eval_items[: cfg.limit]

    print(f"{len(eval_items)}개의 평가 쿼리를 처리합니다...")
    for item in tqdm(eval_items, desc="Evaluating Queries"):
        query = item.get("msg", [{}])[-1].get("content", "")
        eval_id = item.get("eval_id")

        if not query or eval_id is None:
            continue

        retrieval_out = pipeline.run_retrieval_only(query)

        # --- 포맷 수정 로직 (Format Correction Logic) ---
        # 파이프라인 출력에서 standalone_query와 docs를 안전하게 추출합니다.
        standalone_query = query  # 기본값은 원본 쿼리
        docs = []
        if retrieval_out and isinstance(retrieval_out, list) and retrieval_out[0]:
            retrieval_result = retrieval_out[0]
            standalone_query = retrieval_result.get("standalone_query", query)
            docs = retrieval_result.get("docs", [])
        # ----------------------------------------------------

        topk_ids = [d.get("id") for d in docs[: cfg.evaluate.topk]]
        context_texts = [d.get("content", "") for d in docs[: cfg.evaluate.topk]]
        references = [
            {"score": d.get("score", 0.0), "content": d.get("content", "")}
            for d in docs[: cfg.evaluate.topk]
        ]

        try:
            # 생성기는 재구성된 standalone_query를 사용해야 더 정확한 답변을 만듭니다.
            answer_text = pipeline.generator.generate(
                query=standalone_query, context_docs=context_texts
            )
        except Exception as e:
            print(f"경고: eval_id {eval_id}에 대한 답변 생성 실패: {e}")
            answer_text = ""

        # --- 최종 레코드 생성 (수정됨) ---
        # standalone_query 필드를 최종 제출 파일에 추가합니다.
        record = {
            "eval_id": eval_id,
            "standalone_query": standalone_query,
            "topk": topk_ids,
            "answer": answer_text,
            "references": references,
        }

        with open(output_path, "a", encoding="utf-8") as outf:
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n평가 완료. 제출 기록이 다음 파일에 저장되었습니다: {output_path}")
    submission_artifact = wandb.Artifact("submission", type="submission-file")
    submission_artifact.add_file(output_path)
    wandb.log_artifact(submission_artifact)
    print("제출 파일이 WandB 아티팩트로 성공적으로 로깅되었습니다.")

    wandb.finish()


if __name__ == "__main__":
    # .env 파일 로딩을 시도합니다 (OpenAI API 키 등을 위해).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    run()
