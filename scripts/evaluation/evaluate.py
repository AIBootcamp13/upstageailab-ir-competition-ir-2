# scripts/evaluate.py

import os
import sys
import json
import concurrent.futures
from typing import cast
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.errors import CommError # Import the specific error type

# Register 'env' resolver for OmegaConf
OmegaConf.register_new_resolver("env", os.getenv)

def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from ir_core.utils.wandb import generate_run_name

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    _add_src_to_path()

    from ir_core.config import settings

    # Set Wandb timeouts as environment variables before initialization
    os.environ["WANDB_TIMEOUT"] = "30"
    os.environ["WANDB_HTTP_TIMEOUT"] = "30"

    # Generate run name once
    OmegaConf.set_struct(cfg, False)
    cfg.wandb.run_name_prefix = cfg.wandb.run_name.evaluate
    run_name = generate_run_name(cfg)
    OmegaConf.set_struct(cfg, True)

    # Prepare a smaller config for Wandb
    wandb_config = {
        "model": {
            "embedding_model": cfg.model.embedding_model,
            "alpha": cfg.model.alpha,
            "rerank_k": cfg.model.rerank_k
        },
        "pipeline": {
            "generator_type": cfg.pipeline.generator_type,
            "generator_model_name": cfg.pipeline.generator_model_name,
            "query_rewriter_type": cfg.pipeline.query_rewriter_type,
            "rewriter_model": cfg.pipeline.rewriter_model
        },
        "infrastructure": {
            "index_name": cfg.data.index_name if hasattr(cfg.data, 'index_name') else settings.INDEX_NAME,
            "es_host": settings.ES_HOST,
            "redis_url": settings.REDIS_URL
        },
        "translation": {
            "enabled": getattr(settings, 'translation', {}).get('enabled', False),
            "source_lang": getattr(settings, 'translation', {}).get('source_lang', 'ko'),
            "target_lang": getattr(settings, 'translation', {}).get('target_lang', 'en')
        },
        "limit": cfg.limit,
        "evaluate": {"topk": cfg.evaluate.topk}
    }

    wandb_run = None
    try:
        # Attempt online initialization first
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=wandb_config,
            job_type="evaluation"
        )
        wandb_run = wandb.run
        print("✅ WandB initialized in online mode.")

    except (CommError, Exception) as e:
        # If any error occurs, log it and switch to offline mode
        print(f"⚠️ WandB initialization failed ({e}). Attempting offline mode.")
        if wandb_run:
            wandb.finish() # Finish any partial runs if a connection was made then lost

        # Ensure a unique name for the offline run
        offline_run_name = f"{run_name}_offline"
        os.environ["WANDB_MODE"] = "offline"

        try:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=offline_run_name,
                config=wandb_config,
                job_type="evaluation",
                mode="offline"
            )
            wandb_run = wandb.run
            print("✅ WandB initialized in offline mode.")
        except Exception as offline_e:
            print(f"❌ Failed to initialize WandB even in offline mode: {offline_e}")
            wandb_run = None

    # The rest of your script remains the same, but now it references wandb_run correctly

    print(f"평가 실행 시작: {cfg.data.evaluation_path}")
    print(f"적용된 설정:\n{OmegaConf.to_yaml(cfg)}")

    # Validate configuration
    validation_warnings = settings.validate_configuration()
    if validation_warnings:
        print("\n⚠️  구성 검증 경고:")
        for warning in validation_warnings:
            print(f"  {warning}")
        print()

    from ir_core.orchestration.pipeline import RAGPipeline
    from ir_core.generation import get_generator, get_query_rewriter
    from ir_core.utils import read_jsonl

    try:
        with open(cfg.prompts.tool_description, "r", encoding="utf-8") as f:
            tool_desc = f.read()
    except FileNotFoundError:
        print(f"오류: '{cfg.prompts.tool_description}'에서 도구 설명 파일을 찾을 수 없습니다.")
        if wandb_run:
            wandb.finish()
        return

    query_rewriter = get_query_rewriter(cfg)
    generator = get_generator(cfg)
    pipeline = RAGPipeline(
        generator=generator,
        query_rewriter=query_rewriter,
        tool_prompt_description=tool_desc,
        tool_calling_model=cfg.pipeline.tool_calling_model,
    )

    output_path = cfg.data.output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"기존 제출 파일 삭제: {output_path}")

    eval_items = list(read_jsonl(cfg.data.evaluation_path))
    if cfg.limit:
        print(f"평가를 처음 {cfg.limit}개 항목으로 제한합니다.")
        eval_items = eval_items[: cfg.limit]

    print(f"{len(eval_items)}개의 평가 쿼리를 처리합니다...")

    def process_item(item, pipeline, cfg):
        query = item.get("msg", [{}])[-1].get("content", "")
        eval_id = item.get("eval_id")
        if not query or eval_id is None:
            return None
        retrieval_out = pipeline.run_retrieval_only(query)
        standalone_query = query
        docs = []
        if retrieval_out and isinstance(retrieval_out, list) and retrieval_out[0]:
            retrieval_result = retrieval_out[0]
            standalone_query = retrieval_result.get("standalone_query", query)
            docs = retrieval_result.get("docs", [])

        # Ensure docs is a list of dictionaries
        if not isinstance(docs, list) or not all(isinstance(d, dict) for d in docs):
            print(f"Warning: Invalid docs format for eval_id {eval_id}: {type(docs)} - {docs}")
            docs = []

        topk_ids = [d.get("id") for d in docs[: cfg.evaluate.topk]]
        context_texts = [d.get("content", "") for d in docs[: cfg.evaluate.topk]]
        references = [
            {"score": d.get("score", 0.0), "content": d.get("content", "")}
            for d in docs[: cfg.evaluate.topk]
        ]

        try:
            answer_text = pipeline.generator.generate(
                query=standalone_query, context_docs=context_texts
            )
        except Exception as e:
            print(f"경고: eval_id {eval_id}에 대한 답변 생성 실패: {e}")
            answer_text = ""

        record = {
            "eval_id": eval_id,
            "standalone_query": standalone_query,
            "topk": topk_ids,
            "answer": answer_text,
            "references": references,
        }
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.evaluate.max_workers) as executor:
        results = list(tqdm(executor.map(lambda item: process_item(item, pipeline, cfg), eval_items), desc="Evaluating Queries", total=len(eval_items)))

    for record in results:
        if record:
            with open(output_path, "a", encoding="utf-8") as outf:
                outf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n평가 완료. 제출 기록이 다음 파일에 저장되었습니다: {output_path}")

    if wandb_run:
        try:
            submission_artifact = wandb.Artifact("submission", type="submission-file")
            submission_artifact.add_file(output_path)
            wandb.log_artifact(submission_artifact)
            print("제출 파일이 WandB 아티팩트로 성공적으로 로깅되었습니다.")
        except Exception as e:
            print(f"⚠️ WandB 아티팩트 업로드 실패: {e}")
            print("제출 파일은 로컬에 저장되었습니다.")

    if wandb_run:
        wandb.finish()

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    run()