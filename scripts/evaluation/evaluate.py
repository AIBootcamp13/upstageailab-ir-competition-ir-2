# scripts/evaluate.py

import os
import sys
import json
import csv
import math
import signal
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

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully but forcefully."""
    global shutdown_requested
    if shutdown_requested: # If Ctrl+C is pressed a second time
        print("\n🛑 Force exiting immediately.")
        os._exit(1) # Hard exit

    print("\n⚠️  Interrupt signal received. Forcing shutdown...")
    shutdown_requested = True

    try:
        # Attempt to cleanly finish the wandb run if it exists
        if wandb.run:
            print("   Cleaning up WandB run...")
            wandb.finish(exit_code=1)
    except Exception as e:
        print(f"   Could not clean up WandB run: {e}")
    finally:
        # Force exit the script
        print("✅ Shutdown complete.")
        sys.exit(0)

@hydra.main(config_path="../../conf", config_name="settings", version_base=None)
def run(cfg: DictConfig) -> None:
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _add_src_to_path()

    # Set evaluation mode environment variable to disable HyDE if configured
    os.environ['RAG_EVALUATION_MODE'] = 'true'

    # Check if shutdown was requested before starting
    if shutdown_requested:
        print("Shutdown requested before starting. Exiting...")
        return

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

    # Determine output path - use custom file if specified, otherwise use data config default
    if hasattr(cfg.evaluate, 'custom_output_file') and cfg.evaluate.custom_output_file:
        custom_file = cfg.evaluate.custom_output_file
        # If custom file doesn't have directory, put it in outputs/
        if '/' not in custom_file and '\\' not in custom_file:
            custom_file = f"outputs/{custom_file}"
        output_path = custom_file
        print(f"Using custom output file: {output_path}")
    else:
        output_path = cfg.data.output_path
        print(f"Using default output file: {output_path}")

    # Ask for confirmation
    import questionary
    confirmed = questionary.confirm(
        f"Generate output file: {output_path}?",
        default=True
    ).ask()

    if not confirmed:
        new_path = questionary.text(
            "Enter new output file path:",
            default=output_path
        ).ask()
        if new_path:
            output_path = new_path

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"기존 제출 파일 삭제: {output_path}")

    eval_items = list(read_jsonl(cfg.data.evaluation_path))
    if cfg.limit:
        if isinstance(cfg.limit, float) and 0 < cfg.limit < 1:
            # Treat as fraction
            limit_count = int(len(eval_items) * cfg.limit)
            print(f"평가를 처음 {limit_count}개 항목으로 제한합니다 (전체의 {cfg.limit} 비율).")
            eval_items = eval_items[:limit_count]
        else:
            # Treat as absolute number
            limit_count = int(cfg.limit)
            print(f"평가를 처음 {limit_count}개 항목으로 제한합니다.")
            eval_items = eval_items[:limit_count]

    print(f"{len(eval_items)}개의 평가 쿼리를 처리합니다...")

    def process_item(item, pipeline, cfg, index):
        global shutdown_requested

        # Check if shutdown was requested
        if shutdown_requested:
            return None

        query = item.get("msg", [{}])[-1].get("content", "") or item.get("query", "")
        eval_id = item.get("eval_id")
        if not query or eval_id is None:
            return None

        try:
            retrieval_out = pipeline.run_retrieval_only(query)
            standalone_query = query
            docs = []
            technique_used = 'none'
            if retrieval_out and isinstance(retrieval_out, list) and len(retrieval_out) > 0:
                retrieval_result = retrieval_out[0]
                if isinstance(retrieval_result, dict):
                    standalone_query = retrieval_result.get("standalone_query", query)
                    docs = retrieval_result.get("docs", [])
                    technique_used = retrieval_result.get("technique_used", 'none')

            # Ensure docs is a list of dictionaries
            if not isinstance(docs, list) or not all(isinstance(d, dict) for d in docs):
                print(f"Warning: Invalid docs format for eval_id {eval_id}: {type(docs)} - {docs}")
                docs = []

            topk_ids = [d.get("id") for d in docs[: cfg.evaluate.topk]]
            context_texts = [d.get("content", "") for d in docs[: cfg.evaluate.topk]]
            references = []
            for d in docs[: cfg.evaluate.topk]:
                score = d.get("score", 0.0)
                # Handle NaN values
                if score is not None and not (isinstance(score, float) and math.isnan(score)):
                    score = float(score)
                else:
                    score = 0.0

                ref_item = {
                    "score": score,
                    "content": d.get("content", "")
                }
                # Optional interpretability fields
                if "rrf_pct" in d:
                    try:
                        ref_item["rrf_pct"] = float(d.get("rrf_pct"))
                    except Exception:
                        pass
                if "sparse_rank" in d and d.get("sparse_rank") is not None:
                    ref_item["sparse_rank"] = d.get("sparse_rank")
                if "dense_rank" in d and d.get("dense_rank") is not None:
                    ref_item["dense_rank"] = d.get("dense_rank")

                references.append(ref_item)

            # Check again before generation
            if shutdown_requested:
                return None

            answer_text = pipeline.generator.generate(
                query=standalone_query, context_docs=context_texts
            )

            record = {
                "eval_id": eval_id,
                "standalone_query": standalone_query,
                "technique_used": technique_used,
                "topk": topk_ids,
                "answer": answer_text,
                "references": references,
                "_original_index": index,  # Preserve original order
            }
            return record

        except Exception as e:
            if shutdown_requested:
                return None
            print(f"경고: eval_id {eval_id}에 대한 검색 중 오류 발생: {e}")
            return None

    executor = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.evaluate.max_workers) as executor:
            # Use map with timeout to allow for cancellation
            future_to_item = {
                executor.submit(lambda item=item, idx=idx: process_item(item, pipeline, cfg, idx)): item
                for idx, item in enumerate(eval_items)
            }

            results = []
            for future in tqdm(concurrent.futures.as_completed(future_to_item),
                             desc="Evaluating Queries", total=len(eval_items)):
                if shutdown_requested:
                    print("\n🛑 Cancelling remaining tasks...")
                    executor.shutdown(wait=False)
                    break

                try:
                    result = future.result(timeout=60)  # 60 second timeout per task
                    if result:
                        results.append(result)
                except concurrent.futures.TimeoutError:
                    print(f"⚠️  Task timed out, skipping...")
                    continue
                except Exception as e:
                    print(f"⚠️  Task failed: {e}")
                    continue

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received. Cancelling all tasks...")
        if executor:
            executor.shutdown(wait=False)
        print("   Tasks cancelled. Cleaning up...")
        if wandb_run:
            wandb.finish()
        return

    # Sort results by original index to maintain order from evaluation file
    results.sort(key=lambda x: x['_original_index'] if x and '_original_index' in x else float('inf'))

    # Remove the temporary _original_index field before writing
    for record in results:
        if record and '_original_index' in record:
            del record['_original_index']

    # Write results: JSONL (preferred) or CSV based on extension or config
    if results:
        # Prefer JSONL if the path ends with .jsonl
        write_jsonl = output_path.lower().endswith('.jsonl')
        # Optional config override: cfg.evaluate.output_format
        try:
            fmt = getattr(cfg.evaluate, 'output_format', None)
            if fmt:
                write_jsonl = (str(fmt).lower() == 'jsonl')
        except Exception:
            pass

        if write_jsonl:
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in results:
                    if not record:
                        continue
                    # Build output object; omit technique_used in output
                    include_refs = False
                    include_ref_scores = False
                    try:
                        include_refs = bool(getattr(cfg.evaluate, 'include_references', False))
                        include_ref_scores = bool(getattr(cfg.evaluate, 'include_reference_scores', False))
                    except Exception:
                        pass

                    out_refs = []
                    if include_refs:
                        # Optionally include references; optionally drop scores
                        for r in (record.get('references', []) or []):
                            if include_ref_scores:
                                out_refs.append(r)
                            else:
                                out_refs.append({'content': r.get('content', '')})

                    out_obj = {
                        'eval_id': record.get('eval_id'),
                        'standalone_query': record.get('standalone_query', ''),
                        'topk': record.get('topk', []) or [],
                        'answer': record.get('answer', ''),
                        'references': out_refs
                    }
                    f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        else:
            fieldnames = ['eval_id', 'standalone_query', 'technique_used', 'topk', 'answer', 'references']
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for record in results:
                    if record:
                        # Convert complex fields to strings for CSV
                        csv_record = record.copy()
                        csv_record['topk'] = ','.join(record['topk']) if record.get('topk') else ''
                        csv_record['references'] = json.dumps(record['references'], ensure_ascii=False) if record.get('references') else ''
                        writer.writerow(csv_record)

    print(f"\n평가 완료. {len(results)}개의 쿼리가 처리되었습니다. 제출 기록이 다음 파일에 저장되었습니다: {output_path}")

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

    print("✅ Evaluation completed successfully.")

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    run()