# scripts/evaluate.py
import os
import sys
from tqdm import tqdm
import fire
import wandb # --- Phase 1: W&B 임포트 ---
from typing import Optional

def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def run(eval_path: str = 'data/eval.jsonl', out: str = 'outputs/submission.csv', limit: Optional[int] = None, topk: int = 3):
    _add_src_to_path()
    from ir_core.config import settings
    from ir_core.orchestration.pipeline import RAGPipeline
    from ir_core.generation import get_generator
    from ir_core.utils import read_jsonl
    import json

    print(f"Starting evaluation run with file: {eval_path}")

    # --- Phase 1: W&B 연동 ---
    wandb.init(
        project=settings.WANDB_PROJECT,
        config={
            "eval_path": eval_path,
            "limit": limit,
            "topk": topk,
            "embedding_model": settings.EMBEDDING_MODEL,
            "rewriter_model": settings.PIPELINE_REWRITER_MODEL,
            "tool_calling_model": settings.PIPELINE_TOOL_CALLING_MODEL,
        }
    )

    try:
        generator = get_generator()
        pipeline = RAGPipeline(generator)
        print("RAG pipeline initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        wandb.finish()
        return

    output_dir = os.path.dirname(out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 이전 실행 결과 파일이 있다면 삭제
    if os.path.exists(out):
        os.remove(out)

    eval_items = list(read_jsonl(eval_path))
    if limit is not None:
        eval_items = eval_items[:int(limit)]
        print(f"Limiting evaluation to first {limit} items.")

    print(f"Processing {len(eval_items)} evaluation queries...")
    for item in tqdm(eval_items, desc="Evaluating Queries"):
        # --- Phase 2: 파이프라인 입력 변경 ---
        messages = item.get('msg', [])
        eval_id = item.get('eval_id')

        if not messages or eval_id is None:
            continue

        retrieval_out = pipeline.run_retrieval_only(messages)

        standalone_query = ""
        docs = []
        if retrieval_out:
            entry = retrieval_out[0]
            standalone_query = entry.get('standalone_query', "")
            docs = entry.get('docs', [])

        topk_ids = [d.get('id') for d in docs[:topk]]

        query_for_generation = standalone_query or (messages[-1].get("content") if messages else "")
        context_texts = [d.get('content', '') for d in docs[:topk]]

        try:
            # 잡담 케이스를 위해 분기 처리
            if not standalone_query:
                 answer_text = pipeline.generator.generate(
                    query=query_for_generation,
                    context_docs=[],
                    prompt_template_path="prompts/conversational_v1.jinja2"
                )
            else:
                answer_text = pipeline.generator.generate(query=query_for_generation, context_docs=context_texts)

        except Exception as e:
            print(f"Warning: generator failed for eval_id {eval_id}: {e}")
            answer_text = "답변을 생성하는 중 오류가 발생했습니다."

        references = [{"score": d.get("score"), "content": d.get("content", '')} for d in docs[:topk]]

        record = {
            "eval_id": eval_id,
            "standalone_query": standalone_query,
            "topk": topk_ids,
            "answer": answer_text,
            "references": references,
        }

        with open(out, 'a', encoding='utf-8') as outf:
            outf.write(json.dumps(record, ensure_ascii=False) + '\n')

    # --- Phase 1: W&B Artifact로 결과 저장 ---
    print(f"\nEvaluation complete. Submission records written to: {out}")
    artifact = wandb.Artifact('submission', type='dataset', description=f"Submission file from {eval_path}")
    artifact.add_file(out)
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == '__main__':
    if not os.getenv("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is not set.")
            sys.exit(1)

    fire.Fire(run)
