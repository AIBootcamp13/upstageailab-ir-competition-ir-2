# scripts/validate_retrieval.py
import os
import sys
from tqdm import tqdm
import fire
import wandb # --- Phase 1: W&B 임포트 ---

def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def run(
    validation_path: str = "data/validation.jsonl",
    alpha: float = None,
    rerank_k: int = None
):
    _add_src_to_path()
    from ir_core.config import settings
    from ir_core.generation import get_generator
    from ir_core.orchestration.pipeline import RAGPipeline
    from ir_core.utils import read_jsonl
    from ir_core.evaluation.core import mean_average_precision

    print("--- Starting Validation Run ---")
    print(f"Using validation file: {validation_path}")

    if alpha is not None:
        settings.ALPHA = alpha
        print(f"Overriding alpha to: {settings.ALPHA}")
    if rerank_k is not None:
        settings.RERANK_K = rerank_k
        print(f"Overriding rerank_k to: {settings.RERANK_K}")

    # --- Phase 1: W&B 연동 ---
    wandb.init(
        project=settings.WANDB_PROJECT,
        name=f"validation-alpha-{settings.ALPHA}-rerank_k-{settings.RERANK_K}",
        config={
            "alpha": settings.ALPHA,
            "rerank_k": settings.RERANK_K,
            "embedding_model": settings.EMBEDDING_MODEL,
            "rewriter_model": settings.PIPELINE_REWRITER_MODEL,
            "validation_path": validation_path,
        }
    )

    generator = get_generator()
    pipeline = RAGPipeline(generator)

    try:
        validation_data = list(read_jsonl(validation_path))
    except FileNotFoundError:
        print(f"Error: Validation file not found at '{validation_path}'")
        wandb.finish()
        return

    all_results = []

    for item in tqdm(validation_data, desc="Validating Queries"):
        # --- Phase 2: 파이프라인 입력 변경 ---
        messages = item.get("msg", [])
        ground_truth_id = item.get("ground_truth_doc_id")

        if not messages or not ground_truth_id:
            continue

        # run_retrieval_only는 이제 messages 리스트를 입력으로 받음
        retrieval_output = pipeline.run_retrieval_only(messages)

        predicted_docs = []
        if retrieval_output:
            predicted_docs = retrieval_output[0].get("docs", [])

        predicted_ids = [doc["id"] for doc in predicted_docs]
        relevant_ids = [ground_truth_id]

        all_results.append((predicted_ids, relevant_ids))

    map_score = mean_average_precision(all_results)

    print("\n--- Validation Complete ---")
    print(f"Total Queries Validated: {len(all_results)}")
    print(f"MAP Score: {map_score:.4f}")
    print("---------------------------")

    # --- Phase 1: W&B에 결과 로깅 ---
    wandb.log({"map_score": map_score, "total_queries": len(all_results)})
    wandb.finish()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY is not set.")
            sys.exit(1)

    fire.Fire(run)
