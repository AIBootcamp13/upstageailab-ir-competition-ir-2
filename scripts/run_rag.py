# scripts/run_rag.py
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

import fire
import os
import sys

def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def run_pipeline(query: str, generator_type: str = None):
    _add_src_to_path()
    from ir_core.config import settings
    from ir_core.generation import get_generator
    from ir_core.orchestration.pipeline import RAGPipeline

    print("--- Initializing RAG System ---")

    if generator_type:
        print(f"Using generator_type override from CLI: '{generator_type}'")
        settings.GENERATOR_TYPE = generator_type

    try:
        generator = get_generator()
        print(f"Successfully initialized generator of type: '{settings.GENERATOR_TYPE}'")
    except ValueError as e:
        print(f"Error initializing generator: {e}")
        return

    pipeline = RAGPipeline(generator=generator)

    # --- Phase 2: 파이프라인 입력 변경 ---
    # CLI에서 받은 단일 쿼리를 대화 형식으로 변환
    messages = [{"role": "user", "content": query}]

    final_answer = pipeline.run(messages)

    print("\n===================================")
    print("Final Answer:")
    print(final_answer)
    print("===================================")


if __name__ == '__main__':
    _add_src_to_path()
    try:
        from dotenv import load_dotenv
        repo_root = os.path.dirname(os.path.dirname(__file__))
        env_path = os.path.join(repo_root, ".env")
        load_dotenv(env_path)
    except Exception:
        pass

    if (len(sys.argv) > 2 and 'openai' in sys.argv) or os.getenv("GENERATOR_TYPE", "openai") == "openai":
         if not os.getenv("OPENAI_API_KEY"):
              print("Error: OPENAI_API_KEY environment variable is not set.")
              sys.exit(1)

    fire.Fire(run_pipeline)
