# scripts/run_rag.py (top)
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

import fire
import os
import sys

# Add the src directory to the Python path to allow for absolute imports
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def run_pipeline(query: str, generator_type: str = None):
    """
    Initializes and runs the full RAG pipeline for a given query.

    This script demonstrates the end-to-end functionality of the RAG system,
    including tool calling and final answer generation.

    Args:
        query: The question to ask the RAG system.
        generator_type: The type of generator to use (e.g., 'openai' or 'ollama').
                        Overrides the value in the config/environment.
    """
    _add_src_to_path()

    # Import necessary components after setting up the path
    from ir_core.config import settings
    from ir_core.generation import get_generator
    from ir_core.orchestration.pipeline import RAGPipeline

    print("--- Initializing RAG System ---")

    # If a generator_type is passed via CLI, it overrides the settings
    if generator_type:
        print(f"Using generator_type override from CLI: '{generator_type}'")
        settings.GENERATOR_TYPE = generator_type

    # 1. Initialize the appropriate generator using our factory
    try:
        generator = get_generator()
        print(f"Successfully initialized generator of type: '{settings.GENERATOR_TYPE}'")
    except ValueError as e:
        print(f"Error initializing generator: {e}")
        return

    # 2. Initialize the main RAG pipeline with the chosen generator
    pipeline = RAGPipeline(generator=generator)

    # 3. Run the pipeline and print the final answer
    final_answer = pipeline.run(query)

    print("\n===================================")
    print("Final Answer:")
    print(final_answer)
    print("===================================")


if __name__ == '__main__':
    # Make sure src is on path so we can import the project's settings
    _add_src_to_path()

    # Try to load .env using python-dotenv if available; otherwise importing
    # the project's pydantic settings will cause .env to be read as well.
    try:
        from dotenv import load_dotenv

        # Load .env from the repo root
        repo_root = os.path.dirname(os.path.dirname(__file__))
        env_path = os.path.join(repo_root, ".env")
        load_dotenv(env_path)
    except Exception:
        # Fallback: import settings to trigger pydantic's env_file mechanism
        try:
            from ir_core.config import settings  # type: ignore
        except Exception:
            pass

    # Ensure OPENAI_API_KEY is set if using the openai generator
    if (len(sys.argv) > 2 and 'openai' in sys.argv) or os.getenv("GENERATOR_TYPE", "openai") == "openai":
         if not os.getenv("OPENAI_API_KEY"):
              print("Error: OPENAI_API_KEY environment variable is not set.")
              print("Please set it before running the pipeline with the OpenAI generator.")
              sys.exit(1)

    fire.Fire(run_pipeline)
