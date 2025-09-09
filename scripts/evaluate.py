# scripts/evaluate.py
"""
Runs the evaluation process for the RAG pipeline.

This script iterates through an evaluation file (e.g., data/eval.jsonl),
runs the retrieval part of the RAG pipeline for each query, and generates
a submission file in the format required by the competition.
"""
import os
import sys
from tqdm import tqdm
import fire

# Add the src directory to the path to allow for project imports
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def run(eval_path: str = 'data/eval.jsonl', out: str = 'outputs/submission.csv'):
    """
    Executes the evaluation against the RAG pipeline.

    Args:
        eval_path: Path to the evaluation JSONL file.
        out: Path to write the final CSV submission file.
    """
    _add_src_to_path()

    # Lazy imports after path is set
    from ir_core.orchestration.pipeline import RAGPipeline
    from ir_core.generation import get_generator
    from ir_core.utils import read_jsonl

    print(f"Starting evaluation run with file: {eval_path}")

    # 1. Initialize the RAG pipeline.
    # We only need the pipeline for its retrieval logic, but it requires a generator.
    try:
        # The generator type is read from settings (.env file or environment)
        generator = get_generator()
        pipeline = RAGPipeline(generator)
        print("RAG pipeline initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 2. Process each item in the evaluation file
    eval_items = list(read_jsonl(eval_path))
    submission_rows = []

    print(f"Processing {len(eval_items)} evaluation queries...")
    for item in tqdm(eval_items, desc="Evaluating Queries"):
        # The last message in the list is the user's query
        query = item.get('msg', [{'content': ''}])[-1].get('content', '')
        eval_id = item.get('eval_id')

        if not query or eval_id is None:
            continue

        # 3. Use the pipeline's dedicated retrieval method
        # This correctly handles the logic of deciding whether to search or not.
        retrieved_docs = pipeline.run_retrieval_only(query)

        # 4. Extract the document IDs for the submission file
        # The tool returns a list of dicts with 'content', but the original
        # hits from hybrid_retrieve included the full ES hit with '_id'.
        # For now, we assume the evaluation needs the original doc IDs.
        # This part requires an adjustment in the retrieval tool or pipeline
        # to pass through the original doc IDs.
        # TEMPORARY FIX: For now, we will rely on hybrid_retrieve directly
        # but this highlights a needed change for the pipeline to support evaluation.
        # Let's revert to a more direct retrieval for the script to work now.

        # NOTE: Reverting to direct retrieval call to generate submission.
        # This is a temporary measure. The RAG pipeline should be updated
        # to return document IDs for proper evaluation.
        from ir_core.retrieval import hybrid_retrieve

        # Use the low-level hybrid retrieve to get hits with IDs
        hits = hybrid_retrieve(query)
        predicted_ids = [h.get("hit", {}).get("_id", "") for h in hits]

        submission_rows.append({'eval_id': eval_id, 'predicted': predicted_ids})

    # 5. Write the final submission file
    try:
        with open(out, 'w', encoding='utf-8') as f:
            f.write('eval_id,predicted\n')
            for row in submission_rows:
                # Format with spaces as required by the competition
                ids_str = ' '.join(row['predicted'])
                f.write(f"{row['eval_id']},\"{ids_str}\"\n")
        print(f"\nEvaluation complete. Submission file generated at: {out}")
    except IOError as e:
        print(f"Error writing submission file: {e}")


if __name__ == '__main__':
    # Ensure OPENAI_API_KEY is available as the pipeline needs it for decisions.
    if not os.getenv("OPENAI_API_KEY"):
        # Attempt to load from .env file for convenience
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is not set.")
            print("Please set it before running the evaluation script.")
            sys.exit(1)

    fire.Fire(run)
