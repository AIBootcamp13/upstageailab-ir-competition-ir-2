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

from typing import Optional


def run(eval_path: str = 'data/eval.jsonl', out: str = 'outputs/submission.jsonl', limit: Optional[int] = None, topk: int = 3):
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
    import json

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
    # Allow quick testing by limiting the number of evaluation items processed
    if limit is not None:
        try:
            limit = int(limit)
            eval_items = eval_items[:limit]
            print(f"Limiting evaluation to first {limit} items for testing.")
        except Exception:
            print("Warning: Could not parse 'limit' into int; ignoring limit.")
    submission_rows = []

    print(f"Processing {len(eval_items)} evaluation queries...")
    for item in tqdm(eval_items, desc="Evaluating Queries"):
        # The last message in the list is the user's query
        query = item.get('msg', [{'content': ''}])[-1].get('content', '')
        eval_id = item.get('eval_id')

        if not query or eval_id is None:
            continue

        # 3. Use the pipeline's dedicated retrieval method to get tool results.
        # The updated pipeline returns a list containing a dict with
        # 'standalone_query' and 'docs' keys when a tool is called.
        retrieval_out = pipeline.run_retrieval_only(query)

        standalone_query = query
        docs = []
        if retrieval_out:
            # We expect retrieval_out like: [{"standalone_query":..., "docs": [...] }]
            entry = retrieval_out[0]
            standalone_query = entry.get('standalone_query', query)
            docs = entry.get('docs', [])

        # topk ids for submission
        topk_ids = [d.get('id') for d in docs[:topk]]

        # Generate an answer using the generator (pass the content as context)
        context_texts = [d.get('content', '') for d in docs[:topk]]
        try:
            answer_text = pipeline.generator.generate(query=standalone_query, context_docs=context_texts)
        except Exception as e:
            print(f"Warning: generator failed for eval_id {eval_id}: {e}")
            answer_text = ""

        # Build references list with score and content where available
        references = []
        for d in docs[:topk]:
            references.append({
                "score": d.get('score'),
                "content": d.get('content', '')
            })

        record = {
            "eval_id": eval_id,
            "standalone_query": standalone_query,
            "topk": topk_ids,
            "answer": answer_text,
            "references": references,
        }

        # Write a JSONL line immediately to avoid keeping everything in memory
        with open(out, 'a', encoding='utf-8') as outf:
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 5. Final message
    print(f"\nEvaluation complete. Submission records written to: {out}")


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
