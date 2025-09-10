# scripts/validate_retrieval.py
"""
Runs the RAG retrieval pipeline against a validation set and calculates the MAP score.

This script is the primary tool for tuning and evaluating different retrieval
strategies and hyperparameters (e.g., reranking alpha, chunking methods)
before generating a final submission.
"""
import os
import sys
from tqdm import tqdm
import fire

# Add the src directory to the Python path
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def run(
    validation_path: str = "data/validation.jsonl",
    alpha: float = None, # Allow overriding the default alpha
    rerank_k: int = None # Allow overriding the rerank_k
):
    """
    Evaluates the retrieval performance on a validation set using the MAP score.

    Args:
        validation_path: Path to the validation JSONL file.
        alpha: The alpha value for hybrid retrieval (blends BM25 and semantic scores).
        rerank_k: The number of final documents to return.
    """
    _add_src_to_path()

    # Import necessary components after setting up the path
    from ir_core.config import settings
    from ir_core.generation import get_generator
    from ir_core.orchestration.pipeline import RAGPipeline
    from ir_core.utils import read_jsonl
    from ir_core.evaluation.core import mean_average_precision

    print("--- Starting Validation Run ---")
    print(f"Using validation file: {validation_path}")

    # Override settings if CLI arguments are provided
    if alpha is not None:
        settings.ALPHA = alpha
        print(f"Overriding alpha to: {settings.ALPHA}")
    if rerank_k is not None:
        settings.RERANK_K = rerank_k
        print(f"Overriding rerank_k to: {settings.RERANK_K}")

    # Initialize the RAG pipeline. A generator is needed for the tool-calling decision.
    generator = get_generator()
    pipeline = RAGPipeline(generator)

    # Read the validation data
    try:
        validation_data = list(read_jsonl(validation_path))
    except FileNotFoundError:
        print(f"Error: Validation file not found at '{validation_path}'")
        return

    # Store results for final MAP calculation
    all_results = []

    for item in tqdm(validation_data, desc="Validating Queries"):
        query = item.get("msg", [{}])[0].get("content")
        ground_truth_id = item.get("ground_truth_doc_id")

        if not query or not ground_truth_id:
            continue

        # --- FIX: Correctly unpack the structured output from the pipeline ---
        retrieval_output = pipeline.run_retrieval_only(query)

        predicted_docs = []
        if retrieval_output:
            # The output is a list with one entry: [{"standalone_query": ..., "docs": [...]}]
            predicted_docs = retrieval_output[0].get("docs", [])

        predicted_ids = [doc["id"] for doc in predicted_docs]

        # The ground truth is a single relevant document for this set
        relevant_ids = [ground_truth_id]

        all_results.append((predicted_ids, relevant_ids))

    # Calculate the final MAP score
    map_score = mean_average_precision(all_results)

    print("\n--- Validation Complete ---")
    print(f"Total Queries Validated: {len(all_results)}")
    print(f"MAP Score: {map_score:.4f}")
    print("---------------------------")

if __name__ == "__main__":
    fire.Fire(run)

