# scripts/analyze_data.py
import os
import sys
import numpy as np
import fire


# Add the src directory to the path to allow for project imports
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def analyze(file_path: str = "data/documents.jsonl"):
    """
    Analyzes a JSONL document file to provide statistics on token counts.

    This script helps understand the distribution of document lengths, which is
    crucial for setting tokenizer parameters like `max_length`.

    Args:
        file_path: Path to the .jsonl file containing the documents.
    """
    _add_src_to_path()
    from ir_core.utils.core import read_jsonl
    from ir_core.embeddings.core import load_model

    print("--- Starting Dataset Analysis ---")

    try:
        # Load the tokenizer used in the project to get accurate token counts
        print("Loading tokenizer...")
        tokenizer, _ = load_model()
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(
            f"Failed to load tokenizer. Make sure model weights are available. Error: {e}"
        )
        return

    docs = list(read_jsonl(file_path))
    if not docs:
        print("No documents found in the file.")
        return

    print(f"Analyzing {len(docs)} documents from '{file_path}'...")

    token_counts = []
    for doc in docs:
        content = doc.get("content", "")
        if content:
            # Tokenize but don't create tensors; we just need the IDs to count them.
            tokens = tokenizer.encode(content)
            token_counts.append(len(tokens))

    if not token_counts:
        print("Documents found, but they appear to have no content to analyze.")
        return

    # --- Calculate and Print Statistics ---
    counts_np = np.array(token_counts)
    print("\n--- Token Count Statistics ---")
    print(f"Total documents analyzed: {len(counts_np)}")
    print(f"Min token length:         {np.min(counts_np)}")
    print(f"Max token length:         {np.max(counts_np)}")
    print(f"Mean token length:        {np.mean(counts_np):.2f}")
    print(f"Median token length:      {np.median(counts_np)}")
    print(f"Standard Deviation:       {np.std(counts_np):.2f}")

    # --- Print Percentiles ---
    print("\n--- Percentile Distribution ---")
    print(f"90th percentile:          {np.percentile(counts_np, 90):.0f} tokens")
    print(f"95th percentile:          {np.percentile(counts_np, 95):.0f} tokens")
    print(f"99th percentile:          {np.percentile(counts_np, 99):.0f} tokens")

    # Check how many documents are longer than the model's max length
    max_len = 512
    oversized_docs = np.sum(counts_np > max_len)
    print(
        f"\nDocuments longer than {max_len} tokens: {oversized_docs} ({oversized_docs / len(counts_np) * 100:.2f}%)"
    )

    # --- Print a Simple Text-Based Histogram ---
    print("\n--- Histogram of Token Lengths ---")
    hist, bin_edges = np.histogram(counts_np, bins=10)
    max_freq = np.max(hist)

    for i in range(len(hist)):
        bin_start = int(bin_edges[i])
        bin_end = int(bin_edges[i + 1])
        # Scale the bar length for display
        bar_length = int(50 * hist[i] / max_freq) if max_freq > 0 else 0
        bar = "█" * bar_length
        print(f"{bin_start:4d} - {bin_end:4d} | {bar} ({hist[i]})")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    fire.Fire(analyze)
