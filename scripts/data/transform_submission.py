#!/usr/bin/env python3
"""
transform_submission.py

This script converts submission data to structured JSON format with evaluation metrics.
It transforms submission files into a detailed JSON structure for evaluation logging.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_eval_queries(eval_file: str) -> Dict[str, str]:
    """
    Load evaluation queries from a JSONL file.

    Args:
        eval_file (str): Path to the evaluation JSONL file

    Returns:
        Dict[str, str]: Dictionary mapping eval_id to original query
    """
    eval_queries = {}

    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())
                eval_id = entry.get("eval_id")
                if eval_id:
                    # Extract the last message content (usually the user's query)
                    messages = entry.get("msg", [])
                    if messages and isinstance(messages, list):
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict) and "content" in last_msg:
                            eval_queries[eval_id] = last_msg["content"]

    return eval_queries


def process_submission(eval_file: str, submission_file: str, output_file: str):
    """
    Process submission and evaluation files to create structured evaluation logs.

    Args:
        eval_file (str): Path to the evaluation JSONL file
        submission_file (str): Path to the submission CSV file
        output_file (str): Path to the output JSONL file
    """
    print(f"Loading evaluation queries from: {eval_file}")
    eval_queries = load_eval_queries(eval_file)
    print(f"Loaded {len(eval_queries)} evaluation queries")

    print(f"Processing submission file: {submission_file}")

    evaluation_logs = []

    with open(submission_file, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            eval_id = row.get("eval_id")
            if not eval_id:
                continue

            # Get original query from eval file
            original_query = eval_queries.get(eval_id, row.get("standalone_query", ""))

            # Extract retrieved documents (assuming they're in JSON format in the CSV)
            retrieved_docs = []
            if "topk" in row and row["topk"]:
                try:
                    # Parse topk IDs and try to get content if available
                    topk_ids = (
                        json.loads(row["topk"])
                        if row["topk"].startswith("[")
                        else [row["topk"]]
                    )
                    retrieved_docs = [{"id": doc_id} for doc_id in topk_ids]
                except (json.JSONDecodeError, TypeError):
                    retrieved_docs = [{"id": row["topk"]}]

            # Create structured evaluation log entry
            log_entry = {
                "eval_id": eval_id,
                "original_query": original_query,
                "standalone_query": row.get("standalone_query", original_query),
                "retrieved_docs": retrieved_docs,
                "answer": row.get("answer", ""),
                "references": row.get("references", []),
                "metadata": {
                    "submission_source": "transformed_from_csv",
                    "processing_timestamp": None,  # Could add timestamp here
                },
            }

            evaluation_logs.append(log_entry)

    # Write to output file
    print(f"Writing {len(evaluation_logs)} evaluation logs to: {output_file}")

    with open(output_file, "w", encoding="utf-8") as outfile:
        for log_entry in evaluation_logs:
            outfile.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"Successfully created evaluation logs file: {output_file}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 4:
        print(
            "Usage: python transform_submission.py <eval_file> <submission_file> <output_file>"
        )
        print(
            "Example: python transform_submission.py data/eval.jsonl outputs/submission.csv outputs/evaluation_logs.jsonl"
        )
        sys.exit(1)

    eval_file = sys.argv[1]
    submission_file = sys.argv[2]
    output_file = sys.argv[3]

    # Check if input files exist
    if not Path(eval_file).exists():
        print(f"Error: Evaluation file '{eval_file}' does not exist")
        sys.exit(1)

    if not Path(submission_file).exists():
        print(f"Error: Submission file '{submission_file}' does not exist")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        process_submission(eval_file, submission_file, output_file)
    except Exception as e:
        print(f"Error processing files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
