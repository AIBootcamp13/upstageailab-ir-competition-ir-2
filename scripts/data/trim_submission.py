#!/usr/bin/env python3
"""
trim_submission.py

This script trims verbose content in submission files to make them suitable for LLM analysis.
It reads a submission CSV file and creates a trimmed version with shorter content for each entry.
"""

import csv
import json
import sys
from pathlib import Path


def trim_content(content: str, max_length: int = 500) -> str:
    """
    Trim content to a maximum length while preserving word boundaries.

    Args:
        content (str): The original content to trim
        max_length (int): Maximum length for the trimmed content

    Returns:
        str: Trimmed content
    """
    if len(content) <= max_length:
        return content

    # Find the last space within the max_length limit
    trimmed = content[:max_length]
    last_space = trimmed.rfind(" ")

    if last_space > 0:
        return trimmed[:last_space] + "..."
    else:
        return trimmed + "..."


def process_submission(input_file: str, output_file: str, max_length: int = 500):
    """
    Process a submission CSV file and create a trimmed version.

    Args:
        input_file (str): Path to the input submission CSV file
        output_file (str): Path to the output trimmed CSV file
        max_length (int): Maximum content length for trimming
    """
    print(f"Processing submission file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Maximum content length: {max_length}")

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8", newline=""
    ) as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames:
            print("Error: Could not read CSV headers")
            return

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        processed_count = 0
        for row in reader:
            # Trim content in relevant fields (adjust field names as needed)
            if "content" in row:
                row["content"] = trim_content(row["content"], max_length)
            if "answer" in row:
                row["answer"] = trim_content(row["answer"], max_length)

            writer.writerow(row)
            processed_count += 1

    print(f"Successfully processed {processed_count} rows")
    print(f"Trimmed submission saved to: {output_file}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 3:
        print(
            "Usage: python trim_submission.py <input_file> <output_file> [max_length]"
        )
        print(
            "Example: python trim_submission.py outputs/submission.jsonl outputs/submission_trimmed.jsonl 500"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        process_submission(input_file, output_file, max_length)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
