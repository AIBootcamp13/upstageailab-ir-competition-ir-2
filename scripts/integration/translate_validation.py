#!/usr/bin/env python3
"""
Translate Korean Validation Queries to English

This script translates the Korean validation queries to English to match
the English-indexed documents, fixing the language mismatch issue.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import openai
from ir_core.config import settings


def translate_text(text: str, client: openai.OpenAI) -> str:
    """Translate Korean text to English using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the following Korean text to English. Keep scientific terms accurate and maintain the original meaning. Only return the English translation, nothing else."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original if translation fails


def translate_validation_queries(input_file: str, output_file: str):
    """Translate all validation queries from Korean to English."""
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return False

    client = openai.OpenAI()

    print(f"Loading validation data from: {input_file}")

    # Read validation data
    with open(input_file, 'r', encoding='utf-8') as f:
        validation_data = [json.loads(line) for line in f if line.strip()]

    print(f"Found {len(validation_data)} validation queries")

    translated_data = []

    for i, item in enumerate(validation_data):
        print(f"Translating query {i+1}/{len(validation_data)}...")

        # Extract the query from the message
        messages = item.get("msg", [])
        if not messages:
            print(f"  Skipping item {i+1}: no messages")
            continue

        korean_query = messages[0].get("content", "")
        if not korean_query:
            print(f"  Skipping item {i+1}: empty query")
            continue

        # Translate the query
        english_query = translate_text(korean_query, client)

        # Create translated item
        translated_item = item.copy()
        translated_item["original_query"] = korean_query
        translated_item["translated_query"] = english_query
        translated_item["msg"] = [{"content": english_query, "role": "user"}]

        translated_data.append(translated_item)

        # Progress update
        if (i + 1) % 5 == 0:
            print(f"  Translated {i+1}/{len(validation_data)} queries")

    # Save translated data
    print(f"Saving translated data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in translated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Successfully translated {len(translated_data)} queries")
    return True


def validate_translation(input_file: str, translated_file: str):
    """Validate that the translation worked by testing a few queries."""
    print("\n=== Validating Translation ===")

    from ir_core.retrieval.core import hybrid_retrieve

    # Read a few samples from translated data
    with open(translated_file, 'r', encoding='utf-8') as f:
        translated_samples = [json.loads(line) for line in f][:3]

    for i, item in enumerate(translated_samples):
        query = item.get("translated_query", "")
        print(f"\nTest {i+1}:")
        print(f"  Original: {item.get('original_query', '')[:50]}...")
        print(f"  Translated: {query[:50]}...")

        # Test retrieval
        results = hybrid_retrieve(query, rerank_k=2)
        print(f"  Retrieved: {len(results)} documents")

        if results:
            top_score = results[0].get('score', 0)
            print(".3f")


def main():
    """Main function to translate validation queries."""
    input_file = "data/validation_balanced.jsonl"
    output_file = "data/validation_balanced_en.jsonl"

    if not Path(input_file).exists():
        print(f"ERROR: Input file {input_file} does not exist")
        return 1

    print("Starting Korean to English translation of validation queries")
    print("=" * 60)

    success = translate_validation_queries(input_file, output_file)

    if success:
        validate_translation(input_file, output_file)
        print("\n" + "=" * 60)
        print("Translation completed successfully!")
        print(f"Translated file: {output_file}")
        print("\nTo use the translated validation data, update your configuration to point to:")
        print(f"  validation_path: {output_file}")
        return 0
    else:
        print("Translation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())