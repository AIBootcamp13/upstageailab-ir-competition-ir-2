#!/usr/bin/env python3
"""
Translate validation queries from Korean to English.
"""

import json
import asyncio
from pathlib import Path
from googletrans import Translator

async def translate_validation_queries(input_file: str, output_file: str):
    """Translate validation queries to English."""
    translator = Translator()

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    translated_lines = []

    for i, line in enumerate(lines):
        print(f"Translating query {i+1}/{len(lines)}...")
        data = json.loads(line.strip())

        # Translate the query content
        korean_query = data['msg'][0]['content']
        try:
            translation = await translator.translate(korean_query, src='ko', dest='en')
            english_query = translation.text

            # Update the data with English query
            data['msg'][0]['content'] = english_query
            data['original_korean_query'] = korean_query  # Keep original for reference

        except Exception as e:
            print(f"Translation failed for query {i+1}: {e}")
            # Keep original query if translation fails
            data['original_korean_query'] = korean_query

        translated_lines.append(json.dumps(data, ensure_ascii=False))

    # Write translated data
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + '\n')

    print(f"Translated {len(translated_lines)} queries to {output_file}")

if __name__ == "__main__":
    input_file = "outputs/validation_sample.jsonl"
    output_file = "outputs/validation_en.jsonl"

    asyncio.run(translate_validation_queries(input_file, output_file))