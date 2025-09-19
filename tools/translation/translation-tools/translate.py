#!/usr/bin/env python3
"""
Translation utility using uv-managed googletrans.
This script provides translation functionality without dependency conflicts.
"""

import sys
import json
import asyncio
from pathlib import Path
from googletrans import Translator

async def translate_text(text: str, src: str = 'auto', dest: str = 'en') -> str:
    """Translate text using googletrans."""
    translator = Translator()
    try:
        result = await translator.translate(text, src=src, dest=dest)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}", file=sys.stderr)
        return text  # Return original on error

async def translate_jsonl(input_file: str, output_file: str, text_field: str = 'content'):
    """Translate a JSONL file."""
    print(f"Translating {input_file} -> {output_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    translated_lines = []

    for i, line in enumerate(lines):
        data = json.loads(line.strip())

        # Handle nested msg structure for validation data
        if 'msg' in data and isinstance(data['msg'], list) and len(data['msg']) > 0:
            # Find the user message
            user_msg = None
            for msg in data['msg']:
                if msg.get('role') == 'user':
                    user_msg = msg
                    break

            if user_msg and 'content' in user_msg:
                original_text = user_msg['content']
                translated_text = await translate_text(original_text)

                # Update the message content
                user_msg['content'] = translated_text
                user_msg['content_original'] = original_text
                data['translation_status'] = 'success'
            else:
                data['translation_status'] = 'skipped'
        # Handle direct text field
        elif text_field in data and data[text_field]:
            original_text = data[text_field]
            translated_text = await translate_text(original_text)

            # Add translation metadata
            data[f'{text_field}_original'] = original_text
            data[text_field] = translated_text
            data['translation_status'] = 'success'
        else:
            data['translation_status'] = 'skipped'

        translated_lines.append(json.dumps(data, ensure_ascii=False))

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(lines)} items")

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + '\n')

    print(f"Translation complete! Output: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python translate.py <input_file> <output_file> [text_field]")
        print("Example: python translate.py queries.jsonl queries_en.jsonl content")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    text_field = sys.argv[3] if len(sys.argv) > 3 else 'content'

    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)

    asyncio.run(translate_jsonl(input_file, output_file, text_field))

if __name__ == "__main__":
    main()