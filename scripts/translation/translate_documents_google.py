#!/usr/bin/env python3
"""
Google Translate backup translation script.
Uses googletrans library as a fallback when Ollama is unavailable.
"""

import json
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm
import time
import os

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    Translator = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy logs from googletrans and httpx
logging.getLogger('googletrans').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

class GoogleTranslator:
    def __init__(self, source_lang: str = "ko", target_lang: str = "en"):
        if not GOOGLETRANS_AVAILABLE or Translator is None:
            raise ImportError("googletrans library not available. Install with: poetry add googletrans")

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator = Translator()

    async def translate_text(self, text: str) -> str:
        """Translate a single text using Google Translate."""
        if not text or not text.strip():
            return text

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Google Translate has rate limits, so add delay
                await asyncio.sleep(0.1)

                result = await self.translator.translate(
                    text,
                    src=self.source_lang,
                    dest=self.target_lang
                )
                return result.text

            except Exception as e:
                logger.warning(f"Google Translate attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
                continue

        logger.error(f"Google Translate failed after {max_retries} attempts")
        return text  # Return original text if translation fails

    async def translate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate a batch of documents."""
        results = []
        for doc in batch:
            # Handle different document formats
            if "content" in doc and doc["content"].strip():
                # Document format: translate content field
                content = doc.get("content", "")
                translated = await self.translate_text(content)
                new_doc = doc.copy()
                new_doc["content"] = translated
                new_doc["original_content"] = doc.get("content", "")
                new_doc["translation_model"] = "google_translate"
                new_doc["translation_status"] = "success" if translated != doc.get("content", "") else "failed"
                results.append(new_doc)

            elif "msg" in doc:
                # Eval format: translate content in each message
                new_doc = doc.copy()
                new_doc["msg"] = []
                success_count = 0

                for msg in doc["msg"]:
                    if "content" in msg and msg["content"].strip():
                        translated = await self.translate_text(msg["content"])
                        new_msg = msg.copy()
                        new_msg["content"] = translated
                        new_msg["original_content"] = msg.get("content", "")
                        new_doc["msg"].append(new_msg)
                        success_count += 1
                    else:
                        new_doc["msg"].append(msg)

                new_doc["translation_model"] = "google_translate"
                new_doc["translation_status"] = "success" if success_count > 0 else "failed"
                results.append(new_doc)
            else:
                # No translatable content
                new_doc = doc.copy()
                new_doc["translation_status"] = "no_content"
                results.append(new_doc)

        return results

    async def translate_file(self, input_file: str, output_file: str, batch_size: int = 10, max_docs: Optional[int] = None):
        """Translate documents from input file to output file."""
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_file} not found")

        # Count total documents
        total_docs = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
        if max_docs:
            total_docs = min(total_docs, max_docs)

        logger.info(f"Found {total_docs} documents to translate using Google Translate")

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            batch = []
            processed = 0

            with tqdm(total=total_docs, desc="Translating with Google") as pbar:
                for line_num, line in enumerate(f_in):
                    if max_docs and processed >= max_docs:
                        break

                    try:
                        doc = json.loads(line.strip())
                        batch.append(doc)

                        # Check if we should process this batch
                        if len(batch) >= batch_size or (max_docs and processed + len(batch) >= max_docs):
                            # Calculate how many docs we can actually process
                            remaining_slots = max_docs - processed if max_docs else len(batch)
                            docs_to_process = min(len(batch), remaining_slots)

                            translated_batch = await self.translate_batch(batch[:docs_to_process])
                            for translated_doc in translated_batch:
                                f_out.write(json.dumps(translated_doc, ensure_ascii=False) + '\n')
                            processed += len(translated_batch)
                            pbar.update(len(translated_batch))

                            # Remove processed docs from batch
                            batch = batch[docs_to_process:]

                            if max_docs and processed >= max_docs:
                                break

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse line {line_num}: {e}")
                        continue

                # Process remaining batch
                if batch and (not max_docs or processed < max_docs):
                    remaining_slots = max_docs - processed if max_docs else len(batch)
                    docs_to_process = min(len(batch), remaining_slots)

                    translated_batch = await self.translate_batch(batch[:docs_to_process])
                    for translated_doc in translated_batch:
                        f_out.write(json.dumps(translated_doc, ensure_ascii=False) + '\n')
                    processed += len(translated_batch)
                    pbar.update(len(translated_batch))

        logger.info(f"Google Translate completed. Processed {processed} documents.")

async def main():
    if not GOOGLETRANS_AVAILABLE:
        logger.error("googletrans library not installed. Install with: poetry add googletrans")
        return

    parser = argparse.ArgumentParser(description="Translate documents using Google Translate (backup)")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--source-lang", "-s", default="ko", help="Source language code")
    parser.add_argument("--target-lang", "-t", default="en", help="Target language code")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--max-docs", type=int, help="Maximum number of documents to process")

    args = parser.parse_args()

    translator = GoogleTranslator(args.source_lang, args.target_lang)
    await translator.translate_file(args.input, args.output, args.batch_size, args.max_docs)

if __name__ == "__main__":
    asyncio.run(main())