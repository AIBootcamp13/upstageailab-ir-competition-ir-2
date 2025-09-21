#!/usr/bin/env python3
"""
High-quality document translation using Ollama local models.
Supports batch processing with parallel requests for better throughput.
"""

import json
import argparse
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaTranslator:
    def __init__(self, model_name: str = "qwen2:7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def translate_text(self, text: str, source_lang: str = "Korean", target_lang: str = "English") -> str:
        """Translate a single text using Ollama."""
        if not text or not text.strip():
            return text

        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        prompt = f"""Translate the following {source_lang} text to {target_lang}. 
Only provide the translated text, no explanations or additional content:

{text}"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent translation
                "top_p": 0.9,
                "num_predict": 1024
            }
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        translated = result.get("response", "").strip()
                        return translated
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed with status {response.status}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                        continue
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                continue

        logger.error(f"Failed to translate text after {max_retries} attempts")
        return text  # Return original text if translation fails

    async def translate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate a batch of documents in parallel."""
        # Collect all translation tasks
        all_tasks = []
        task_metadata = []  # Store metadata for each task
        
        for doc in batch:
            # Handle different document formats
            if "content" in doc and doc["content"].strip():
                # Document format: translate content field
                content = doc.get("content", "")
                task = self.translate_text(content)
                all_tasks.append(task)
                task_metadata.append(("content", doc))
            elif "msg" in doc:
                # Eval format: translate content in each message
                msg_tasks = []
                msg_metadata = []
                for i, msg in enumerate(doc["msg"]):
                    if "content" in msg and msg["content"].strip():
                        task = self.translate_text(msg["content"])
                        msg_tasks.append(task)
                        msg_metadata.append((i, msg))
                    else:
                        msg_tasks.append(None)
                        msg_metadata.append((i, msg))
                all_tasks.append(msg_tasks)  # Store list of tasks for this doc
                task_metadata.append(("msg", doc, msg_metadata))
            else:
                # No translatable content
                all_tasks.append(None)
                task_metadata.append((None, doc))

        # Execute all translation tasks in parallel
        if all_tasks:
            # Flatten tasks for parallel execution, but keep track of structure
            flat_tasks = []
            task_indices = []
            
            for i, task_info in enumerate(all_tasks):
                if isinstance(task_info, list):  # msg format with multiple tasks
                    for j, task in enumerate(task_info):
                        if task is not None:
                            flat_tasks.append(task)
                            task_indices.append((i, j))  # (doc_index, msg_index)
                elif task_info is not None:  # single content task
                    flat_tasks.append(task_info)
                    task_indices.append((i, None))  # (doc_index, None)
            
            # Execute all tasks in parallel
            if flat_tasks:
                try:
                    translated_results = await asyncio.gather(*flat_tasks, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Batch translation failed: {e}")
                    # Fall back to sequential processing on error
                    translated_results = []
                    for task in flat_tasks:
                        try:
                            result = await task
                            translated_results.append(result)
                        except Exception as task_e:
                            logger.error(f"Individual task failed: {task_e}")
                            translated_results.append("")  # Empty string on failure
            else:
                translated_results = []
            
            # Reconstruct results
            result_index = 0
            results = []
            
            for i, (field_type, doc, *extra) in enumerate(task_metadata):
                if field_type == "content":
                    # Single content translation
                    if (i, None) in task_indices:
                        idx = task_indices.index((i, None))
                        if idx < len(translated_results):
                            result = translated_results[idx]
                            if isinstance(result, Exception):
                                logger.error(f"Translation failed for doc {doc.get('docid')}: {result}")
                                translated = doc.get("content", "")
                            else:
                                translated = result
                        else:
                            translated = doc.get("content", "")
                        
                        new_doc = doc.copy()
                        new_doc["content"] = translated
                        new_doc["original_content"] = doc.get("content", "")
                        new_doc["translation_model"] = self.model_name
                        new_doc["translation_status"] = "success" if translated != doc.get("content", "") else "failed"
                        results.append(new_doc)
                    
                elif field_type == "msg":
                    # Multiple message translations
                    msg_metadata = extra[0] if extra else []
                    new_doc = doc.copy()
                    new_doc["msg"] = []
                    success_count = 0
                    
                    for msg_idx, (orig_idx, msg) in enumerate(msg_metadata):
                        if (i, msg_idx) in task_indices:
                            task_idx = task_indices.index((i, msg_idx))
                            if task_idx < len(translated_results):
                                result = translated_results[task_idx]
                                if isinstance(result, Exception):
                                    logger.error(f"Translation failed for msg {orig_idx} in doc {doc.get('eval_id')}: {result}")
                                    translated = msg.get("content", "")
                                else:
                                    translated = result
                            else:
                                translated = msg.get("content", "")
                            
                            new_msg = msg.copy()
                            new_msg["content"] = translated
                            new_msg["original_content"] = msg.get("content", "")
                            new_doc["msg"].append(new_msg)
                            if translated != msg.get("content", ""):
                                success_count += 1
                        else:
                            # No translation needed
                            new_doc["msg"].append(msg)
                    
                    new_doc["translation_model"] = self.model_name
                    new_doc["translation_status"] = "success" if success_count > 0 else "failed"
                    results.append(new_doc)
                    
                else:
                    # No translatable content
                    new_doc = doc.copy()
                    new_doc["translation_status"] = "no_content"
                    results.append(new_doc)
            
            return results
        else:
            return []

    async def translate_file(self, input_file: str, output_file: str, batch_size: int = 10, max_docs: Optional[int] = None, resume: bool = False):
        """Translate documents from input file to output file."""
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_file} not found")

        # Count total documents
        total_docs = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
        if max_docs:
            total_docs = min(total_docs, max_docs)

        # Check if resuming and count already processed
        processed_ids = set()
        if resume and output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        if "docid" in doc:
                            processed_ids.add(doc["docid"])
                        elif "eval_id" in doc:
                            processed_ids.add(doc["eval_id"])
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Resuming translation, already processed {len(processed_ids)} documents")

        logger.info(f"Found {total_docs} documents to translate")

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'a' if resume else 'w', encoding='utf-8') as f_out:

            batch = []
            processed = len(processed_ids) if resume else 0

            with tqdm(total=total_docs, desc="Translating documents") as pbar:
                pbar.update(processed)  # Update progress bar for resumed docs
                
                for line_num, line in enumerate(f_in):
                    if max_docs and processed >= max_docs:
                        break

                    try:
                        doc = json.loads(line.strip())
                        doc_id = doc.get("docid") or doc.get("eval_id")
                        
                        # Skip if already processed (when resuming)
                        if resume and doc_id in processed_ids:
                            continue
                            
                        batch.append(doc)

                        if len(batch) >= batch_size:
                            translated_batch = await self.translate_batch(batch)
                            for translated_doc in translated_batch:
                                f_out.write(json.dumps(translated_doc, ensure_ascii=False) + '\n')
                            processed += len(batch)
                            pbar.update(len(batch))
                            batch = []

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse line {line_num}: {e}")
                        continue

                # Process remaining batch
                if batch:
                    translated_batch = await self.translate_batch(batch)
                    for translated_doc in translated_batch:
                        f_out.write(json.dumps(translated_doc, ensure_ascii=False) + '\n')
                    processed += len(batch)
                    pbar.update(len(batch))

        logger.info(f"Translation completed. Processed {processed} documents.")

async def main():
    parser = argparse.ArgumentParser(description="Translate documents using Ollama")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--model", "-m", default="qwen2:1.5b", help="Ollama model name")
    parser.add_argument("--batch-size", "-b", type=int, default=20, help="Batch size for parallel processing")
    parser.add_argument("--max-docs", type=int, help="Maximum number of documents to process")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--resume", action="store_true", help="Resume translation from existing output file")

    args = parser.parse_args()

    async with OllamaTranslator(args.model, args.base_url) as translator:
        await translator.translate_file(args.input, args.output, args.batch_size, args.max_docs, args.resume)

if __name__ == "__main__":
    asyncio.run(main())