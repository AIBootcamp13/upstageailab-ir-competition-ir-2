#!/usr/bin/env python3
"""
Translation Integration Script for RAG Validation Pipeline

This script integrates the uv-powered translation system into the validation pipeline.
It provides caching, batch processing, and seamless integration with existing workflows.

Usage:
    python scripts/translation/integrate_translation.py --input data/validation_balanced.jsonl --output data/validation_balanced_en.jsonl --cache
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import redis
from tqdm import tqdm

# Add src to path
scripts_dir = Path(__file__).parent
repo_dir = scripts_dir.parent.parent
src_dir = repo_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from ir_core.config import settings
    from ir_core.utils.core import read_jsonl
except ImportError:
    # Fallback if imports fail
    class FallbackSettings:
        REDIS_URL = "redis://localhost:6379/0"
    settings = FallbackSettings()

    def read_jsonl(path):
        """Fallback JSONL reader."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranslationCache:
    """Redis-based caching for translation results."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self.redis_client = None
        self._connect()

    def _connect(self):
        """Connect to Redis with error handling."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            self.redis_client.ping()
            logger.info("Connected to Redis for translation caching")
        except redis.ConnectionError:
            logger.warning("Redis not available, translation caching disabled")
            self.redis_client = None

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text content."""
        return f"translation:ko_en:{hashlib.md5(text.encode('utf-8')).hexdigest()}"

    def get_cached_translation(self, text: str) -> Optional[str]:
        """Get cached translation if available."""
        if not self.redis_client:
            return None

        cache_key = self._get_cache_key(text)
        try:
            cached = self.redis_client.get(cache_key)
            if cached and isinstance(cached, bytes):
                return cached.decode('utf-8')
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")

        return None

    def cache_translation(self, original_text: str, translated_text: str):
        """Cache translation result."""
        if not self.redis_client:
            return

        cache_key = self._get_cache_key(original_text)
        try:
            # Cache for 30 days (30 * 24 * 60 * 60 seconds)
            self.redis_client.setex(cache_key, 2592000, translated_text.encode('utf-8'))
        except Exception as e:
            logger.warning(f"Error caching translation: {e}")

def translate_with_uv(input_file: str, output_file: str, use_cache: bool = True) -> bool:
    """
    Translate JSONL file using the uv-powered translation system.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        use_cache: Whether to use Redis caching

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get absolute paths
        repo_dir = Path(__file__).parent.parent.parent
        input_path = repo_dir / input_file
        output_path = repo_dir / output_file

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return False

        # Run the uv translation script
        translate_script = repo_dir / "tools" / "translation" / "translate.sh"

        if not translate_script.exists():
            logger.error(f"Translation script not found: {translate_script}")
            return False

        cmd = [str(translate_script), str(input_path), str(output_path)]
        logger.info(f"Running translation: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Translation completed successfully")
            return True
        else:
            logger.error(f"Translation failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error during translation: {e}")
        return False

def translate_with_caching(input_file: str, output_file: str, use_cache: bool = True) -> bool:
    """
    Translate JSONL file with Redis caching support.

    This function reads the input file, checks cache for each translation,
    translates uncached items, and writes the results.
    """
    cache = TranslationCache() if use_cache else None

    try:
        # Read input data
        input_data = list(read_jsonl(input_file))
        logger.info(f"Loaded {len(input_data)} entries from {input_file}")

        translated_data = []
        cache_hits = 0
        cache_misses = 0

        for entry in tqdm(input_data, desc="Processing translations"):
            # Handle nested msg structure
            if 'msg' in entry and isinstance(entry['msg'], list):
                for msg in entry['msg']:
                    if msg.get('role') == 'user' and 'content' in msg:
                        original_text = msg['content']

                        # Check cache first
                        cached_translation = None
                        if cache:
                            cached_translation = cache.get_cached_translation(original_text)

                        if cached_translation:
                            msg['content'] = cached_translation
                            msg['content_original'] = original_text
                            entry['translation_status'] = 'cached'
                            cache_hits += 1
                        else:
                            # Need to translate - for now, we'll use the uv script
                            # In a future enhancement, we could integrate direct translation here
                            msg['translation_status'] = 'needs_translation'
                            cache_misses += 1

            translated_data.append(entry)

        # If we have uncached translations, run the uv translation
        needs_translation = any(
            msg.get('translation_status') == 'needs_translation'
            for entry in translated_data
            for msg in entry.get('msg', [])
        )

        if needs_translation:
            logger.info(f"Found {cache_misses} uncached translations, running uv translation...")

            # Write intermediate file for uv translation
            temp_input = f"{input_file}.temp"
            with open(temp_input, 'w', encoding='utf-8') as f:
                for entry in translated_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            # Run uv translation
            if translate_with_uv(temp_input, output_file, use_cache=False):
                # Read back the translated results and update cache
                translated_results = list(read_jsonl(output_file))

                for result in translated_results:
                    if 'msg' in result and isinstance(result['msg'], list):
                        for msg in result['msg']:
                            if (msg.get('role') == 'user' and
                                'content' in msg and
                                'content_original' in msg):
                                # Cache the translation
                                if cache:
                                    cache.cache_translation(
                                        msg['content_original'],
                                        msg['content']
                                    )

                # Clean up temp file
                Path(temp_input).unlink(missing_ok=True)
            else:
                logger.error("UV translation failed")
                return False
        else:
            # All translations were cached, just write the results
            logger.info(f"All {cache_hits} translations were cached")
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in translated_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"Translation complete. Cache hits: {cache_hits}, Cache misses: {cache_misses}")
        return True

    except Exception as e:
        logger.error(f"Error in cached translation: {e}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Translate validation data with caching")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--cache", action="store_true", help="Enable Redis caching")
    parser.add_argument("--force", action="store_true", help="Force translation even if output exists")

    args = parser.parse_args()

    # Check if output already exists
    if Path(args.output).exists() and not args.force:
        logger.info(f"Output file {args.output} already exists. Use --force to overwrite.")
        return

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = translate_with_caching(args.input, args.output, args.cache)

    if success:
        logger.info(f"Translation completed successfully: {args.output}")
        sys.exit(0)
    else:
        logger.error("Translation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()