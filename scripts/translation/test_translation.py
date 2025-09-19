#!/usr/bin/env python3
"""
Test script for document translation

Tests the translation functionality with a small sample before running full translation.
"""

import json
import sys
import asyncio
from pathlib import Path

#!/usr/bin/env python3
"""
Test script for document translation

Tests the translation functionality with a small sample before running full translation.
"""

import json
import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import translators using absolute paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from translate_documents_ollama import OllamaTranslator
from translate_documents_google import GoogleTranslator


async def test_ollama_translation():
    """Test translation using Ollama (preferred local method)."""
    print("Testing Ollama translation...")

    try:
        async with OllamaTranslator() as translator:
            # Test with a simple Korean text
            test_text = "안녕하세요, 오늘 날씨가 좋네요."
            translated = await translator.translate_text(test_text)

            if translated and translated != test_text:
                print(f"✓ Ollama translation successful:")
                print(f"  Original: {test_text}")
                print(f"  Translated: {translated}")
                return True
            else:
                print("✗ Ollama translation failed - no translation produced")
                return False

    except Exception as e:
        print(f"✗ Ollama translation failed: {e}")
        return False


async def test_google_translation():
    """Test translation using Google Translate (fallback method)."""
    print("Testing Google Translate...")

    try:
        translator = GoogleTranslator()
        test_text = "안녕하세요, 오늘 날씨가 좋네요."
        translated = await translator.translate_text(test_text)

        if translated and translated != test_text:
            print(f"✓ Google translation successful:")
            print(f"  Original: {test_text}")
            print(f"  Translated: {translated}")
            return True
        else:
            print("✗ Google translation failed - no translation produced")
            return False

    except Exception as e:
        print(f"✗ Google translation failed: {e}")
        return False


async def test_batch_translation():
    """Test batch translation with sample data."""
    print("Testing batch translation with sample data...")

    # Load a few sample documents
    data_file = Path(__file__).parent.parent.parent / "data" / "validation_balanced.jsonl"

    if not data_file.exists():
        print(f"✗ Sample data file not found: {data_file}")
        return False

    sample_docs = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Test with first 3 documents
                break
            try:
                doc = json.loads(line.strip())
                sample_docs.append(doc)
            except json.JSONDecodeError:
                continue

    if not sample_docs:
        print("✗ No valid sample documents found")
        return False

    print(f"Loaded {len(sample_docs)} sample documents")

    # Try Ollama first, then Google as fallback
    translator = None
    try:
        translator = OllamaTranslator()
        async with translator:
            # Test batch translation
            results = await translator.translate_batch(sample_docs[:1])  # Test with just one doc

        if results and len(results) > 0:
            original_content = sample_docs[0]['msg'][0]['content'][:100] + "..."
            translated_content = results[0]['msg'][0]['content'][:100] + "..."

            if translated_content != original_content:
                print("✓ Batch translation successful")
                print(f"  Original: {original_content}")
                print(f"  Translated: {translated_content}")
                return True
            else:
                print("✗ Batch translation failed - content unchanged")
                return False

    except Exception as e:
        print(f"✗ Batch translation with Ollama failed: {e}")

        # Try Google as fallback
        try:
            translator = GoogleTranslator()
            results = await translator.translate_batch(sample_docs[:1])

            if results and len(results) > 0:
                original_content = sample_docs[0]['msg'][0]['content'][:100] + "..."
                translated_content = results[0]['msg'][0]['content'][:100] + "..."

                if translated_content != original_content:
                    print("✓ Batch translation successful (Google fallback)")
                    print(f"  Original: {original_content}")
                    print(f"  Translated: {translated_content}")
                    return True
                else:
                    print("✗ Batch translation failed - content unchanged")
                    return False
        except Exception as e2:
            print(f"✗ Batch translation with Google also failed: {e2}")
            return False


async def test_translation():
    """Test translation with a few sample documents."""
    print("🚀 Starting translation functionality test...\n")

    # Test 1: Single text translation with Ollama
    ollama_success = await test_ollama_translation()
    print()

    # Test 2: Single text translation with Google (if Ollama failed)
    google_success = False
    if not ollama_success:
        google_success = await test_google_translation()
        print()

    # Test 3: Batch translation
    batch_success = await test_batch_translation()
    print()

    # Overall result
    if ollama_success or google_success or batch_success:
        print("🎉 Translation test passed! Ready to run full translation.")
        return True
    else:
        print("❌ All translation tests failed. Please check your setup:")
        print("  - For Ollama: Ensure Ollama is running with a translation model")
        print("  - For Google: Ensure googletrans library is installed")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_translation())
    sys.exit(0 if success else 1)