#!/usr/bin/env python3
"""
Test script for document translation

Tests the translation functionality with a small sample before running full translation.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from scripts.translation.translate_documents import LocalDocumentTranslator, TranslationConfig


def test_translation():
    """Test translation with a few sample documents."""

    # Sample Korean scientific documents
    test_docs = [
        {
            "docid": "test_001",
            "content": "건강한 사람이 에너지 균형을 평형 상태로 유지하는 것은 중요합니다. 에너지 균형은 에너지 섭취와 에너지 소비의 수학적 동등성을 의미합니다.",
            "src": "test"
        },
        {
            "docid": "test_002",
            "content": "수소, 산소, 질소 가스의 혼합물에서 평균 속도가 가장 빠른 분자는 수소입니다. 수소 분자는 가장 가볍고 작은 원자로 구성되어 있기 때문에 다른 분자들보다 더 빠르게 움직입니다.",
            "src": "test"
        },
        {
            "docid": "test_003",
            "content": "마이애미파랑나비는 남부 플로리다에서 멸종 위기에 처한 종입니다. 이 나비의 개체수 감소를 초래했을 가능성이 가장 높은 요인은 주택 건설 증가입니다.",
            "src": "test"
        }
    ]

    print("Testing document translation...")
    print("=" * 50)

    # Create translator with conservative settings for testing
    config = TranslationConfig(
        model_name="Helsinki-NLP/opus-mt-ko-en",
        batch_size=2,  # Small batch for testing
        max_length=256  # Shorter for testing
    )

    try:
        translator = LocalDocumentTranslator(config)
        print("✓ Translation model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load translation model: {e}")
        return False

    # Test translation
    translated_docs = []
    for i, doc in enumerate(test_docs, 1):
        print(f"\nTranslating document {i}/{len(test_docs)}...")
        print(f"Original: {doc['content'][:100]}...")

        translated_doc = translator.translate_document(doc)
        translated_docs.append(translated_doc)

        print(f"Translated: {translated_doc['content'][:100]}...")
        print(f"Status: {translated_doc.get('translation_status', 'unknown')}")

        if translated_doc.get('translation_status') != 'success':
            print(f"✗ Translation failed: {translated_doc.get('translation_error', 'Unknown error')}")
            return False

    print("\n" + "=" * 50)
    print("✓ All translations completed successfully!")
    print("\nDetailed Results:")
    print("=" * 50)

    for i, (original, translated) in enumerate(zip(test_docs, translated_docs), 1):
        print(f"\nDocument {i}:")
        print(f"Original:  {original['content']}")
        print(f"Translated: {translated['content']}")
        print(f"Status: {translated.get('translation_status')}")

    # Save test results
    output_file = Path(__file__).parent / "translation_test_results.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in translated_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"\n✓ Test results saved to: {output_file}")
    return True


if __name__ == "__main__":
    success = test_translation()
    if success:
        print("\n🎉 Translation test passed! Ready to run full translation.")
        sys.exit(0)
    else:
        print("\n❌ Translation test failed. Please check the errors above.")
        sys.exit(1)