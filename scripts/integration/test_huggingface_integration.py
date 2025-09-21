#!/usr/bin/env python3
"""
Test script for HuggingFace integration with KLUE-RoBERTa models.
Tests both retrieval (embeddings) and generation components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from ir_core.embeddings.core import encode_query, encode_texts
from ir_core.generation import get_generator
from omegaconf import OmegaConf

def test_gpu_resources():
    """Test GPU availability and resources."""
    print("=== GPU Resources Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"GPU compute capability: {props.major}.{props.minor}")
    else:
        print("No GPU available - models will run on CPU (slower)")
    print()

def test_embeddings():
    """Test embedding generation with KLUE-RoBERTa."""
    print("=== Embeddings Test ===")
    try:
        # Test query encoding
        query = "What is the role of proteins in biological systems?"
        print(f"Encoding query: {query}")
        query_emb = encode_query(query)
        print(f"Query embedding shape: {query_emb.shape}")
        print(f"Query embedding norm: {torch.norm(torch.tensor(query_emb)):.3f}")

        # Test batch encoding
        texts = [
            "Proteins are essential molecules in biological systems.",
            "They perform various functions including catalysis and structure.",
            "Enzymes are a type of protein that speed up chemical reactions."
        ]
        print(f"Encoding {len(texts)} texts...")
        text_embs = encode_texts(texts)
        print(f"Text embeddings shape: {text_embs.shape}")

        print("✓ Embeddings test passed")
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        return False
    print()
    return True

def test_generation():
    """Test text generation with KLUE-RoBERTa."""
    print("=== Generation Test ===")
    try:
        # Create mock config for testing
        config_dict = {
            'pipeline': {
                'generator_type': 'huggingface',
                'generator_model_name': 'microsoft/DialoGPT-medium',  # Use a model that can actually generate text
                'huggingface': {
                    'max_tokens': 50,
                    'temperature': 0.1
                }
            },
            'prompts': {
                'generation_qa': 'prompts/scientific_qa/scientific_qa_v1.jinja2'
            }
        }
        cfg = OmegaConf.create(config_dict)

        print("Initializing HuggingFace generator...")
        generator = get_generator(cfg)

        query = "What are proteins?"
        context_docs = [
            "Proteins are large biomolecules consisting of amino acids.",
            "They perform crucial roles in organisms and viruses."
        ]

        print(f"Generating answer for query: {query}")
        print(f"Context documents: {len(context_docs)}")

        response = generator.generate(query, context_docs)
        print(f"Generated response: {response}")

        print("✓ Generation test passed")
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    return True

def main():
    """Run all tests."""
    print("Testing HuggingFace KLUE-RoBERTa Integration")
    print("=" * 50)

    test_gpu_resources()

    embeddings_ok = test_embeddings()
    generation_ok = test_generation()

    print("=" * 50)
    if embeddings_ok and generation_ok:
        print("✓ All tests passed! HuggingFace integration is working.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())