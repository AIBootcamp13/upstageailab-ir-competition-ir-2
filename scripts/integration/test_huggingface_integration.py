#!/usr/bin/env python3
"""
Test script for HuggingFace integration with nlpai-lab/KURE-v1 models.
Tests both retrieval (embeddings) and generation components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse

def test_gpu_resources():
    """Test GPU availability and resources."""
    import torch
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
    """Test embedding generation with nlpai-lab/KURE-v1."""
    import torch
    from ir_core.embeddings.core import encode_query, encode_texts

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

def test_generation(config_path=None, config_name="settings"):
    """Test text generation with HuggingFace models."""
    from omegaconf import OmegaConf
    import hydra
    from hydra import compose, initialize_config_dir
    from ir_core.generation import get_generator

    print("=== Generation Test ===")
    try:
        # Load configuration using Hydra
        config_dir = config_path or os.path.join(os.path.dirname(__file__), '..', '..', 'conf')
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)

        # Override for HuggingFace testing if not already set
        if cfg.pipeline.generator_type != 'huggingface':
            print("Overriding generator to use HuggingFace for testing...")
            cfg.pipeline.generator_type = 'huggingface'
            cfg.pipeline.generator_model_name = 'nlpai-lab/KURE-v1'

        print(f"Using generator: {cfg.pipeline.generator_type}")
        print(f"Model: {cfg.pipeline.generator_model_name}")

        print("Initializing generator...")
        generator = get_generator(cfg)

        query = "What are proteins?"
        context_docs = [
            "Proteins are large bio molecules consisting of amino acids.",
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
    parser = argparse.ArgumentParser(description="Test HuggingFace integration")
    parser.add_argument("--config-dir", type=str, help="Path to config directory (default: conf/)")
    parser.add_argument("--config-name", type=str, default="settings", help="Config name to use (default: settings)")
    args = parser.parse_args()

    print("Testing HuggingFace Integration")
    print("=" * 50)

    test_gpu_resources()

    embeddings_ok = test_embeddings()
    generation_ok = test_generation(args.config_dir, args.config_name)

    print("=" * 50)
    if embeddings_ok and generation_ok:
        print("✓ All tests passed! HuggingFace integration is working.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())