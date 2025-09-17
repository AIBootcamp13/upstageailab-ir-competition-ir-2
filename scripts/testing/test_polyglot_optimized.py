#!/usr/bin/env python3
"""
Test script for Polyglot-Ko embedding provider with memory optimization
"""

import sys
import os
sys.path.insert(0, 'src')

def test_polyglot_optimization():
    """Test the PolyglotKo embedding provider with memory optimizations"""
    try:
        from ir_core.embeddings.polyglot import PolyglotKoEmbeddingProvider
        print("✅ PolyglotKoEmbeddingProvider imported successfully")

        # Test instantiation with smaller model
        provider = PolyglotKoEmbeddingProvider()
        print(f"✅ Provider instantiated with model: {provider.model_name}")
        print(f"✅ Quantization: {provider.quantization}")
        print(f"✅ Batch size: {provider.batch_size}")
        print(f"✅ Max threads: {provider.max_threads}")
        print(f"✅ Dimension: {provider.dimension}")

        # Test with very small batch to avoid memory issues
        test_texts = ["Hello world", "This is a test"]
        print("🧪 Testing encoding with small batch...")
        embeddings = provider.encode_texts(test_texts, batch_size=1)
        print(f"✅ Encoded {len(test_texts)} texts to shape: {embeddings.shape}")
        print(f"✅ Embedding dimension: {embeddings.shape[1]}")

        # Test memory usage
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 Memory usage: {memory_usage:.1f} MB")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Polyglot-Ko Embedding Provider with Memory Optimization...")
    success = test_polyglot_optimization()
    if success:
        print("✅ All tests passed!")
        print("\n💡 Memory Optimization Tips:")
        print("   - Use smaller models (1.3B, 3.8B, 5.8B) for limited RAM")
        print("   - 16-bit quantization provides good balance of memory/performance")
        print("   - Reduce batch size and max threads for memory-constrained systems")
        print("   - Consider CPU-only operation for very limited GPU memory")
    else:
        print("❌ Tests failed!")
        sys.exit(1)