#!/usr/bin/env python3
"""
Ollama Integration Demo for Scientific QA System

This script demonstrates how to integrate Ollama models into the existing
Scientific QA pipeline for cost-effective, local LLM usage.
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ir_core.utils.ollama_client import (
    OllamaClient,
    rewrite_query_ollama,
    generate_answer_ollama,
    generate_validation_queries_ollama,
    benchmark_ollama_model
)


def demo_query_rewriting():
    """Demonstrate query rewriting with Ollama."""
    print("🔄 Query Rewriting Demo")
    print("=" * 50)

    client = OllamaClient(model="llama3.1:8b")

    test_queries = [
        "태양계의 나이를 알아내는 데 어떤 방법들이 사용되고 있을까요?",
        "원자의 구성 요소와 원자의 화학 반응에서의 역할은 무엇인가요?",
        "가금류 알 내부에서 발견되는 난백은 어떤 역할을 하는 건가요?",
        "빅뱅 이론은 무엇인가요?"
    ]

    print("Original → Rewritten")
    print("-" * 50)

    for query in test_queries:
        start_time = time.time()
        rewritten = rewrite_query_ollama(query, client)
        elapsed = time.time() - start_time

        print(f"Original: {query}")
        print(f"Rewritten: {rewritten}")
        print(".2f")
        print()


def demo_answer_generation():
    """Demonstrate answer generation with Ollama."""
    print("🤖 Answer Generation Demo")
    print("=" * 50)

    client = OllamaClient(model="llama3.1:8b")

    # Sample query and context
    query = "원자의 기본 구성 요소는 무엇인가요?"
    context_docs = [
        "원자는 물질의 기본 단위로, 양성자, 중성자, 전자로 구성되어 있습니다. 양성자와 중성자는 원자핵에 위치하며, 전자는 원자핵 주위를 공전합니다.",
        "원자 번호는 원자핵에 있는 양성자의 수를 나타내며, 각 원소의 특징을 결정합니다.",
        "전자 배치는 원자의 화학적 성질을 결정하는 중요한 요소입니다."
    ]

    print(f"Query: {query}")
    print(f"Context documents: {len(context_docs)}")
    print()

    start_time = time.time()
    answer = generate_answer_ollama(query, context_docs, client)
    elapsed = time.time() - start_time

    print("Generated Answer:")
    print(answer)
    print(".2f")


def demo_validation_query_generation():
    """Demonstrate validation query generation with Ollama."""
    print("📝 Validation Query Generation Demo")
    print("=" * 50)

    client = OllamaClient(model="llama3.1:8b")

    domains = {
        "physics": "물리학 (힘, 에너지, 운동, 원자, 입자 등)",
        "chemistry": "화학 (화합물, 반응, 원소, 산, 염기 등)",
        "biology": "생물학 (세포, 유전자, 단백질, 생명, 진화 등)"
    }

    for domain, description in domains.items():
        print(f"\n🔬 Generating queries for {domain}")
        print(f"Description: {description}")

        start_time = time.time()
        queries = generate_validation_queries_ollama(domain, description, 3, client)
        elapsed = time.time() - start_time

        print("Generated queries:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
        print(".2f")


def demo_performance_benchmark():
    """Demonstrate performance benchmarking."""
    print("⚡ Performance Benchmark Demo")
    print("=" * 50)

    client = OllamaClient(model="llama3.1:8b")

    test_prompts = [
        "원자의 구조에 대해 설명해주세요.",
        "화학 반응에서 촉매의 역할은 무엇인가요?",
        "DNA 복제 과정은 어떻게 이루어지나요?",
        "빅뱅 이론의 주요 증거는 무엇인가요?"
    ]

    print("Benchmarking Ollama model performance...")
    print(f"Model: {client.default_model}")
    print(f"Test prompts: {len(test_prompts)}")
    print()

    results = benchmark_ollama_model(client, test_prompts, num_runs=2)

    print("📊 Benchmark Results:")
    print(".1%")
    print()

    print("Per-prompt results:")
    for result in results["benchmark_results"]:
        print(f"Prompt: {result['prompt']}")
        print(".1%")
        print(".2f")
        print(".1f")
        print()


def demo_integration_comparison():
    """Compare OpenAI vs Ollama performance."""
    print("🔄 Integration Comparison: OpenAI vs Ollama")
    print("=" * 50)

    client = OllamaClient(model="llama3.1:8b")

    test_query = "태양계의 나이를 알아내는 데 어떤 방법들이 사용되고 있을까요?"

    print(f"Test Query: {test_query}")
    print()

    # Ollama performance
    print("🦙 Ollama Performance:")
    start_time = time.time()
    ollama_result = rewrite_query_ollama(test_query, client)
    ollama_time = time.time() - start_time

    print(f"Rewritten: {ollama_result}")
    print(".2f")
    print("Cost: $0.00 (local)")
    print()

    # OpenAI comparison (estimated)
    print("🤖 OpenAI GPT-3.5-turbo Comparison:")
    print("Estimated time: 1-3 seconds")
    print("Estimated cost: $0.001-0.003 per request")
    print("Privacy: Data sent to OpenAI servers")
    print()

    print("💰 Cost Savings:")
    print("- Ollama: Completely free (one-time model download)")
    print("- OpenAI: ~$0.002 per request × 1000 requests = $2.00")
    print("- Monthly savings: Significant for high-volume usage")


def main():
    """Run all Ollama integration demos."""
    print("🚀 Ollama Integration Demo for Scientific QA System")
    print("=" * 60)
    print("This demo shows how to leverage local Ollama models for:")
    print("• Query rewriting")
    print("• Answer generation")
    print("• Validation data creation")
    print("• Performance benchmarking")
    print()

    try:
        # Check Ollama availability
        client = OllamaClient()
        if not client.check_health():
            print("❌ Ollama server not available. Please start Ollama first:")
            print("   ollama serve")
            return

        models = client.list_models()
        if not models:
            print("❌ No models available. Please pull a model first:")
            print("   ollama pull llama3.1:8b")
            return

        print(f"✅ Ollama ready with models: {[m['name'] for m in models]}")
        print()

        # Run demos
        demo_query_rewriting()
        print("\n" + "="*60 + "\n")

        demo_answer_generation()
        print("\n" + "="*60 + "\n")

        demo_validation_query_generation()
        print("\n" + "="*60 + "\n")

        demo_performance_benchmark()
        print("\n" + "="*60 + "\n")

        demo_integration_comparison()

        print("\n" + "="*60)
        print("🎉 Ollama Integration Demo Complete!")
        print("\nNext Steps:")
        print("1. Integrate Ollama into your validation scripts")
        print("2. Replace OpenAI calls with Ollama for cost savings")
        print("3. Experiment with different models (llama3.1:70b, codellama, etc.)")
        print("4. Set up automated model updates and performance monitoring")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure Ollama is running and accessible")


if __name__ == "__main__":
    main()
