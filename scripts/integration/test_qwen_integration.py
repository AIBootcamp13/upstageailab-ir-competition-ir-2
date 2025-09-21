#!/usr/bin/env python3
"""
Test script to validate Qwen2:7b integration with the RAG pipeline.
"""

import sys
import os
sys.path.insert(0, 'src')

from ir_core.orchestration.pipeline import RAGPipeline
from ir_core.utils.core import read_jsonl
import json

def test_qwen_pipeline():
    """Test the RAG pipeline with Qwen2:7b model."""
    print("Testing RAG Pipeline with Qwen2:7b")
    print("=" * 50)

    # Initialize pipeline with Qwen
    pipeline = RAGPipeline(
        model_name='qwen2:7b',
        tool_prompt_description='Search scientific documents for relevant information',
        tool_calling_model='qwen2:7b'
    )

    # Test queries
    test_queries = [
        "나무의 분류에 대해 조사해 보기 위한 방법은?",
        "What are the main approaches for classifying trees in machine learning?",
        "How do decision trees work in classification tasks?"
    ]

    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        try:
            # Run retrieval only to see enhanced query
            retrieval_result = pipeline.run_retrieval_only(query)
            enhanced_query = retrieval_result[0].get("standalone_query", query)
            docs_count = len(retrieval_result[0].get("docs", []))

            # Run full pipeline
            answer = pipeline.run(query)

            result = {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "retrieved_docs": docs_count,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer
            }
            results.append(result)

            print(f"  Enhanced: {enhanced_query}")
            print(f"  Retrieved: {docs_count} documents")
            print(f"  Answer: {result['answer']}")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "original_query": query,
                "error": str(e)
            })

    return results

def validate_with_sample_data():
    """Validate using sample validation data."""
    print("\nValidating with Sample Data")
    print("=" * 30)

    # Load sample validation data
    validation_file = "data/validation_balanced.jsonl"
    if not os.path.exists(validation_file):
        print(f"Validation file not found: {validation_file}")
        return

    # Initialize pipeline
    pipeline = RAGPipeline(
        model_name='qwen2:7b',
        tool_prompt_description='Search scientific documents for relevant information',
        tool_calling_model='qwen2:7b'
    )

    # Process first 3 validation queries
    scores = []
    for i, item in enumerate(read_jsonl(validation_file)):
        if i >= 3:  # Only test first 3
            break

        query = item.get("query", "")
        expected_answer = item.get("answer", "")

        print(f"\nValidation Query {i+1}: {query[:100]}...")

        try:
            # Generate answer
            generated_answer = pipeline.run(query)

            # Simple scoring (just check if answer is generated)
            score = 1.0 if generated_answer and len(generated_answer) > 10 else 0.0
            scores.append(score)

            print(f"  Generated answer length: {len(generated_answer)}")
            print(f"  Score: {score}")

        except Exception as e:
            print(f"  Error: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\nAverage Score: {avg_score:.2f}")

    return avg_score

if __name__ == "__main__":
    # Test basic functionality
    test_results = test_qwen_pipeline()

    # Validate with sample data
    avg_score = validate_with_sample_data()

    print("\n" + "=" * 50)
    print("Qwen2:7b Integration Test Complete")
    print(f"Basic tests passed: {len([r for r in test_results if 'error' not in r])}/{len(test_results)}")
    print(".2f")
    print("=" * 50)