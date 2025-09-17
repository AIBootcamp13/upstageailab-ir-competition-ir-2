#!/usr/bin/env python3
"""
Integration script for using fine-tuned retrieval models.

This script provides utilities to:
1. Load fine-tuned embedding models
2. Load fine-tuned reranker models
3. Test fine-tuned models against baseline
4. Update configuration to use fine-tuned models
"""

import os
import sys
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import hydra
from omegaconf import DictConfig

# Add src to path
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from ir_core.utils import read_jsonl
from ir_core.retrieval.core import hybrid_retrieve


class FineTunedRetriever:
    """Wrapper for using fine-tuned retrieval models."""

    def __init__(self, embedding_model_path: Optional[str] = None, reranker_model_path: Optional[str] = None):
        """
        Initialize with fine-tuned models.

        Args:
            embedding_model_path: Path to fine-tuned embedding model
            reranker_model_path: Path to fine-tuned reranker model
        """
        self.embedding_model = None
        self.reranker_model = None
        self.reranker_tokenizer = None

        if embedding_model_path and os.path.exists(embedding_model_path):
            print(f"Loading fine-tuned embedding model from: {embedding_model_path}")
            self.embedding_model = SentenceTransformer(embedding_model_path)

        if reranker_model_path and os.path.exists(reranker_model_path):
            print(f"Loading fine-tuned reranker model from: {reranker_model_path}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_path)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using fine-tuned embedding model."""
        if self.embedding_model:
            return self.embedding_model.encode(query, convert_to_numpy=True)
        else:
            # Fallback to original encoding
            from ir_core.embeddings.core import encode_query
            return encode_query(query)

    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank documents using fine-tuned reranker model.

        Args:
            query: Search query
            documents: List of document dictionaries
            top_k: Number of top documents to return

        Returns:
            Reranked documents
        """
        if not self.reranker_model or not self.reranker_tokenizer:
            return documents  # Return original order if no reranker

        reranked_docs = []

        for doc in documents:
            content = doc.get("content", "")
            text_pair = f"{query} [SEP] {content}"

            # Tokenize
            inputs = self.reranker_tokenizer(
                text_pair,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            # Get reranker score
            with torch.no_grad():
                outputs = self.reranker_model(**inputs)
                score = torch.softmax(outputs.logits, dim=1)[0][1].item()  # Probability of positive class

            # Add reranker score to document
            doc_copy = doc.copy()
            doc_copy["reranker_score"] = score
            reranked_docs.append(doc_copy)

        # Sort by reranker score (descending)
        reranked_docs.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)

        return reranked_docs[:top_k]

    def retrieve_and_rerank(self, query: str, bm25_k: int = 50, rerank_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform retrieval with fine-tuned models.

        Args:
            query: Search query
            bm25_k: Number of BM25 results to retrieve initially
            rerank_k: Number of final results to return

        Returns:
            Retrieved and reranked documents
        """
        # Get initial retrieval results
        initial_results = hybrid_retrieve(query=query, bm25_k=bm25_k, rerank_k=bm25_k)

        # Convert to document format expected by reranker
        documents = []
        for result in initial_results:
            hit = result.get("hit", {})
            src = hit.get("_source", {})
            doc = {
                "content": src.get("content", ""),
                "id": hit.get("_id", ""),
                "score": result.get("score", 0),
                "src": src.get("src", ""),
                "title": src.get("title", "")
            }
            documents.append(doc)

        # Apply reranking if available
        if self.reranker_model:
            documents = self.rerank_documents(query, documents, rerank_k)

        return documents


def evaluate_fine_tuned_model(
    retriever: FineTunedRetriever,
    test_data: List[Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, float]:
    """
    Evaluate fine-tuned model performance.

    Args:
        retriever: Fine-tuned retriever instance
        test_data: Test data with ideal contexts
        top_k: Number of documents to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    total_queries = len(test_data)
    recall_at_k = 0
    precision_at_k = 0

    print(f"Evaluating on {total_queries} queries...")

    for i, item in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processed {i}/{total_queries} queries")

        query = item["query"]
        ideal_contexts = set(item.get("ideal_context", []))

        # Get retrieval results
        results = retriever.retrieve_and_rerank(query, rerank_k=top_k)

        # Check if any ideal context is in top-k results
        retrieved_contents = set()
        for result in results:
            content = result.get("content", "")
            # Simple substring matching (could be improved with better similarity)
            for ideal in ideal_contexts:
                if ideal.strip() in content or content.strip() in ideal:
                    retrieved_contents.add(ideal)
                    break

        # Calculate metrics
        if ideal_contexts:
            recall = len(retrieved_contents) / len(ideal_contexts)
            precision = len(retrieved_contents) / top_k
            recall_at_k += recall
            precision_at_k += precision

    # Average metrics
    avg_recall = recall_at_k / total_queries
    avg_precision = precision_at_k / total_queries

    return {
        "recall_at_k": avg_recall,
        "precision_at_k": avg_precision,
        "total_queries": total_queries
    }


@hydra.main(config_path="../../conf", config_name="settings", version_base=None)
def main(cfg: DictConfig):
    """Main function for testing fine-tuned models."""
    _add_src_to_path()

    # Configuration
    fine_tune_config = cfg.get("fine_tune", {})
    embedding_model_path = fine_tune_config.get("output_dir", "models/fine_tuned_embedding")
    reranker_model_path = fine_tune_config.get("output_dir", "models/fine_tuned_reranker")

    # Override with specific paths if provided
    embedding_model_path = cfg.get("test_fine_tuned", {}).get("embedding_model_path", embedding_model_path)
    reranker_model_path = cfg.get("test_fine_tuned", {}).get("reranker_model_path", reranker_model_path)

    enhanced_data_path = cfg.get("enhanced_validation", {}).get("output_file", "data/eval_enhanced.jsonl")

    print("=== Testing Fine-tuned Retrieval Models ===")
    print(f"Embedding model: {embedding_model_path}")
    print(f"Reranker model: {reranker_model_path}")
    print(f"Test data: {enhanced_data_path}")

    # Load fine-tuned models
    retriever = FineTunedRetriever(
        embedding_model_path=embedding_model_path,
        reranker_model_path=reranker_model_path
    )

    # Load test data
    test_data = list(read_jsonl(enhanced_data_path))
    print(f"Loaded {len(test_data)} test queries")

    # Evaluate performance
    metrics = evaluate_fine_tuned_model(retriever, test_data, top_k=3)

    print("\n=== Evaluation Results ===")
    print(".3f")
    print(".3f")
    print(f"Total queries evaluated: {metrics['total_queries']}")

    # Save results
    results_file = "outputs/fine_tuned_evaluation.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/scripts/test_fine_tuned_models.py