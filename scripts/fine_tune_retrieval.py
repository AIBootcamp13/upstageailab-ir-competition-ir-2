#!/usr/bin/env python3
"""
Fine-tuning script for retrieval models using enhanced validation data.

This script supports fine-tuning:
1. Embedding models (Polyglot-Ko) using contrastive learning
2. Cross-encoder reranker models for improved ranking

Usage:
    PYTHONPATH=src poetry run python scripts/fine_tune_retrieval.py --config-path ../../conf --config-name settings
"""

import os
import sys
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
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


class RetrievalDataset(Dataset):
    """Dataset for fine-tuning retrieval models."""

    def __init__(self, data: List[Dict[str, Any]], model_type: str = "embedding"):
        """
        Args:
            data: List of enhanced validation data
            model_type: "embedding" or "reranker"
        """
        self.data = data
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.model_type == "embedding":
            # For embedding fine-tuning, return InputExample for contrastive learning
            query = item["query"]
            positive_docs = item.get("ideal_context", [])
            negative_docs = item.get("hard_negative_context", [])

            # Create positive examples
            examples = []
            for pos_doc in positive_docs:
                examples.append(InputExample(texts=[query, pos_doc], label=1.0))

            # Create negative examples
            for neg_doc in negative_docs[:2]:  # Limit negatives to avoid too many
                examples.append(InputExample(texts=[query, neg_doc], label=0.0))

            return examples

        elif self.model_type == "reranker":
            # For reranker fine-tuning, return query-doc pairs with labels
            query = item["query"]
            positive_docs = item.get("ideal_context", [])
            negative_docs = item.get("hard_negative_context", [])

            examples = []
            for pos_doc in positive_docs:
                examples.append({
                    "query": query,
                    "document": pos_doc,
                    "label": 1  # Positive
                })

            for neg_doc in negative_docs:
                examples.append({
                    "query": query,
                    "document": neg_doc,
                    "label": 0  # Negative
                })

            return examples

        return None


def prepare_embedding_training_data(data: List[Dict[str, Any]]) -> List[InputExample]:
    """Prepare training data for embedding model fine-tuning."""
    train_examples = []

    for item in data:
        query = item["query"]
        positive_docs = item.get("ideal_context", [])
        negative_docs = item.get("hard_negative_context", [])

        # Add positive examples
        for pos_doc in positive_docs:
            train_examples.append(InputExample(
                texts=[query, pos_doc],
                label=1.0
            ))

        # Add negative examples (limit to 2 per query to balance)
        for neg_doc in negative_docs[:2]:
            train_examples.append(InputExample(
                texts=[query, neg_doc],
                label=0.0
            ))

    return train_examples


def prepare_reranker_training_data(data: List[Dict[str, Any]]) -> Dict[str, List]:
    """Prepare training data for reranker model fine-tuning."""
    queries = []
    documents = []
    labels = []

    for item in data:
        query = item["query"]
        positive_docs = item.get("ideal_context", [])
        negative_docs = item.get("hard_negative_context", [])

        # Add positive examples
        for pos_doc in positive_docs:
            queries.append(query)
            documents.append(pos_doc)
            labels.append(1)

        # Add negative examples
        for neg_doc in negative_docs:
            queries.append(query)
            documents.append(neg_doc)
            labels.append(0)

    return {
        "queries": queries,
        "documents": documents,
        "labels": labels
    }


class RerankerDataset(Dataset):
    """Dataset for reranker training."""

    def __init__(self, queries: List[str], documents: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        document = self.documents[idx]
        label = self.labels[idx]

        # Combine query and document
        text_pair = f"{query} [SEP] {document}"

        encoding = self.tokenizer(
            text_pair,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def fine_tune_embedding_model(
    train_data: List[Dict[str, Any]],
    val_data: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "EleutherAI/polyglot-ko-1.3b",
    output_dir: str = "models/fine_tuned_embedding",
    epochs: int = 3,
    batch_size: int = 16
):
    """Fine-tune embedding model using contrastive learning."""

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Prepare training data
    train_examples = prepare_embedding_training_data(train_data)

    # Prepare validation data if provided
    val_evaluator = None
    if val_data:
        val_examples = prepare_embedding_training_data(val_data)
        # Create evaluator for validation
        val_evaluator = InformationRetrievalEvaluator(
            queries=[ex.texts[0] for ex in val_examples[:100]],  # Sample for evaluation
            corpus={f"doc_{i}": ex.texts[1] for i, ex in enumerate(val_examples[:100])},
            relevant_docs={f"query_{i}": {f"doc_{i}"} for i in range(min(100, len(val_examples)))}
        )

    # Define loss function
    train_loss = losses.ContrastiveLoss(model)

    # Fine-tune the model
    print(f"Fine-tuning embedding model for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_examples, train_loss)],
        evaluator=val_evaluator,
        epochs=epochs,
        warmup_steps=int(len(train_examples) * 0.1),
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True
    )

    print(f"Embedding model fine-tuned and saved to: {output_dir}")
    return model


def fine_tune_reranker_model(
    train_data: List[Dict[str, Any]],
    val_data: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "bert-base-multilingual-cased",
    output_dir: str = "models/fine_tuned_reranker",
    epochs: int = 3,
    batch_size: int = 8
):
    """Fine-tune reranker model using cross-encoder approach."""

    print(f"Loading reranker model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Prepare training data
    train_dict = prepare_reranker_training_data(train_data)
    train_dataset = RerankerDataset(
        train_dict["queries"],
        train_dict["documents"],
        train_dict["labels"],
        tokenizer
    )

    # Prepare validation data
    val_dataset = None
    if val_data:
        val_dict = prepare_reranker_training_data(val_data)
        val_dataset = RerankerDataset(
            val_dict["queries"],
            val_dict["documents"],
            val_dict["labels"],
            tokenizer
        )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=500 if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="accuracy" if val_dataset else None,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    print(f"Fine-tuning reranker model for {epochs} epochs...")
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Reranker model fine-tuned and saved to: {output_dir}")
    return model, tokenizer


@hydra.main(config_path="../../conf", config_name="settings", version_base=None)
def main(cfg: DictConfig):
    """Main fine-tuning function."""
    _add_src_to_path()

    # Configuration
    enhanced_data_path = cfg.get("enhanced_validation", {}).get("output_file", "data/eval_enhanced.jsonl")
    fine_tune_config = cfg.get("fine_tune", {})

    model_type = fine_tune_config.get("model_type", "embedding")  # "embedding" or "reranker"
    base_model = fine_tune_config.get("base_model", "EleutherAI/polyglot-ko-1.3b")
    output_dir = fine_tune_config.get("output_dir", f"models/fine_tuned_{model_type}")
    epochs = fine_tune_config.get("epochs", 3)
    batch_size = fine_tune_config.get("batch_size", 16)
    val_split = fine_tune_config.get("val_split", 0.1)

    print("=== Retrieval Model Fine-tuning ===")
    print(f"Model type: {model_type}")
    print(f"Base model: {base_model}")
    print(f"Enhanced data: {enhanced_data_path}")
    print(f"Output directory: {output_dir}")

    # Load enhanced validation data
    print("Loading enhanced validation data...")
    data = list(read_jsonl(enhanced_data_path))
    print(f"Loaded {len(data)} training examples")

    # Split into train/val if validation is requested
    if val_split > 0:
        split_idx = int(len(data) * (1 - val_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    else:
        train_data = data
        val_data = None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Fine-tune based on model type
    if model_type == "embedding":
        fine_tune_embedding_model(
            train_data=train_data,
            val_data=val_data,
            model_name=base_model,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size
        )
    elif model_type == "reranker":
        fine_tune_reranker_model(
            train_data=train_data,
            val_data=val_data,
            model_name=base_model,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print("✅ Fine-tuning completed!")


if __name__ == "__main__":
    main()</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/scripts/fine_tune_retrieval.py