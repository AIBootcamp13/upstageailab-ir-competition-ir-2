#!/usr/bin/env python3
"""
Fine-tuning script for retrieval models using enhanced validation data.

This script supports fine-tuning:
1. Embedding models (Polyglot-Ko) using Multiple Negatives Ranking Loss
2. Cross-encoder reranker models for improved ranking

Usage:
    PYTHONPATH=src uv run python scripts/fine_tuning/fine_tune_retrieval.py --config-path ../../conf --config-name settings
"""

import os
import sys
import torch
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

# NOTE: The custom `utils` import is assumed to be correct for the project structure.
from ir_core.utils import read_jsonl

# --- Removed the unused and incorrect `RetrievalDataset` class ---

class EmbeddingDataset(Dataset):
    """Dataset for embedding model training."""
    def __init__(self, examples: List[InputExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def prepare_embedding_training_data(data: List[Dict[str, Any]]) -> List[InputExample]:
    """
    Prepare training data for MultipleNegativesRankingLoss.
    This loss only requires positive (query, positive_document) pairs.
    """
    train_examples = []
    for item in data:
        query = item.get("query")
        positive_docs = item.get("ideal_context", [])
        if not query:
            continue
        for pos_doc in positive_docs:
            train_examples.append(InputExample(texts=[query, pos_doc]))
    return train_examples


def prepare_reranker_training_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare training data for reranker model fine-tuning."""
    examples = []
    for item in data:
        query = item["query"]
        # Add positive examples
        for pos_doc in item.get("ideal_context", []):
            examples.append({"query": query, "document": pos_doc, "label": 1})
        # Add negative examples
        for neg_doc in item.get("hard_negative_context", []):
            examples.append({"query": query, "document": neg_doc, "label": 0})
    return examples


class RerankerDataset(Dataset):
    """Dataset for reranker training."""
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        document = item["document"]
        label = item["label"]

        # Tokenize the query-document pair
        encoding = self.tokenizer(
            query,
            document,
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
    """Fine-tune embedding model using Multiple Negatives Ranking Loss."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Preparing training data for embedding model...")
    train_examples = prepare_embedding_training_data(train_data)
    train_dataloader = DataLoader(EmbeddingDataset(train_examples), shuffle=True, batch_size=batch_size)

    # **FIXED**: Use the more effective MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # **FIXED**: Correctly prepare the InformationRetrievalEvaluator
    val_evaluator = None
    if val_data:
        print("Preparing validation data for evaluator...")
        queries, corpus, relevant_docs = {}, {}, {}
        for item in val_data:
            query = item['query']
            # Use a unique ID for each query to handle potential duplicates
            query_id = f"q_{len(queries)}"
            queries[query_id] = query
            relevant_docs[query_id] = set()

            # Add positive documents to the corpus and mark them as relevant
            for doc in item.get('ideal_context', []):
                doc_id = f"doc_{len(corpus)}"
                corpus[doc_id] = doc
                relevant_docs[query_id].add(doc_id)

            # Add hard negative documents to the corpus so they can be ranked
            for doc in item.get('hard_negative_context', []):
                doc_id = f"doc_{len(corpus)}"
                corpus[doc_id] = doc

        val_evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs)

    print(f"Fine-tuning embedding model for {epochs} epochs...")
    warmup_steps = int(len(train_dataloader) * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        save_best_model=True if val_evaluator else False,
        show_progress_bar=True
    )

    print(f"Embedding model fine-tuned and saved to: {output_dir}")
    return model


def fine_tune_reranker_model(
    train_data: List[Dict[str, Any]],
    val_data: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "klue/roberta-large",
    output_dir: str = "models/fine_tuned_reranker",
    epochs: int = 3,
    batch_size: int = 8
):
    """Fine-tune reranker model using cross-encoder approach."""
    print(f"Loading reranker model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Preparing training data for reranker...")
    train_examples = prepare_reranker_training_data(train_data)
    train_dataset = RerankerDataset(train_examples, tokenizer)

    val_dataset = None
    if val_data:
        print("Preparing validation data for reranker...")
        val_examples = prepare_reranker_training_data(val_data)
        val_dataset = RerankerDataset(val_examples, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print(f"Fine-tuning reranker model for {epochs} epochs...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Reranker model fine-tuned and saved to: {output_dir}")
    return model, tokenizer


@hydra.main(config_path="../../conf", config_name="settings", version_base=None)
def main(cfg: DictConfig):
    """Main fine-tuning function."""
    _add_src_to_path()

    enhanced_data_path = cfg.data.eval_enhanced.output_file
    fine_tune_config = cfg.fine_tune

    model_type = fine_tune_config.model_type
    base_model = fine_tune_config.base_model
    output_dir = fine_tune_config.output_dir
    epochs = fine_tune_config.epochs
    batch_size = fine_tune_config.batch_size
    val_split = fine_tune_config.val_split

    print("=== Retrieval Model Fine-tuning ===")
    print(f"Model type: {model_type}")
    print(f"Base model: {base_model}")
    print(f"Enhanced data: {enhanced_data_path}")
    print(f"Output directory: {output_dir}")

    print("Loading enhanced data...")
    if not os.path.exists(enhanced_data_path):
        raise FileNotFoundError(f"Enhanced data file not found at: {enhanced_data_path}")

    data = list(read_jsonl(enhanced_data_path))
    print(f"Loaded {len(data)} examples")

    if val_split > 0:
        split_idx = int(len(data) * (1 - val_split))
        train_data, val_data = data[:split_idx], data[split_idx:]
        print(f"Split data into Train: {len(train_data)}, Val: {len(val_data)}")
    else:
        train_data, val_data = data, None

    os.makedirs(output_dir, exist_ok=True)

    if model_type == "embedding":
        fine_tune_embedding_model(
            train_data=train_data, val_data=val_data, model_name=base_model,
            output_dir=output_dir, epochs=epochs, batch_size=batch_size
        )
    elif model_type == "reranker":
        fine_tune_reranker_model(
            train_data=train_data, val_data=val_data, model_name=base_model,
            output_dir=output_dir, epochs=epochs, batch_size=batch_size
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'embedding' or 'reranker'.")

    print("\n✅ Fine-tuning completed!")
    print(f"Final model saved in: {output_dir}")


if __name__ == "__main__":
    main()