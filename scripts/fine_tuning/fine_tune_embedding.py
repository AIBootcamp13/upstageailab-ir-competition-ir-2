#!/usr/bin/env python3
"""
Fine-tune KURE-v1 Embedding Model for Scientific Document Retrieval (Memory-Efficient Version)

This script fine-tunes the nlpai-lab/KURE-v1 model using enhanced evaluation data
to improve dense retrieval alignment with sparse retrieval results.

Supports both document format (with hypothetical_questions) and eval_enhanced format
(with ideal_context and hard_negative_context).

Usage:
    PYTHONPATH=src uv run python scripts/fine_tuning/fine_tune_embedding.py
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Iterator
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

# --- Memory-Efficient Streaming Dataset Class ---
class StreamingJsonlDataset(Dataset):
    """
    An iterable dataset that reads from a JSONL file line by line.
    This avoids loading the entire dataset into memory.

    Supports both document format (with hypothetical_questions) and
    eval_enhanced format (with ideal_context and hard_negative_context).
    """
    def __init__(self, jsonl_path: str):
        self.file_path = jsonl_path
        # First, we need the total length for the DataLoader.
        # This is a quick pass and doesn't load content into memory.
        print("Counting examples in dataset...")
        self._len = 0
        self.is_eval_enhanced = False

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        doc = json.loads(line)
                        # Check if this is eval_enhanced format
                        if 'ideal_context' in doc and 'hard_negative_context' in doc:
                            self.is_eval_enhanced = True
                            # Count positive examples (query + ideal_context)
                            if doc.get('ideal_context') and len(doc['ideal_context']) > 0:
                                self._len += len(doc['ideal_context'])
                            # Count negative examples (query + hard_negative_context)
                            if doc.get('hard_negative_context') and len(doc['hard_negative_context']) > 0:
                                self._len += len(doc['hard_negative_context'])
                        else:
                            # Original format with hypothetical_questions
                            for question in doc.get('hypothetical_questions', []):
                                if question and len(question.strip()) > 10:
                                    self._len += 1
                    except json.JSONDecodeError:
                        print("Warning: Skipping malformed JSON line.")
        print(f"Found {self._len} total examples.")
        print(f"Dataset format: {'eval_enhanced' if self.is_eval_enhanced else 'documents'}")

    def __len__(self):
        return self._len

    def __iter__(self) -> Iterator[InputExample]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    doc = json.loads(line)

                    if self.is_eval_enhanced:
                        # Handle eval_enhanced format
                        query = doc.get('query', '').strip()
                        if not query:
                            continue

                        # Positive examples: query + ideal_context
                        ideal_context = doc.get('ideal_context', [])
                        for context in ideal_context:
                            if context and len(context.strip()) > 10:
                                yield InputExample(texts=[query, context.strip()], label=1.0)

                        # Negative examples: query + hard_negative_context
                        hard_negative_context = doc.get('hard_negative_context', [])
                        for context in hard_negative_context:
                            if context and len(context.strip()) > 10:
                                yield InputExample(texts=[query, context.strip()], label=0.0)
                    else:
                        # Handle original document format
                        content = doc.get('content', '')
                        summary = doc.get('summary', '')
                        hypothetical_questions = doc.get('hypothetical_questions', [])

                        passage = summary or content[:512]

                        for question in hypothetical_questions:
                            if question and len(question.strip()) > 10:
                                yield InputExample(texts=[question.strip(), passage], label=1.0)

                except json.JSONDecodeError:
                    # Silently skip malformed lines during iteration
                    continue

# --- Memory-efficient chunk dataset for training ---
class ChunkDataset(Dataset):
    """Simple dataset for training chunks."""
    def __init__(self, examples: List[InputExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputExample:
        return self.examples[idx]

def fine_tune_model(
    base_model_name: str,
    train_dataset: StreamingJsonlDataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,  # Reduced batch size
    val_split: float = 0.1 # This parameter is currently unused with streaming
):
    """Fine-tune the embedding model using Multiple Negatives Ranking Loss."""

    # Force CPU to avoid OOM
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"Loading base model components: {base_model_name}")

    # --- Manually load the transformer model to enable gradient checkpointing ---
    # 1. Load the raw transformer model from Hugging Face
    word_embedding_model = models.Transformer(base_model_name)

    # 2. Enable gradient checkpointing on the transformer model itself
    try:
        # This is the correct way to do it for models that support it
        if hasattr(word_embedding_model, 'model') and hasattr(word_embedding_model.model, 'gradient_checkpointing_enable'):
            word_embedding_model.model.gradient_checkpointing_enable() # type: ignore
            print("Enabled gradient checkpointing for memory efficiency")
        else:
            print("Gradient checkpointing not available for this model")
    except Exception as e:
        print(f"Could not enable gradient checkpointing: {e}")

    # 3. Create a pooling layer (mean pooling is a good default)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    # 4. Create the SentenceTransformer model from our modules
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Force CPU usage to avoid CUDA issues
    model.to('cpu')
    print("Model forced to CPU to avoid CUDA compatibility issues.")

    print("Model loaded successfully.")
    print("Training data: Streaming dataset (size determined at runtime)")
    print("Note: Validation split is ignored for streaming datasets.")

    # Use Multiple Negatives Ranking Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting fine-tuning for {epochs} epochs...")
    print(f"Output directory: {output_path}")

    chunk_size = 2000  # Process in chunks of 2000 examples

    # --- FIXED: Added main epoch loop ---
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        all_examples = []

        # The StreamingJsonlDataset's __iter__ method will be called anew for each epoch,
        # effectively re-reading the data from the file.
        for i, example in enumerate(train_dataset):
            all_examples.append(example)

            # Train when a chunk is full
            if (i + 1) % chunk_size == 0:
                print(f"Training on chunk ending at example {i + 1}...")
                chunk_dataset = ChunkDataset(all_examples)
                # Use pin_memory=false since we're on CPU
                chunk_dataloader = DataLoader(
                    chunk_dataset, 
                    shuffle=True, 
                    batch_size=batch_size,
                    pin_memory=False
                )

                model.fit(
                    train_objectives=[(chunk_dataloader, train_loss)],
                    evaluator=None,
                    epochs=1, # We train for 1 epoch on each chunk
                    warmup_steps=max(1, len(chunk_dataloader) // 10),
                    output_path=None,  # Don't save intermediate models
                    save_best_model=False,
                    show_progress_bar=True,
                    use_amp=False  # Disable AMP to save memory
                )

                # Clear the chunk to free memory
                all_examples = []

        # Process any remaining examples at the end of the epoch
        if all_examples:
            print(f"Training on final chunk of {len(all_examples)} examples for this epoch...")
            chunk_dataset = ChunkDataset(all_examples)
            # Use pin_memory=false since we're on CPU
            chunk_dataloader = DataLoader(
                chunk_dataset, 
                shuffle=True, 
                batch_size=batch_size,
                pin_memory=False
            )

            model.fit(
                train_objectives=[(chunk_dataloader, train_loss)],
                evaluator=None,
                epochs=1,
                warmup_steps=max(1, len(chunk_dataloader) // 10),
                output_path=None,
                save_best_model=False,
                show_progress_bar=True,
                use_amp=False  # Disable AMP
            )

    # Save the final model after all epochs are complete
    model.save(str(output_path))
    print(f"\nFine-tuned model saved to: {output_path}")
    return str(output_path)

# --- NOTE: The redundant second definition of fine_tune_model has been removed ---

def main():
    """Main fine-tuning pipeline."""
    config_path = Path(__file__).parent.parent.parent / "conf" / "settings.yaml"
    cfg = OmegaConf.load(config_path)
    data_config_path = Path(__file__).parent.parent.parent / "conf" / "data" / "science_qa_ko_metadata.yaml"
    data_cfg = OmegaConf.load(data_config_path)

    fine_tune_config = getattr(cfg, 'fine_tune', {})
    base_model = fine_tune_config.get('base_model', 'EleutherAI/polyglot-ko-1.3b')
    output_dir = fine_tune_config.get('output_dir', 'models/fine_tuned')
    epochs = fine_tune_config.get('epochs', 3)
    batch_size = fine_tune_config.get('batch_size', 4)
    val_split = fine_tune_config.get('val_split', 0.1)

    data_path = getattr(data_cfg, 'documents_path', 'data/documents_ko_with_metadata.jsonl')

    # For fine-tuning, use the enhanced evaluation dataset instead of documents
    enhanced_eval_path = 'data/eval_enhanced.jsonl'
    if Path(enhanced_eval_path).exists():
        data_path = enhanced_eval_path
        print(f"Using enhanced evaluation dataset: {data_path}")
    else:
        print(f"Enhanced dataset not found, using documents: {data_path}")

    print("=== Embedding Model Fine-tuning Pipeline ===")
    print(f"Base model: {base_model}")
    print(f"Data file: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("Device: CPU (forced for compatibility)")
    print()

    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    print("Initializing streaming dataset...")
    training_dataset = StreamingJsonlDataset(data_path)

    if len(training_dataset) == 0:
        print("Error: No training data found in the file. Check if documents have hypothetical_questions.")
        return 1

    print("Starting model fine-tuning...")
    model_path = fine_tune_model(
        base_model_name=base_model,
        train_dataset=training_dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split
    )

    print("\n=== Fine-tuning Complete ===")
    print(f"Fine-tuned model saved to: {model_path}")
    print()
    print("To use the fine-tuned model, update your settings.yaml:")
    print(f"  EMBEDDING_MODEL: {model_path}")
    print("  EMBEDDING_PROVIDER: sentence_transformers")

    return 0

if __name__ == "__main__":
    exit(main())