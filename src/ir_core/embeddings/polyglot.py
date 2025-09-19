# src/ir_core/embeddings/polyglot.py
from typing import List, Optional
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading

from .base import BaseEmbeddingProvider
from ..config import settings

class PolyglotKoEmbeddingProvider(BaseEmbeddingProvider):
    """
    Polyglot-Ko embedding provider using causal language model for embedding extraction.
    Only supports full precision (PolyGlot doesn't support quantization).
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize Polyglot-Ko embedding provider.

        Args:
            model_name: HuggingFace model name. If None, uses settings.POLYGLOT_MODEL.
        """
        self.model_name = model_name or getattr(settings, 'POLYGLOT_MODEL', 'EleutherAI/polyglot-ko-1.3b')
        self.batch_size: int = getattr(settings, 'POLYGLOT_BATCH_SIZE', 8) or 8
        self.max_threads = getattr(settings, 'POLYGLOT_MAX_THREADS', 8)
        self._tokenizer = None
        self._model = None
        self._device = None
        self._lock = threading.Lock()
        self._load_model()

    def _get_device(self):
        if self._device is None:
            device_str = getattr(settings, 'EMBEDDING_DEVICE', 'auto')
            if device_str == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(device_str)
        return self._device

    def _load_model(self):
        """Load tokenizer and model in full precision (PolyGlot doesn't support quantization)."""
        with self._lock:
            if self._tokenizer is None or self._model is None:
                print(f"🔄 Loading Polyglot-Ko model: {self.model_name} in full precision")

                # Load tokenizer with timeout
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, timeout=300)  # 5 minute timeout
                except Exception as e:
                    raise RuntimeError(f"Failed to load tokenizer for {self.model_name}: {e}")

                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                device = self._get_device()

                # Load model in full precision (PolyGlot doesn't support quantization)
                try:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        dtype=torch.float32,  # Use full precision
                        low_cpu_mem_usage=True
                    )
                    # Move to device explicitly
                    self._model = self._model.to(device)
                    print(f"✅ Loaded Polyglot-Ko model: {self.model_name} in full precision on {device}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

                # Assertions to help type checker
                assert self._tokenizer is not None
                assert self._model is not None
                self._model.eval()

    def _mean_pool(self, last_hidden, attention_mask):
        """Mean pool the hidden states."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).to(last_hidden.dtype)
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @property
    def dimension(self) -> int:
        """Return the embedding dimension based on the model."""
        # Map common Polyglot-Ko models to their dimensions
        model_dims = {
            'EleutherAI/polyglot-ko-1.3b': 2048,
            'EleutherAI/polyglot-ko-3.8b': 3072,
            'EleutherAI/polyglot-ko-5.8b': 4096,
            'EleutherAI/polyglot-ko-12.8b': 5120,
        }

        # Try to get dimension from model name mapping first
        if self.model_name in model_dims:
            return model_dims[self.model_name]

        # Fallback to model config if available
        if self._model is not None:
            return self._model.config.hidden_size

        # Final fallback to settings
        return getattr(settings, 'EMBEDDING_DIMENSION', 2048)

    def encode_texts(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Encode texts using Polyglot-Ko model by extracting embeddings from hidden states.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for processing (default from POLYGLOT_BATCH_SIZE setting)
            **kwargs: Additional parameters

        Returns:
            numpy array of embeddings
        """
        if batch_size is None:
            batch_size = self.batch_size
        if not texts:
            dtype = getattr(np, getattr(settings, 'EMBEDDING_DTYPE', 'float32'))
            return np.zeros((0, self.dimension), dtype=dtype)

        device = self._get_device()
        all_embs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Add EOS token to each text for better representation
                assert self._tokenizer is not None, "Tokenizer not loaded"
                batch_with_eos = [text + self._tokenizer.eos_token for text in batch]

                encoded = self._tokenizer(
                    batch_with_eos,
                    padding=True,
                    truncation=True,
                    max_length=getattr(settings, 'EMBEDDING_MAX_LENGTH', 512),
                    return_tensors="pt"
                )

                for k, v in encoded.items():
                    encoded[k] = v.to(device)

                assert self._model is not None, "Model not loaded"
                with self._lock:
                    # Get hidden states from the model
                    outputs = self._model(**encoded, output_hidden_states=True)
                    # Use the second last layer's hidden states for better embeddings
                    last_hidden = outputs.hidden_states[-2]

                # Mean pool across sequence dimension
                emb = self._mean_pool(last_hidden, encoded["attention_mask"])
                emb = emb.cpu().numpy()

                # Handle NaN and inf values before normalization
                emb = np.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=-1.0)

                # L2 normalize
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.nan_to_num(norms, nan=1.0, posinf=1.0, neginf=1.0)  # Handle NaN norms
                norms[norms == 0] = 1.0  # Avoid division by zero
                emb = emb / norms

                # Final safety check - replace any remaining NaN/inf values
                emb = np.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=-1.0)

                dtype = getattr(np, getattr(settings, 'EMBEDDING_DTYPE', 'float32'))
                all_embs.append(emb.astype(dtype))

        return np.vstack(all_embs)

    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """
        Encode a single query.

        Args:
            query: Query string to encode
            **kwargs: Additional parameters

        Returns:
            numpy array of shape (dimension,)
        """
        result = self.encode_texts([query], **kwargs)
        return result[0]