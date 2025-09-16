# src/ir_core/embeddings/huggingface.py
from typing import List, Optional
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import threading

from .base import BaseEmbeddingProvider
from ..config import settings

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """
    HuggingFace embedding provider using transformers.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize HuggingFace embedding provider.

        Args:
            model_name: HuggingFace model name. If None, uses settings.EMBEDDING_MODEL.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._tokenizer = None
        self._model = None
        self._device = None
        self._lock = threading.Lock()
        self._load_model()

    def _get_device(self):
        if self._device is None:
            device_str = settings.EMBEDDING_DEVICE
            if device_str == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(device_str)
        return self._device

    def _load_model(self):
        """Load tokenizer and model with thread safety."""
        with self._lock:
            if self._tokenizer is None or self._model is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=settings.EMBEDDING_USE_FAST_TOKENIZER)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.to(self._get_device())
                self._model.eval()
                # Assertions to help type checker
                assert self._tokenizer is not None
                assert self._model is not None

    def _mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=settings.EMBEDDING_NORMALIZATION_EPS)
        return summed / counts

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._model.config.hidden_size

    def encode_texts(self, texts: List[str], batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Encode texts using HuggingFace model.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for processing (default from settings)
            **kwargs: Additional parameters

        Returns:
            numpy array of embeddings
        """
        if batch_size is None:
            batch_size = int(settings.EMBEDDING_BATCH_SIZE)
        if not texts:
            dtype = getattr(np, settings.EMBEDDING_DTYPE)
            return np.zeros((0, self.dimension), dtype=dtype)

        device = self._get_device()
        all_embs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=settings.EMBEDDING_MAX_LENGTH,
                    return_tensors="pt"
                )

                for k, v in encoded.items():
                    encoded[k] = v.to(device)

                with self._lock:
                    out = self._model(**encoded)
                last_hidden = out.last_hidden_state
                emb = self._mean_pool(last_hidden, encoded["attention_mask"])
                emb = emb.cpu().numpy()
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb = emb / norms
                dtype = getattr(np, settings.EMBEDDING_DTYPE)
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