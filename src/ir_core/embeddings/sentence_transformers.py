# src/ir_core/embeddings/sentence_transformers.py
from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseEmbeddingProvider
from ..config import settings

class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """
    SentenceTransformer embedding provider.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize SentenceTransformer embedding provider.

        Args:
            model_name: SentenceTransformer model name. If None, uses settings.EMBEDDING_MODEL.
        """
        self.model_name = model_name or getattr(settings, 'EMBEDDING_MODEL', 'jhgan/ko-sroberta-multitask')
        self._model = None
        self._cached_dimension = None
        self._load_model()

    def _load_model(self):
        """Load the model."""
        if self._model is None:
            device_str = getattr(settings, 'EMBEDDING_DEVICE', 'auto')
            device = 'cuda' if device_str == 'auto' and torch.cuda.is_available() else device_str
            self._model = SentenceTransformer(self.model_name, device=device)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._model is None:
            self._load_model()
        if self._cached_dimension is None:
            if self._model is None:
                # Use configured embedding dimension as fallback
                self._cached_dimension = getattr(settings, 'EMBEDDING_DIMENSION', 768)
            else:
                try:
                    # Use dummy encoding to get dimension
                    dummy_embedding = self._model.encode(['test'])
                    self._cached_dimension = dummy_embedding.shape[1]
                except Exception:
                    # Fallback to configured dimension if encoding fails
                    self._cached_dimension = getattr(settings, 'EMBEDDING_DIMENSION', 768)
        return self._cached_dimension

    def encode_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode texts using SentenceTransformer.

        Args:
            texts: List of text strings to encode
            **kwargs: Additional parameters

        Returns:
            numpy array of embeddings
        """
        if self._model is None:
            self._load_model()
        assert self._model is not None, "Model should be loaded at this point"
        return self._model.encode(texts, **kwargs)

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