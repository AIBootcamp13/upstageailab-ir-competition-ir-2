# src/ir_core/embeddings/base.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseEmbeddingProvider(ABC):
    """
    Abstract Base Class for an embedding provider.
    """

    @abstractmethod
    def encode_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode
            **kwargs: Additional provider-specific parameters

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """
        Encode a single query into an embedding.

        Args:
            query: Query string to encode
            **kwargs: Additional provider-specific parameters

        Returns:
            numpy array of shape (embedding_dim,)
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the embedding dimension for this provider.

        Returns:
            int: Embedding dimension
        """
        pass