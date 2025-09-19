# src/ir_core/embeddings/solar.py
import os
import requests
from typing import List, Optional
import numpy as np
from .base import BaseEmbeddingProvider
from ..config import settings

class SolarEmbeddingProvider(BaseEmbeddingProvider):
    """
    Upstage Solar API embedding provider.
    Generates 4096-dimensional embeddings using Upstage's Solar API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Solar embedding provider.

        Args:
            api_key: Upstage API key. If None, uses UPSTAGE_API_KEY environment variable.
            base_url: Base URL for Solar API. Defaults to Upstage's endpoint.
        """
        self.api_key = api_key or os.getenv('UPSTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY environment variable must be set")

        self.base_url = base_url or "https://api.upstage.ai/v1/solar"
        self.embedding_url = f"{self.base_url}/embeddings"

        # Solar API uses 4096 dimensions
        self._dimension = 4096

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode texts using Solar API.

        Args:
            texts: List of text strings to encode
            **kwargs: Additional parameters (model, etc.)

        Returns:
            numpy array of shape (len(texts), 4096)
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # Prepare request payload
        payload = {
            "model": kwargs.get("model", "solar-embedding-1-large-passage"),
            "input": texts
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.embedding_url,
                json=payload,
                headers=headers,
                timeout=30  # 30 second timeout
            )
            response.raise_for_status()

            data = response.json()

            # Extract embeddings from response
            # Solar API returns {"data": [{"embedding": [...], "index": 0}, ...]}
            embeddings = []
            for item in data["data"]:
                embeddings.append(item["embedding"])

            return np.array(embeddings, dtype=np.float32)

        except requests.RequestException as e:
            raise RuntimeError(f"Solar API request failed: {e}")
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Invalid Solar API response format: {e}")

    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """
        Encode a single query using Solar API.

        Args:
            query: Query string to encode
            **kwargs: Additional parameters

        Returns:
            numpy array of shape (4096,)
        """
        # Use query model for queries
        kwargs.setdefault("model", "solar-embedding-1-large-query")
        result = self.encode_texts([query], **kwargs)
        return result[0] if len(result) > 0 else np.zeros(self.dimension, dtype=np.float32)