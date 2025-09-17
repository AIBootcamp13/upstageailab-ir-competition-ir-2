# src/ir_core/embeddings/core.py
from typing import List, Optional
import numpy as np

from ..config import settings
from .base import BaseEmbeddingProvider
from .huggingface import HuggingFaceEmbeddingProvider
from .solar import SolarEmbeddingProvider
from .polyglot import PolyglotKoEmbeddingProvider

# Global provider instance
_provider = None
_current_provider_type = None

def get_embedding_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
    """
    Get or create embedding provider based on type.

    Args:
        provider_type: Type of provider ('huggingface' or 'solar'). If None, uses settings.

    Returns:
        BaseEmbeddingProvider instance
    """
    global _provider, _current_provider_type

    # Determine provider type
    if provider_type is None:
        provider_type = getattr(settings, 'EMBEDDING_PROVIDER', 'huggingface')

    # Create new provider if type changed
    if _current_provider_type != provider_type or _provider is None:
        if provider_type == 'huggingface':
            _provider = HuggingFaceEmbeddingProvider()
        elif provider_type == 'solar':
            _provider = SolarEmbeddingProvider()
        elif provider_type == 'polyglot':
            _provider = PolyglotKoEmbeddingProvider()
        else:
            raise ValueError(f"Unknown embedding provider type: {provider_type}")

        _current_provider_type = provider_type

    return _provider

def load_model(name: Optional[str] = None):
    """
    Load model for backward compatibility.
    This function is kept for backward compatibility but now delegates to provider.
    """
    provider = get_embedding_provider()
    if isinstance(provider, HuggingFaceEmbeddingProvider):
        # For HuggingFace, we still need to load the model
        return provider._tokenizer, provider._model
    else:
        # For API providers, return None
        return None, None


def encode_texts(texts: List[str], batch_size: int = 32, device: Optional[str] = None, model_name: Optional[str] = None, provider_type: Optional[str] = None) -> np.ndarray:
    """
    Encode texts into embeddings using the configured provider.

    Args:
        texts: List of text strings to encode
        batch_size: Batch size (for HuggingFace provider)
        device: Device for computation (for HuggingFace provider)
        model_name: Model name (for HuggingFace provider)
        provider_type: Force specific provider type

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    provider = get_embedding_provider(provider_type)

    # Pass provider-specific kwargs
    kwargs = {}
    if isinstance(provider, HuggingFaceEmbeddingProvider):
        kwargs.update({
            'batch_size': batch_size,
            'device': device,
            'model_name': model_name
        })

    return provider.encode_texts(texts, **kwargs)


def encode_query(text: str, **kwargs) -> np.ndarray:
    """
    Encode a single query into an embedding.

    Args:
        text: Query string to encode
        **kwargs: Additional parameters passed to provider

    Returns:
        numpy array of shape (embedding_dim,)
    """
    provider = get_embedding_provider(kwargs.get('provider_type'))
    return provider.encode_query(text, **kwargs)
