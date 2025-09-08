"""Embeddings implementation moved into subpackage.

This file contains the concrete implementations previously in
`src/ir_core/embeddings.py`.
"""
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ..config import settings

_tokenizer = None
_model = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model(name: Optional[str] = None):
    """Load tokenizer and model.

    Args:
        name: model name (HuggingFace). Defaults to settings.EMBEDDING_MODEL.
    Returns:
        (tokenizer, model)
    """
    global _tokenizer, _model
    model_name = name or settings.EMBEDDING_MODEL
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _model = AutoModel.from_pretrained(model_name)
        _model.to(_get_device())
        _model.eval()
    return _tokenizer, _model


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts: List[str], batch_size: int = 32, device: Optional[str] = None) -> np.ndarray:
    if not texts:
        try:
            _, m = load_model()
            dim = m.config.hidden_size
        except Exception:
            dim = 768
        return np.zeros((0, dim), dtype=np.float32)

    tokenizer, model = load_model()
    dev = torch.device(device) if device else _get_device()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            for k, v in encoded.items():
                encoded[k] = v.to(dev)
            out = model(**encoded)
            last_hidden = out.last_hidden_state
            emb = _mean_pool(last_hidden, encoded["attention_mask"])  # (batch, dim)
            emb = emb.cpu().numpy()
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
            all_embs.append(emb.astype(np.float32))
    return np.vstack(all_embs)


def encode_query(text: str, **kwargs) -> np.ndarray:
    arr = encode_texts([text], **kwargs)
    return arr[0]
