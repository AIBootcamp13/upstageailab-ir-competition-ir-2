"""Application settings (pydantic BaseSettings).

This package contains the concrete Settings implementation that was
previously located in the top-level `ir_core.config` module. Importing
``ir_core.config.settings`` will return a configured ``Settings``
instance (reads from environment or .env file).
"""
from pydantic import BaseSettings


class Settings(BaseSettings):
	"""Application settings (overridable from environment).

	Examples: ES_HOST, INDEX_NAME, EMBEDDING_MODEL, etc.
	"""

	ES_HOST: str = "http://localhost:9200"
	INDEX_NAME: str = "test"
	EMBEDDING_MODEL: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
	BM25_K: int = 200
	RERANK_K: int = 10
	ALPHA: float = 0.0  # interpolation weight: 0 => cosine-only, 1 => bm25-only
	REDIS_URL: str = "redis://localhost:6379/0"
	USE_WANDB: bool = False
	WANDB_PROJECT: str = "ir-rag"

	class Config:
		env_file = ".env"


settings = Settings()


__all__ = ["Settings", "settings"]
