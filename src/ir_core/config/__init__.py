"""Application settings (pydantic BaseSettings).

This package contains the concrete Settings implementation that was
previously located in the top-level `ir_core.config` module. Importing
``ir_core.config.settings`` will return a configured ``Settings``
instance (reads from environment or .env file).
"""
from pydantic import BaseSettings
from typing import Dict, Any

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

	# --- Phase 1: W&B 설정 추가 ---
	WANDB_API_KEY: str = ""
	WANDB_PROJECT: str = "ir-rag"

	# --- New Settings for the Generation Layer ---
	GENERATOR_TYPE: str = "openai"  # Can be "openai" or "ollama"
	GENERATOR_MODEL_NAME: str = "gpt-3.5-turbo" # e.g., "gpt-3.5-turbo", "llama3"

	# --- Phase 1: 중앙 설정 추가 ---
	PIPELINE_TOOL_CALLING_MODEL: str = "gpt-3.5-turbo-1106"
	PIPELINE_REWRITER_MODEL: str = "gpt-4o-mini"
	PIPELINE_DEFAULT_TOP_K: int = 5

	PROMPT_GENERATION_QA: str = "prompts/scientific_qa_v1.jinja2"
	PROMPT_PERSONA: str = "prompts/persona_qa.txt"
	PROMPT_REPHRASE_QUERY: str = "prompts/rephrase_query_v1.jinja2"

	# --- 기존 프롬프트 설정 (하위 호환성을 위해 유지) ---
	PROMPT_TEMPLATE_PATH: str = "prompts/scientific_qa_v1.jinja2"
	GENERATOR_SYSTEM_MESSAGE_FILE: str = "prompts/persona_qa.txt"
	GENERATOR_SYSTEM_MESSAGE: str = ""


	class Config(BaseSettings.Config):
		env_file = ".env"


settings = Settings()


__all__ = ["Settings", "settings"]
