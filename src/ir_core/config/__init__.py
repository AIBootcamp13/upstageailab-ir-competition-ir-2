"""Application settings (pydantic BaseSettings).

This package contains the concrete Settings implementation that was
previously located in the top-level `ir_core.config` module. Importing
``ir_core.config.settings`` will return a configured ``Settings``
instance (reads from environment or .env file).
"""
import os
from pathlib import Path
from typing import cast, Dict, Any
from omegaconf import OmegaConf
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import InitSettingsSource


class Settings(BaseSettings):
	"""Application settings (overridable from environment).

	Examples: ES_HOST, INDEX_NAME, EMBEDDING_MODEL, etc.
	"""

	ES_HOST: str = ""
	INDEX_NAME: str = ""
	EMBEDDING_MODEL: str = ""
	BM25_K: int = 0
	RERANK_K: int = 0
	ALPHA: float = 0.0
	REDIS_URL: str = ""
	USE_WANDB: bool = False
	WANDB_PROJECT: str = ""

	# Index / orchestrator defaults
	INDEX_ALIAS: str = ""
	INDEX_NAME_PREFIX: str = ""
	REINDEX_BATCH_SIZE: int = 500
	KEEP_OLD_INDEX_DAYS: int = 3

	# Retrieval tuning from profiling
	USE_SRC_BOOSTS: bool = False  # enables boosted sparse retrieval using keywords_per_src.json
	USE_STOPWORD_FILTERING: bool = False  # strip global stopwords from queries 
	USE_DUPLICATE_FILTERING: bool = False  # filter exact duplicates using duplicates.json
	USE_NEAR_DUP_PENALTY: bool = False  # penalize near-duplicates using near_duplicates.json
	PROFILE_REPORT_DIR: str = ""

	# --- New Settings for the Generation Layer ---
	GENERATOR_TYPE: str = ""
	GENERATOR_MODEL_NAME: str = ""
	PROMPT_TEMPLATE_PATH: str = ""
	GENERATOR_SYSTEM_MESSAGE_FILE: str = ""
	GENERATOR_SYSTEM_MESSAGE: str = ""

	model_config = SettingsConfigDict(env_file=".env", extra='allow')

	@classmethod
	def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
		"""Load defaults from YAML file."""
		project_root = Path(__file__).parent.parent.parent.parent  # Adjust path to project root
		settings_file = project_root / "conf" / "settings.yaml"
		if settings_file.exists():
			yaml_data = cast(Dict[str, Any], OmegaConf.to_container(OmegaConf.load(settings_file), resolve=True))
			yaml_source = InitSettingsSource(settings_cls, yaml_data)
			return (yaml_source, env_settings, dotenv_settings, file_secret_settings)
		return (env_settings, dotenv_settings, file_secret_settings)


settings = Settings()


__all__ = ["Settings", "settings"]
