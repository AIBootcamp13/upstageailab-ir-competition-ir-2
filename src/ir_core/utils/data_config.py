"""Data configuration utilities.

This module provides utilities for loading and managing data configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf


def get_data_config_path() -> Path:
    """Get the path to the current data configuration file."""
    # Get the project root (assuming this is in src/ir_core/utils/)
    project_root = Path(__file__).parent.parent.parent.parent

    # Load main settings to determine which data config to use
    settings_file = project_root / "conf" / "settings.yaml"
    if settings_file.exists():
        with open(settings_file, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)

        # Get the data config name from defaults
        defaults = settings.get('defaults', [])
        for default in defaults:
            if isinstance(default, dict) and 'data' in default:
                data_config_name = default['data']
                data_config_file = project_root / "conf" / "data" / f"{data_config_name}.yaml"
                if data_config_file.exists():
                    return data_config_file

    # Fallback to default data config
    return project_root / "conf" / "data" / "science_qa_ko_metadata.yaml"


def load_data_config() -> Dict[str, Any]:
    """Load the current data configuration."""
    config_path = get_data_config_path()

    if not config_path.exists():
        # Fallback configuration
        return {
            'documents_path': 'data/documents_ko.jsonl',
            'validation_path': 'data/validation_balanced.jsonl',
            'evaluation_path': 'data/eval.jsonl',
            'output_path': 'outputs/submission.jsonl'
        }

    with open(config_path, 'r', encoding='utf-8') as f:
        # Use OmegaConf to resolve any environment variables
        try:
            OmegaConf.register_new_resolver("env", os.getenv)
        except ValueError:
            pass  # Resolver already registered

        config = OmegaConf.create(yaml.safe_load(f))
        return OmegaConf.to_container(config, resolve=True)  # type: ignore


def get_current_documents_path() -> str:
    """Get the current documents path from the active data configuration."""
    data_config = load_data_config()
    return data_config.get('documents_path', 'data/documents_ko.jsonl')


def get_current_validation_path() -> str:
    """Get the current validation path from the active data configuration."""
    data_config = load_data_config()
    return data_config.get('validation_path', 'data/validation_balanced.jsonl')


def get_current_evaluation_path() -> str:
    """Get the current evaluation path from the active data configuration."""
    data_config = load_data_config()
    return data_config.get('evaluation_path', 'data/eval.jsonl')


def get_current_output_path() -> str:
    """Get the current output path from the active data configuration."""
    data_config = load_data_config()
    return data_config.get('output_path', 'outputs/submission.jsonl')