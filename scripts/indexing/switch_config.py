#!/usr/bin/env python3
"""
Configuration Switcher for Korean/English RAG Setup

This script reads profiles from conf/embedding_profiles.yaml and applies them.
Usage: python switch_config.py [profile_name|show|list]
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf
from ruamel.yaml import YAML

# Initialize ruamel.yaml with formatting preservation
yaml_handler = YAML()
yaml_handler.preserve_quotes = True
yaml_handler.indent(mapping=2, sequence=4, offset=2)

def _add_src_to_path():
    """Add src directory to sys.path for imports."""
    scripts_dir = Path(__file__).parent
    repo_dir = scripts_dir.parent
    src_dir = repo_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# Add src to path for imports
_add_src_to_path()

# Import centralized path utilities
from ir_core.utils.path_utils import (
    get_project_root, get_conf_path, get_settings_path,
    get_embedding_profiles_path, get_data_config_path
)

# Legacy constants for backward compatibility (now use path_utils)
PROJECT_ROOT = get_project_root()
PROFILES_PATH = get_embedding_profiles_path()
SETTINGS_PATH = get_settings_path()
DATA_CONFIG_DIR = get_conf_path() / "data"

def load_profiles() -> Dict[str, Any]:
    """Load all profiles from embedding_profiles.yaml."""
    if not PROFILES_PATH.exists():
        print(f"❌ embedding_profiles.yaml not found at {PROFILES_PATH}")
        return {}
    with open(PROFILES_PATH, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data if data else {}

def load_settings() -> Dict[str, Any]:
    """Load current settings from settings.yaml"""
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Use OmegaConf to resolve environment variables
    try:
        OmegaConf.register_new_resolver("env", os.getenv)
    except ValueError:
        # Resolver already registered, continue
        pass
    config = OmegaConf.create(yaml.safe_load(content))
    return OmegaConf.to_container(config, resolve=True)  # type: ignore

def load_settings_preserve_format():
    """Load settings while preserving YAML structure and comments"""
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        return yaml_handler.load(f)

def load_model_config() -> Dict[str, Any]:
    """Load current model configuration from consolidated settings.yaml"""
    settings = load_settings()
    # Model config is now part of the main settings
    return settings.get('model', {})

def save_model_config(config: Dict[str, Any]) -> None:
    """Save model configuration to consolidated settings.yaml"""
    settings = load_settings()
    settings['model'] = config
    save_settings(settings)

def save_settings(settings: Dict[str, Any]) -> None:
    """Save settings to settings.yaml while preserving formatting and comments"""
    # Load the current file with ruamel.yaml to preserve structure
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        current_data = yaml_handler.load(f)

    # Update only the specific settings that need to change
    _update_nested_dict(current_data, settings)

    # Write back with preserved formatting
    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        yaml_handler.dump(current_data, f)

def load_data_config() -> Dict[str, Any]:
    """Load current data configuration based on defaults in settings.yaml"""
    settings = load_settings_preserve_format()

    # Find the data config from defaults
    data_config_name = None
    if 'defaults' in settings:
        for default_entry in settings['defaults']:
            if isinstance(default_entry, str) and default_entry.startswith('data:'):
                data_config_name = default_entry.split(':', 1)[1].strip()
                break
            elif isinstance(default_entry, dict) and 'data' in default_entry:
                data_config_name = default_entry['data']
                break

    if not data_config_name:
        # Fallback to science_qa_ko if not found
        data_config_name = 'science_qa_ko'

    # Load the data config file
    data_config_file = DATA_CONFIG_DIR / f"{data_config_name}.yaml"
    if data_config_file.exists():
        with open(data_config_file, 'r', encoding='utf-8') as f:
            return yaml_handler.load(f)
    else:
        # Fallback configuration
        return {
            'documents_path': 'data/documents_ko.jsonl',
            'validation_path': 'data/validation_balanced.jsonl',
            'evaluation_path': 'data/eval.jsonl',
            'output_path': 'outputs/submission_ko.jsonl'
        }

def get_current_documents_path() -> str:
    """Get the current documents path from the active data configuration"""
    data_config = load_data_config()
    return data_config.get('documents_path', 'data/documents_ko.jsonl')

def switch_to_profile(profile_name: str) -> None:
    """Switch configuration to a named profile."""
    profiles = load_profiles()
    if profile_name not in profiles:
        print(f"❌ Error: Profile '{profile_name}' not found in {PROFILES_PATH}")
        print("Available profiles:", ", ".join(profiles.keys()))
        return

    profile = profiles[profile_name]
    updates = profile.get("config", {})
    data_config = profile.get("data_config")

    print(f"🔄 Switching to '{profile_name}' configuration...")

    # Load settings.yaml while preserving structure
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        current_data = yaml_handler.load(f)

    # Apply updates from the profile
    _update_nested_dict(current_data, updates)

    # Save updated settings.yaml
    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        yaml_handler.dump(current_data, f)

    # Update data configuration if specified
    if data_config:
        update_data_config(data_config)

    print(f"✅ Switched to '{profile_name}' configuration.")
    print(f"   Description: {profile.get('description', 'N/A')}")

    # Special warnings for specific providers
    if profile_name == "solar":
        api_key_set = bool(os.getenv('UPSTAGE_API_KEY'))
        if not api_key_set:
            print("\n⚠️  Warning: UPSTAGE_API_KEY environment variable not set!")
            print("   Please set it in your .env file or environment before using Solar embeddings.")

    # Show key configuration details
    config = profile.get("config", {})
    print(f"   - Provider: {config.get('EMBEDDING_PROVIDER', 'N/A')}")
    print(f"   - Model: {config.get('EMBEDDING_MODEL', 'N/A')}")
    print(f"   - Dimension: {config.get('EMBEDDING_DIMENSION', 'N/A')}")
    print(f"   - Index: {config.get('INDEX_NAME', 'N/A')}")

def list_profiles() -> None:
    """List all available profiles."""
    profiles = load_profiles()
    if not profiles:
        print("❌ No profiles found or profiles file is missing.")
        return

    print("📋 Available Configuration Profiles:")
    for name, data in profiles.items():
        print(f"  - {name}: {data.get('description', 'No description')}")

def _update_nested_dict(base_dict, updates):
    """Recursively update nested dictionary with new values"""
    for key, value in updates.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _update_nested_dict(base_dict[key], value)
        else:
            base_dict[key] = value

def update_data_config(language):
    """Update the data configuration in consolidated settings.yaml while preserving formatting"""
    # Define the mapping of language to data config name
    data_configs = {
        "ko": "science_qa_ko",
        "bilingual": "science_qa_bilingual",
        "en": "science_qa"
    }

    if language not in data_configs:
        print(f"❌ Unknown language: {language}")
        return

    target_config = data_configs[language]

    # Load current settings with preserved format
    current_data = load_settings_preserve_format()

    # Update the defaults section
    if 'defaults' in current_data:
        for i, default_entry in enumerate(current_data['defaults']):
            if isinstance(default_entry, str) and default_entry.startswith('data:'):
                current_data['defaults'][i] = f"data: {target_config}"
                break
            elif isinstance(default_entry, dict) and 'data' in default_entry:
                current_data['defaults'][i]['data'] = target_config
                break

    # Save with preserved formatting
    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        yaml_handler.dump(current_data, f)

def show_current_config():
    """Show current configuration"""
    settings = load_settings()
    model_config = load_model_config()

    print("📋 Current Configuration:")
    print(f"   Embedding Provider: {settings.get('EMBEDDING_PROVIDER', 'huggingface')}")
    print(f"   Embedding Model: {settings.get('EMBEDDING_MODEL', 'Not set')}")
    print(f"   Embedding Dimension: {settings.get('EMBEDDING_DIMENSION', 'Not set')}")
    print(f"   Index Name: {settings.get('INDEX_NAME', 'Not set')}")
    print(f"   Translation Enabled: {settings.get('translation', {}).get('enabled', 'Not set')}")
    print(f"   Alpha (BM25/Dense balance): {settings.get('ALPHA', 'Not set')}")
    print(f"   BM25 K: {settings.get('BM25_K', 'Not set')}")
    print(f"   Rerank K: {settings.get('RERANK_K', 'Not set')}")

    # Show data configuration
    defaults = settings.get('defaults', [])
    data_config = "Not set"
    for default_entry in defaults:
        if isinstance(default_entry, str) and 'data:' in default_entry:
            data_config = default_entry.split(':', 1)[1].strip()
            break
        elif isinstance(default_entry, dict) and 'data' in default_entry:
            data_config = default_entry['data']
            break

    print(f"   Data Configuration: {data_config}")

    # Check for API keys
    if settings.get('EMBEDDING_PROVIDER') == 'solar':
        api_key_set = bool(os.getenv('UPSTAGE_API_KEY'))
        print(f"   Solar API Key Set: {'✅ Yes' if api_key_set else '❌ No'}")
    elif settings.get('EMBEDDING_PROVIDER') == 'polyglot':
        polyglot_model = settings.get('POLYGLOT_MODEL', 'Not set')
        polyglot_quant = settings.get('POLYGLOT_QUANTIZATION', 'Not set')
        polyglot_batch = settings.get('POLYGLOT_BATCH_SIZE', 'Not set')
        polyglot_threads = settings.get('POLYGLOT_MAX_THREADS', 'Not set')
        print(f"   Polyglot Model: {polyglot_model}")
        print(f"   Polyglot Quantization: {polyglot_quant}")
        print(f"   Polyglot Batch Size: {polyglot_batch}")
        print(f"   Polyglot Max Threads: {polyglot_threads}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python switch_config.py [profile_name|show|list]")
        print("  profile_name - Switch to a specific profile (e.g., korean, english, bilingual)")
        print("  show         - Show current configuration")
        print("  list         - List all available profiles")
        return

    command = sys.argv[1].lower()

    if command == "show":
        show_current_config()
    elif command == "list":
        list_profiles()
    else:
        # Assume it's a profile name
        switch_to_profile(command)

if __name__ == "__main__":
    main()
