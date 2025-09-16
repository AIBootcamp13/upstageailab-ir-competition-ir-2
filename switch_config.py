#!/usr/bin/env python3
"""
Configuration Switcher for Korean/English RAG Setup

This script helps switch between Korean and English configurations
by updating the embedding model and index settings.
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

def load_settings() -> Dict[str, Any]:
    """Load current settings from settings.yaml"""
    settings_file = Path(__file__).parent / "conf" / "settings.yaml"
    with open(settings_file, 'r', encoding='utf-8') as f:
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
    settings_file = Path(__file__).parent / "conf" / "settings.yaml"
    with open(settings_file, 'r', encoding='utf-8') as f:
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
    settings_file = Path(__file__).parent / "conf" / "settings.yaml"

    # Load the current file with ruamel.yaml to preserve structure
    with open(settings_file, 'r', encoding='utf-8') as f:
        current_data = yaml_handler.load(f)

    # Update only the specific settings that need to change
    _update_nested_dict(current_data, settings)

    # Write back with preserved formatting
    with open(settings_file, 'w', encoding='utf-8') as f:
        yaml_handler.dump(current_data, f)

def _update_nested_dict(base_dict, updates):
    """Recursively update nested dictionary with new values"""
    for key, value in updates.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _update_nested_dict(base_dict[key], value)
        else:
            base_dict[key] = value

def switch_to_korean():
    """Switch configuration to Korean setup"""
    print("🔄 Switching to Korean configuration...")

    # Load current settings with preserved format
    current_data = load_settings_preserve_format()

    # Update only the specific settings that need to change
    updates = {
        'EMBEDDING_MODEL': "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        'EMBEDDING_DIMENSION': 768,
        'INDEX_NAME': "documents_ko_with_embeddings_new",
        'model': {
            'embedding_model': "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
            'alpha': 0.4,
            'bm25_k': 200,
            'rerank_k': 10
        },
        'translation': {
            'enabled': False
        }
    }

    # Apply updates while preserving structure
    _update_nested_dict(current_data, updates)

    # Save with preserved formatting
    settings_file = Path(__file__).parent / "conf" / "settings.yaml"
    with open(settings_file, 'w', encoding='utf-8') as f:
        yaml_handler.dump(current_data, f)

    # Update data configuration to use Korean data
    update_data_config("ko")

    print("✅ Switched to Korean configuration")
    print("   - Embedding model: snunlp/KR-SBERT-V40K-klueNLI-augSTS (768d)")
    print("   - Index: documents_ko_with_embeddings_new")
    print("   - Documents: data/documents_ko.jsonl")
    print("   - Translation: disabled")
    print("\n⚠️  Make sure the Korean index exists or create it with the Korean embedding model!")

def switch_to_english():
    """Switch configuration to English setup"""
    print("🔄 Switching to English configuration...")

    # Load current settings with preserved format
    current_data = load_settings_preserve_format()

    # Update only the specific settings that need to change
    updates = {
        'EMBEDDING_MODEL': "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        'EMBEDDING_DIMENSION': 768,
        'INDEX_NAME': "documents_en_with_embeddings_new",
        'model': {
            'embedding_model': "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
            'alpha': 0.4,
            'bm25_k': 200,
            'rerank_k': 10
        },
        'translation': {
            'enabled': True
        }
    }

    # Apply updates while preserving structure
    _update_nested_dict(current_data, updates)

    # Save with preserved formatting
    settings_file = Path(__file__).parent / "conf" / "settings.yaml"
    with open(settings_file, 'w', encoding='utf-8') as f:
        yaml_handler.dump(current_data, f)

    # Update data configuration to use English data
    update_data_config("en")

    print("✅ Switched to English configuration")
    print("   - Embedding model: snunlp/KR-SBERT-V40K-klueNLI-augSTS (768d)")
    print("   - Index: documents_en_with_embeddings_new")
    print("   - Documents: data/documents_bilingual.jsonl")
    print("   - Translation: enabled")

def switch_to_bilingual():
    """Switch configuration to Bilingual setup"""
    print("🔄 Switching to Bilingual configuration...")

    # Load current settings with preserved format
    current_data = load_settings_preserve_format()

    # Update only the specific settings that need to change
    updates = {
        'EMBEDDING_MODEL': "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        'EMBEDDING_DIMENSION': 768,
        'INDEX_NAME': "documents_bilingual_with_embeddings_new",
        'model': {
            'embedding_model': "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
            'alpha': 0.4,
            'bm25_k': 200,
            'rerank_k': 10
        },
        'translation': {
            'enabled': False
        }
    }

    # Apply updates while preserving structure
    _update_nested_dict(current_data, updates)

    # Save with preserved formatting
    settings_file = Path(__file__).parent / "conf" / "settings.yaml"
    with open(settings_file, 'w', encoding='utf-8') as f:
        yaml_handler.dump(current_data, f)

    # Update data configuration to use bilingual data
    update_data_config("bilingual")

    print("✅ Switched to Bilingual configuration")
    print("   - Embedding model: snunlp/KR-SBERT-V40K-klueNLI-augSTS (768d)")
    print("   - Index: documents_bilingual_with_embeddings_new")
    print("   - Documents: data/documents_bilingual.jsonl")
    print("   - Translation: disabled")

def switch_to_solar():
    """Switch configuration to Solar API setup (4096D embeddings)"""
    print("🔄 Switching to Solar API configuration...")

    # Check if UPSTAGE_API_KEY is set
    if not os.getenv('UPSTAGE_API_KEY'):
        print("⚠️  Warning: UPSTAGE_API_KEY environment variable not set!")
        print("   Please set it in your .env file or environment before using Solar embeddings.")

    # Load current settings with preserved format
    current_data = load_settings_preserve_format()

    # Update only the specific settings that need to change
    updates = {
        'EMBEDDING_PROVIDER': "solar",
        'EMBEDDING_DIMENSION': 4096,
        'INDEX_NAME': "documents_solar_with_embeddings_new",
        'model': {
            'embedding_model': "solar-embedding-1-large",
            'alpha': 0.4,
            'bm25_k': 200,
            'rerank_k': 10
        },
        'translation': {
            'enabled': False
        }
    }

    # Apply updates while preserving structure
    _update_nested_dict(current_data, updates)

    # Save with preserved formatting
    settings_file = Path(__file__).parent / "conf" / "settings.yaml"
    with open(settings_file, 'w', encoding='utf-8') as f:
        yaml_handler.dump(current_data, f)

    # Update data configuration to use bilingual data (Solar can handle both languages)
    update_data_config("bilingual")

    print("✅ Switched to Solar API configuration")
    print("   - Embedding provider: solar")
    print("   - Embedding model: solar-embedding-1-large (4096d)")
    print("   - Index: documents_solar_with_embeddings_new")
    print("   - Documents: data/documents_bilingual.jsonl")
    print("   - Translation: disabled")
    print("\n⚠️  Make sure UPSTAGE_API_KEY is set in your environment!")
    print("⚠️  Make sure the Solar index exists or create it with Solar embeddings!")

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
    settings_file = Path(__file__).parent / "conf" / "settings.yaml"
    with open(settings_file, 'w', encoding='utf-8') as f:
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python switch_config.py [korean|english|bilingual|solar|show]")
        print("  korean   - Switch to Korean configuration")
        print("  english  - Switch to English configuration")
        print("  bilingual- Switch to Bilingual configuration")
        print("  solar    - Switch to Solar API configuration (4096D embeddings)")
        print("  show     - Show current configuration")
        return

    command = sys.argv[1].lower()

    if command == "korean":
        switch_to_korean()
    elif command == "english":
        switch_to_english()
    elif command == "bilingual":
        switch_to_bilingual()
    elif command == "solar":
        switch_to_solar()
    elif command == "show":
        show_current_config()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
