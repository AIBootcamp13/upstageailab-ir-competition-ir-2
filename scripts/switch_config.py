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

def load_settings() -> Dict[str, Any]:
    """Load current settings from settings.yaml"""
    settings_file = Path(__file__).parent.parent / "conf" / "settings.yaml"
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
    """Save settings to settings.yaml"""
    settings_file = Path(__file__).parent.parent / "conf" / "settings.yaml"
    with open(settings_file, 'w', encoding='utf-8') as f:
        yaml.dump(settings, f, default_flow_style=False, allow_unicode=True)

def switch_to_korean():
    """Switch configuration to Korean setup"""
    print("🔄 Switching to Korean configuration...")

    settings = load_settings()

    # Update embedding model in settings
    settings['EMBEDDING_MODEL'] = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    settings['EMBEDDING_DIMENSION'] = 768

    # Update model configuration
    if 'model' not in settings:
        settings['model'] = {}
    settings['model']['embedding_model'] = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    settings['model']['alpha'] = 0.4
    settings['model']['bm25_k'] = 200
    settings['model']['rerank_k'] = 10

    # Update index name (use Korean-specific index)
    settings['INDEX_NAME'] = "documents_ko_with_embeddings_new"

    # Update translation settings
    if 'translation' in settings:
        settings['translation']['enabled'] = False  # No need for translation with Korean model

    save_settings(settings)

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

    settings = load_settings()

    # Update embedding model in settings (using Korean model for English content)
    settings['EMBEDDING_MODEL'] = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    settings['EMBEDDING_DIMENSION'] = 768

    # Update model configuration
    if 'model' not in settings:
        settings['model'] = {}
    settings['model']['embedding_model'] = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    settings['model']['alpha'] = 0.4
    settings['model']['bm25_k'] = 200
    settings['model']['rerank_k'] = 10

    # Update index name (use English-specific index)
    settings['INDEX_NAME'] = "documents_en_with_embeddings_new"

    # Update translation settings
    if 'translation' in settings:
        settings['translation']['enabled'] = True  # Enable translation for Korean queries

    save_settings(settings)

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

    settings = load_settings()

    # Update embedding model (use Korean model for bilingual content)
    settings['EMBEDDING_MODEL'] = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    settings['EMBEDDING_DIMENSION'] = 768

    # Update model configuration
    if 'model' not in settings:
        settings['model'] = {}
    settings['model']['embedding_model'] = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    settings['model']['alpha'] = 0.4
    settings['model']['bm25_k'] = 200
    settings['model']['rerank_k'] = 10

    # Update index name (use bilingual index)
    settings['INDEX_NAME'] = "documents_bilingual_with_embeddings_new"

    # Update translation settings
    if 'translation' in settings:
        settings['translation']['enabled'] = False  # Bilingual content doesn't need translation

    save_settings(settings)

    # Update data configuration to use bilingual data
    update_data_config("bilingual")

    print("✅ Switched to Bilingual configuration")
    print("   - Embedding model: snunlp/KR-SBERT-V40K-klueNLI-augSTS (768d)")
    print("   - Index: documents_bilingual_with_embeddings_new")
    print("   - Documents: data/documents_bilingual.jsonl")
    print("   - Translation: disabled")

def update_data_config(language):
    """Update the data configuration in consolidated settings.yaml"""
    settings = load_settings()

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

    # Update the defaults section in settings
    if 'defaults' in settings:
        for i, default_entry in enumerate(settings['defaults']):
            if isinstance(default_entry, str) and default_entry.startswith('data:'):
                settings['defaults'][i] = f"data: {target_config}"
                break
            elif isinstance(default_entry, dict) and 'data' in default_entry:
                settings['defaults'][i]['data'] = target_config
                break

    save_settings(settings)

def show_current_config():
    """Show current configuration"""
    settings = load_settings()
    model_config = load_model_config()

    print("📋 Current Configuration:")
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python switch_config.py [korean|english|bilingual|show]")
        print("  korean   - Switch to Korean configuration")
        print("  english  - Switch to English configuration")
        print("  bilingual- Switch to Bilingual configuration")
        print("  show     - Show current configuration")
        return

    command = sys.argv[1].lower()

    if command == "korean":
        switch_to_korean()
    elif command == "english":
        switch_to_english()
    elif command == "bilingual":
        switch_to_bilingual()
    elif command == "show":
        show_current_config()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
