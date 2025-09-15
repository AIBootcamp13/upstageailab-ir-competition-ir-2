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

def load_settings() -> Dict[str, Any]:
    """Load current settings from settings.yaml"""
    settings_file = Path(__file__).parent.parent / "conf" / "settings.yaml"
    with open(settings_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_settings(settings: Dict[str, Any]) -> None:
    """Save settings to settings.yaml"""
    settings_file = Path(__file__).parent.parent / "conf" / "settings.yaml"
    with open(settings_file, 'w', encoding='utf-8') as f:
        yaml.dump(settings, f, default_flow_style=False, allow_unicode=True)

def switch_to_korean():
    """Switch configuration to Korean setup"""
    print("🔄 Switching to Korean configuration...")

    settings = load_settings()

    # Update embedding model
    settings['EMBEDDING_MODEL'] = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

    # Update index name (use Korean-specific index)
    settings['INDEX_NAME'] = "documents_ko_with_embeddings_new"

    # Update translation settings
    if 'translation' in settings:
        settings['translation']['enabled'] = False  # No need for translation with Korean model

    save_settings(settings)

    # Update data configuration file to use Korean data
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

    # Update embedding model
    settings['EMBEDDING_MODEL'] = "sentence-transformers/all-MiniLM-L6-v2"

    # Update index name (use English-specific index)
    settings['INDEX_NAME'] = "documents_en_with_embeddings_new"

    # Update translation settings
    if 'translation' in settings:
        settings['translation']['enabled'] = True  # Enable translation for Korean queries

    save_settings(settings)

    # Update data configuration file to use English data
    update_data_config("en")

    print("✅ Switched to English configuration")
    print("   - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384d)")
    print("   - Index: documents_en_with_embeddings_new")
    print("   - Documents: data/documents_bilingual.jsonl")
    print("   - Translation: enabled")

def switch_to_bilingual():
    """Switch configuration to Bilingual setup"""
    print("🔄 Switching to Bilingual configuration...")

    settings = load_settings()

    # Update embedding model (use Korean model for bilingual content)
    settings['EMBEDDING_MODEL'] = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

    # Update index name (use bilingual index)
    settings['INDEX_NAME'] = "documents_bilingual_with_embeddings_new"

    # Update translation settings
    if 'translation' in settings:
        settings['translation']['enabled'] = False  # Bilingual content doesn't need translation

    save_settings(settings)

    # Update data configuration file to use bilingual data
    update_data_config("bilingual")

    print("✅ Switched to Bilingual configuration")
    print("   - Embedding model: snunlp/KR-SBERT-V40K-klueNLI-augSTS (768d)")
    print("   - Index: documents_bilingual_with_embeddings_new")
    print("   - Documents: data/documents_bilingual.jsonl")
    print("   - Translation: disabled")

def update_data_config(language):
    """Update the data configuration file - only change the data field while preserving file structure"""
    config_file = Path(__file__).parent.parent / "conf" / "config.yaml"

    # Read the entire file as text to preserve formatting and comments
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()

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

    # Use regex to find and replace only the data line in defaults section
    # This preserves all comments, formatting, and other content
    import re

    # Pattern to match: - data: <anything> (with optional quotes)
    pattern = r'(- data:\s*)(["\']?)[^\'"\s]+(["\']?)(\s*#.*)?$'

    # Replacement: - data: <target_config> (no quotes)
    replacement = r'\1' + target_config + r'\4'

    # Find the first occurrence in the defaults section
    lines = content.split('\n')
    in_defaults = False
    new_lines = []

    for line in lines:
        if line.strip() == 'defaults:':
            in_defaults = True
            new_lines.append(line)
        elif in_defaults and line.strip().startswith('- ') and not line.strip().startswith('- _self_'):
            # This is a defaults entry, check if it's the data entry
            if re.search(r'- data:', line):
                # Replace the data entry
                new_line = re.sub(pattern, replacement, line)
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        elif in_defaults and (line.strip() == '' or line.strip().startswith('#')):
            # Empty lines or comments within defaults section
            new_lines.append(line)
        elif in_defaults and line.strip().startswith('- _self_'):
            # End of defaults section
            new_lines.append(line)
            in_defaults = False
        else:
            new_lines.append(line)

    # Write back the modified content
    new_content = '\n'.join(new_lines)
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

def show_current_config():
    """Show current configuration"""
    settings = load_settings()

    print("📋 Current Configuration:")
    print(f"   Embedding Model: {settings.get('EMBEDDING_MODEL', 'Not set')}")
    print(f"   Index Name: {settings.get('INDEX_NAME', 'Not set')}")
    print(f"   Translation Enabled: {settings.get('translation', {}).get('enabled', 'Not set')}")

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
