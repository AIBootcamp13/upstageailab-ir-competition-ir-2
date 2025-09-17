#!/usr/bin/env python3
"""
Configuration Switcher for RAG System Data Sources

This script helps switch between different data configurations easily.
"""
import sys
from pathlib import Path
import yaml

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def switch_data_config(config_name: str):
    """
    Switch the data configuration in settings.yaml

    Args:
        config_name: Name of the data config file (without .yaml extension)
    """
    settings_path = Path(__file__).parent / "conf" / "settings.yaml"
    data_config_path = Path(__file__).parent / "conf" / "data" / f"{config_name}.yaml"

    if not data_config_path.exists():
        print(f"❌ Data configuration '{config_name}.yaml' not found in conf/data/")
        list_available_configs()
        return

    # Read current settings
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)

    # Update the defaults section
    if 'defaults' in settings:
        for i, default in enumerate(settings['defaults']):
            if default.startswith('data:'):
                settings['defaults'][i] = f'data: {config_name}'
                break

    # Write back to settings.yaml
    with open(settings_path, 'w', encoding='utf-8') as f:
        yaml.dump(settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"✅ Successfully switched to data configuration: {config_name}")
    print(f"📁 Documents path: {get_config_value(data_config_path, 'documents_path')}")
    print(f"📁 Validation path: {get_config_value(data_config_path, 'validation_path')}")
    print(f"📁 Output path: {get_config_value(data_config_path, 'output_path')}")

def get_config_value(config_path: Path, key: str) -> str:
    """Get a value from a YAML config file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get(key, 'Not set')

def list_available_configs():
    """List all available data configurations"""
    data_dir = Path(__file__).parent / "conf" / "data"
    configs = [f.stem for f in data_dir.glob("*.yaml")]
    print("\n📋 Available data configurations:")
    for config in configs:
        config_path = data_dir / f"{config}.yaml"
        docs_path = get_config_value(config_path, 'documents_path')
        print(f"  • {config}: {docs_path}")

def show_current_config():
    """Show the current data configuration"""
    settings_path = Path(__file__).parent / "conf" / "settings.yaml"

    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)

    current_config = "unknown"
    if 'defaults' in settings:
        for default in settings['defaults']:
            if isinstance(default, str) and default.startswith('data:'):
                current_config = default.split(':', 1)[1].strip()
                break
            elif isinstance(default, dict) and 'data' in default:
                current_config = default['data']
                break

    print(f"🔧 Current data configuration: {current_config}")

    # Show the actual paths being used
    data_config_path = Path(__file__).parent / "conf" / "data" / f"{current_config}.yaml"
    if data_config_path.exists():
        print(f"📁 Documents: {get_config_value(data_config_path, 'documents_path')}")
        print(f"📁 Validation: {get_config_value(data_config_path, 'validation_path')}")
        print(f"📁 Output: {get_config_value(data_config_path, 'output_path')}")
    else:
        print("⚠️  Configuration file not found")

if __name__ == "__main__":
    import fire
    print("🔄 RAG Data Configuration Switcher")
    print("=" * 40)
    fire.Fire({
        'switch': switch_data_config,
        'list': list_available_configs,
        'current': show_current_config,
    })