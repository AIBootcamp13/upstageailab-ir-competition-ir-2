# CLI Tool Documentation (`cli.py`)

## Overview

The `cli.py` is a modern command-line interface tool built with [Typer](https://typer.tiangolo.com/) for managing the RAG (Retrieval-Augmented Generation) project's configuration and operations. It provides a user-friendly way to switch between different embedding configurations, validate settings, and manage project status.

## Features

- 🎯 **Profile-based Configuration**: Switch between predefined embedding configurations
- ✅ **Configuration Validation**: Verify current settings against known profiles
- 📊 **Project Status**: Quick overview of project state and data files
- 🚀 **Easy Setup**: Automated project initialization
- 🛡️ **Error Handling**: Comprehensive error reporting and validation
- 📖 **Auto-generated Help**: Built-in help system with command completion

## Installation & Setup

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- Project dependencies installed via `poetry install`

### Usage

All commands should be run from the project root directory:

```bash
# Set PYTHONPATH and use Poetry
PYTHONPATH=src poetry run python cli.py [COMMAND]
```

## Available Commands

### Main Commands

#### `status`
Shows a quick overview of the project status including current configuration and data file availability.

```bash
PYTHONPATH=src poetry run python cli.py status
```

**Output:**
```
🔍 Project Status:

📊 Services:

⚙️  Current Configuration:
   Provider: huggingface
   Model: snunlp/KR-SBERT-V40K-klueNLI-augSTS
   Index: documents_ko_with_embeddings_fixed

📁 Data Files:
   ✅ data/documents_ko.jsonl
   ✅ data/documents_bilingual.jsonl
   ✅ data/eval.jsonl
```

#### `setup`
Initial project setup and configuration. Can optionally switch to a specific profile after setup.

```bash
# Basic setup
PYTHONPATH=src poetry run python cli.py setup

# Setup with specific profile
PYTHONPATH=src poetry run python cli.py setup --profile korean

# Force setup even if already configured
PYTHONPATH=src poetry run python cli.py setup --force
```

**Options:**
- `--profile PROFILE`: Profile to switch to after initialization
- `--force, -f`: Force setup even if already configured

### Configuration Commands (`config` subcommand)

#### `config list`
Lists all available configuration profiles defined in `conf/embedding_profiles.yaml`.

```bash
PYTHONPATH=src poetry run python cli.py config list
```

**Output:**
```
📋 Available Configuration Profiles:
  - korean: Korean setup using KR-SBERT (768d)
  - english: English setup using all-MiniLM-L6-v2 (384d)
  - bilingual: Bilingual setup using KR-SBERT (768d)
  - solar: Solar API setup (4096d)
  - polyglot: Polyglot-Ko-3.8B setup (3072d)
  - polyglot-3b: Polyglot-Ko-3.8B setup (3072d)
  - polyglot-1b: Polyglot-Ko-1.3B setup (2048d)
```

#### `config switch <profile_name>`
Switches the active embedding configuration to a defined profile.

```bash
# Switch to Korean configuration
PYTHONPATH=src poetry run python cli.py config switch korean

# Switch to English configuration with verbose output
PYTHONPATH=src poetry run python cli.py config switch english --verbose
```

**Arguments:**
- `profile_name`: Name of the profile to switch to (required)

**Options:**
- `--verbose, -v`: Show detailed output after switching

**Example Output:**
```
🔄 Switching to 'korean' configuration...
✅ Switched to 'korean' configuration.
   Description: Korean setup using KR-SBERT (768d)
   - Provider: huggingface
   - Model: snunlp/KR-SBERT-V40K-klueNLI-augSTS
   - Dimension: 768
   - Index: documents_ko_with_embeddings_fixed
```

#### `config show`
Shows the current configuration from `settings.yaml`.

```bash
PYTHONPATH=src poetry run python cli.py config show
```

**Output:**
```
📋 Current Configuration:
   Embedding Provider: huggingface
   Embedding Model: snunlp/KR-SBERT-V40K-klueNLI-augSTS
   Embedding Dimension: 768
   Index Name: documents_ko_with_embeddings_fixed
   Translation Enabled: False
   Alpha (BM25/Dense balance): 0.4
   BM25 K: 200
   Rerank K: 10
   Data Configuration: science_qa_ko
```

#### `config validate`
Validates the current configuration for consistency and checks if it matches any defined profile.

```bash
PYTHONPATH=src poetry run python cli.py config validate
```

**Output:**
```
⚙️ Validating current configuration...
✅ Configuration structure is valid
   Current provider: huggingface
   Current model: snunlp/KR-SBERT-V40K-klueNLI-augSTS
   Current dimension: 768
   Current index: documents_ko_with_embeddings_fixed
✅ Configuration matches profile: korean
```

## Configuration Profiles

The CLI tool reads configuration profiles from `conf/embedding_profiles.yaml`. Each profile defines:

### Available Profiles

| Profile | Description | Dimensions | Provider | Data Config |
|---------|-------------|------------|----------|-------------|
| `korean` | Korean setup using KR-SBERT | 768d | huggingface | ko |
| `english` | English setup using all-MiniLM-L6-v2 | 384d | huggingface | en |
| `bilingual` | Bilingual setup using KR-SBERT | 768d | huggingface | bilingual |
| `solar` | Solar API setup | 4096d | solar | bilingual |
| `polyglot` | Polyglot-Ko-3.8B setup | 3072d | polyglot | ko |
| `polyglot-3b` | Polyglot-Ko-3.8B setup | 3072d | polyglot | ko |
| `polyglot-1b` | Polyglot-Ko-1.3B setup | 2048d | polyglot | ko |

### Profile Structure

Each profile in `embedding_profiles.yaml` contains:

```yaml
profile_name:
  description: "Human-readable description"
  config:
    EMBEDDING_PROVIDER: "provider_name"
    EMBEDDING_MODEL: "model_name"
    EMBEDDING_DIMENSION: 768
    INDEX_NAME: "elasticsearch_index_name"
    model:
      embedding_model: "model_name"
      alpha: 0.4
      bm25_k: 200
      rerank_k: 10
    translation:
      enabled: false
  data_config: "data_configuration_name"
```

## Usage Examples

### Quick Start

```bash
# Check current status
PYTHONPATH=src poetry run python cli.py status

# List available profiles
PYTHONPATH=src poetry run python cli.py config list

# Switch to Korean configuration
PYTHONPATH=src poetry run python cli.py config switch korean

# Verify the switch worked
PYTHONPATH=src poetry run python cli.py config show
```

### Development Workflow

```bash
# Validate current configuration before running experiments
PYTHONPATH=src poetry run python cli.py config validate

# Switch to different embedding model for testing
PYTHONPATH=src poetry run python cli.py config switch polyglot-1b

# Check that data files exist
PYTHONPATH=src poetry run python cli.py status
```

### Profile Management

```bash
# Add new profile to embedding_profiles.yaml
# Then validate it works
PYTHONPATH=src poetry run python cli.py config list

# Switch to new profile
PYTHONPATH=src poetry run python cli.py config switch new_profile
```

## Integration with Existing Tools

### Compatibility with `switch_config.py`

The CLI tool is fully compatible with the existing `switch_config.py` script:

```bash
# These commands are equivalent:
PYTHONPATH=src poetry run python switch_config.py korean
PYTHONPATH=src poetry run python cli.py config switch korean

# Both update the same settings.yaml file
```

### Integration with Scripts

The CLI tool integrates seamlessly with existing project scripts:

```bash
# Switch configuration and run evaluation
PYTHONPATH=src poetry run python cli.py config switch english
PYTHONPATH=src poetry run python scripts/evaluation/evaluate.py

# Setup and run indexing
PYTHONPATH=src poetry run python cli.py setup --profile bilingual
PYTHONPATH=src poetry run python scripts/maintenance/reindex.py
```

## Error Handling

The CLI tool provides comprehensive error handling:

### Common Error Messages

- **Profile not found**: `❌ Error: Profile 'xyz' not found in conf/embedding_profiles.yaml`
- **Configuration file missing**: `❌ No profiles found or profiles file is missing`
- **Validation failed**: `❌ Validation failed: [specific error]`
- **API key missing**: Warning for Solar API when `UPSTAGE_API_KEY` is not set

### Troubleshooting

1. **Command not found**: Ensure you're using `PYTHONPATH=src poetry run python cli.py`
2. **Profile not available**: Check `conf/embedding_profiles.yaml` for available profiles
3. **Configuration not updating**: Verify write permissions on `conf/settings.yaml`
4. **Data files missing**: Check that required data files exist in the `data/` directory

## Advanced Usage

### Custom Profiles

To add a new profile:

1. Edit `conf/embedding_profiles.yaml`
2. Add your new profile following the existing structure
3. Test with: `PYTHONPATH=src poetry run python cli.py config switch your_profile`

### Batch Operations

```bash
# Validate before switching
PYTHONPATH=src poetry run python cli.py config validate

# Switch and show result
PYTHONPATH=src poetry run python cli.py config switch korean --verbose

# Check status after operations
PYTHONPATH=src poetry run python cli.py status
```

### Integration with CI/CD

The CLI tool can be used in automated pipelines:

```bash
# Setup for Korean evaluation
PYTHONPATH=src poetry run python cli.py setup --profile korean --force

# Validate configuration
PYTHONPATH=src poetry run python cli.py config validate

# Run evaluation pipeline
PYTHONPATH=src poetry run python scripts/evaluation/evaluate.py
```

## Help System

The CLI includes a comprehensive help system:

```bash
# Main help
PYTHONPATH=src poetry run python cli.py --help

# Config subcommand help
PYTHONPATH=src poetry run python cli.py config --help

# Specific command help
PYTHONPATH=src poetry run python cli.py config switch --help
```

## Contributing

To extend the CLI tool:

1. Add new commands in `cli.py`
2. Update this documentation
3. Test with various profiles
4. Ensure backward compatibility

## Related Files

- `conf/embedding_profiles.yaml`: Configuration profiles
- `conf/settings.yaml`: Current active configuration
- `switch_config.py`: Legacy configuration switching tool
- `scripts/`: Project scripts that use the configuration</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/docs/CLI_TOOL_README.md