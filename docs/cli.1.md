# cli.py(1) - RAG Configuration Management Tool

## NAME
cli.py - Modern command-line interface for RAG project configuration management

## SYNOPSIS
`PYTHONPATH=src poetry run python cli.py` [COMMAND] [OPTIONS]

## DESCRIPTION
The cli.py tool provides a modern, user-friendly interface for managing embedding configurations in the RAG (Retrieval-Augmented Generation) project. It supports profile-based configuration switching, validation, and project status monitoring.

## COMMANDS

### Main Commands

**status**
: Shows a quick overview of project status including current configuration and data file availability.

**help**
: Displays detailed help and usage examples with common workflows.

**setup** [OPTIONS]
: Initial project setup and configuration.
: - `--profile PROFILE`: Profile to switch to after initialization
: - `--force, -f`: Force setup even if already configured

### Configuration Subcommands (config)

**config list**
: Lists all available configuration profiles from embedding_profiles.yaml.

**config switch** PROFILE [OPTIONS]
: Switches to a specified configuration profile.
: - `PROFILE`: Name of the profile to switch to (required)
: - `--verbose, -v`: Show detailed output after switching

**config show**
: Displays the current configuration from settings.yaml.

**config validate**
: Validates current configuration for consistency and profile matching.

## CONFIGURATION PROFILES

| Profile | Description | Dimensions | Provider | Data |
|---------|-------------|------------|----------|------|
| korean | Korean KR-SBERT | 768d | huggingface | ko |
| english | English MiniLM-L6-v2 | 384d | huggingface | en |
| bilingual | Bilingual KR-SBERT | 768d | huggingface | bilingual |
| solar | Solar API | 4096d | solar | bilingual |
| polyglot | Polyglot-Ko-3.8B | 3072d | polyglot | ko |
| polyglot-3b | Polyglot-Ko-3.8B | 3072d | polyglot | ko |
| polyglot-1b | Polyglot-Ko-1.3B | 2048d | polyglot | ko |

## EXAMPLES

### Basic Usage
```bash
# Check project status
PYTHONPATH=src poetry run python cli.py status

# List available profiles
PYTHONPATH=src poetry run python cli.py config list

# Switch to Korean configuration
PYTHONPATH=src poetry run python cli.py config switch korean

# Validate configuration
PYTHONPATH=src poetry run python cli.py config validate
```

### Development Workflow
```bash
# Setup with specific profile
PYTHONPATH=src poetry run python cli.py setup --profile bilingual

# Switch between models for testing
PYTHONPATH=src poetry run python cli.py config switch english
PYTHONPATH=src poetry run python cli.py config switch polyglot-1b

# Get detailed help
PYTHONPATH=src poetry run python cli.py help
```

### Advanced Usage
```bash
# Switch with verbose output
PYTHONPATH=src poetry run python cli.py config switch solar --verbose

# Force setup
PYTHONPATH=src poetry run python cli.py setup --force --profile korean
```

## ENVIRONMENT

**PYTHONPATH**
: Must include `src` directory for proper module resolution.

**UPSTAGE_API_KEY**
: Required when using Solar API profiles.

## FILES

- `conf/embedding_profiles.yaml`: Configuration profiles definition
- `conf/settings.yaml`: Current active configuration
- `conf/data/`: Data configuration files
- `data/`: Document and evaluation data files

## SEE ALSO

switch_config.py(1), embedding_profiles.yaml(5), settings.yaml(5)

## BUGS

Report issues to the project repository.

## AUTHOR

RAG Project Team

## VERSION

CLI Tool v1.0 - Built with Typer</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/docs/cli.1.md