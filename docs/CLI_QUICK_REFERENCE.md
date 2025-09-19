# CLI Quick Reference Card

## Essential Commands

```bash
# Check project status
PYTHONPATH=src poetry run python cli.py status

# List available profiles
PYTHONPATH=src poetry run python cli.py config list

# Switch configuration
PYTHONPATH=src poetry run python cli.py config switch korean

# Validate current config
PYTHONPATH=src poetry run python cli.py config validate

# Show current settings
PYTHONPATH=src poetry run python cli.py config show

# Get detailed help
PYTHONPATH=src poetry run python cli.py help
```

## Available Profiles

| Profile | Use Case | Dimensions | Data |
|---------|----------|------------|------|
| `korean` | Korean documents | 768d | Korean |
| `english` | English documents | 384d | English |
| `bilingual` | Mixed languages | 768d | Bilingual |
| `solar` | High quality | 4096d | Bilingual |
| `polyglot` | Korean optimized | 3072d | Korean |
| `polyglot-1b` | Lightweight | 2048d | Korean |

## Common Workflows

### Development Setup
```bash
# Initial setup
PYTHONPATH=src poetry run python cli.py setup --profile korean

# Verify configuration
PYTHONPATH=src poetry run python cli.py config validate
```

### Experimentation
```bash
# Switch models for testing
PYTHONPATH=src poetry run python cli.py config switch english
PYTHONPATH=src poetry run python cli.py config switch polyglot-1b

# Check status between experiments
PYTHONPATH=src poetry run python cli.py status
```

### Production
```bash
# Validate before deployment
PYTHONPATH=src poetry run python cli.py config validate

# Use high-quality embeddings
PYTHONPATH=src poetry run python cli.py config switch solar
```

## Troubleshooting

- **Command not found**: Use `PYTHONPATH=src poetry run python cli.py`
- **Profile not found**: Run `config list` to see available profiles
- **Config not updating**: Check write permissions on `conf/settings.yaml`
- **Data files missing**: Run `status` to verify data file availability

## Integration

Works seamlessly with existing tools:
- Compatible with `switch_config.py`
- Updates same `settings.yaml` file
- Integrates with all project scripts
- Can be used in CI/CD pipelines

---
📖 Full documentation: `docs/CLI_TOOL_README.md`