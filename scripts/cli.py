#!/usr/bin/env python3
"""
Modern CLI for RAG Project Configuration Management

This tool provides a modern command-line interface for managing
embedding configurations and other project operations.

Available Profiles:
- korean: Korean setup using KR-SBERT (768d)
- english: English setup using all-MiniLM-L6-v2 (384d)
- bilingual: Bilingual setup using KR-SBERT (768d)
- solar: Solar API setup (4096d)
- polyglot: Polyglot-Ko-3.8B setup (3072d)
- polyglot-3b: Polyglot-Ko-3.8B setup (3072d)
- polyglot-1b: Polyglot-Ko-1.3B setup (2048d)

Usage Examples:
  PYTHONPATH=src uv run python cli.py config list
  PYTHONPATH=src uv run python cli.py config switch korean
  PYTHONPATH=src uv run python cli.py config validate
  PYTHONPATH=src uv run python cli.py status

For full documentation, see: docs/CLI_TOOL_README.md
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add parent directory for switch_config

import typer

# Import functions from switch_config (keeping for compatibility)
try:
    from switch_config import (  # type: ignore
        switch_to_profile,
        show_current_config,
        list_profiles,
        load_profiles
    )
except ImportError:
    # Fallback if switch_config is not available
    def switch_to_profile(profile_name: str) -> None:
        print(f"❌ switch_config not available. Cannot switch to profile: {profile_name}")

    def show_current_config() -> None:
        print("❌ switch_config not available. Cannot show current config.")

    def list_profiles() -> None:
        print("❌ switch_config not available. Cannot list profiles.")

    def load_profiles() -> dict:
        print("❌ switch_config not available. Returning empty profiles.")
        return {}

# Create a Typer app
app = typer.Typer(
    help="A modern CLI for managing the RAG project configuration.",
    add_completion=False
)

config_app = typer.Typer(
    help="Switch, show, or list embedding configurations."
)
app.add_typer(config_app, name="config")


@config_app.command("switch")
def config_switch(
    profile_name: str = typer.Argument(..., help="The profile to switch to."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output.")
):
    """Switches the active embedding configuration to a defined profile."""
    try:
        switch_to_profile(profile_name)
        if verbose:
            show_current_config()
    except Exception as e:
        typer.echo(f"❌ Error switching to profile '{profile_name}': {e}", err=True)
        raise typer.Exit(1)


@config_app.command("show")
def config_show():
    """Shows the current configuration from settings.yaml."""
    try:
        show_current_config()
    except Exception as e:
        typer.echo(f"❌ Error showing configuration: {e}", err=True)
        raise typer.Exit(1)


@config_app.command("list")
def config_list():
    """Lists all available configuration profiles."""
    try:
        list_profiles()
    except Exception as e:
        typer.echo(f"❌ Error listing profiles: {e}", err=True)
        raise typer.Exit(1)


@config_app.command("validate")
def config_validate():
    """Validates the current configuration for consistency."""
    typer.echo("⚙️ Validating current configuration...")

    try:
        profiles = load_profiles()
        if not profiles:
            typer.echo("❌ No profiles found in configuration file.", err=True)
            raise typer.Exit(1)

        # Basic validation - check if current settings match a known profile
        try:
            from switch_config import load_settings  # type: ignore
            current_settings = load_settings()
        except ImportError:
            typer.echo("❌ switch_config module not available for validation", err=True)
            raise typer.Exit(1)

        embedding_provider = current_settings.get('EMBEDDING_PROVIDER', 'huggingface')
        embedding_model = current_settings.get('EMBEDDING_MODEL')
        embedding_dimension = current_settings.get('EMBEDDING_DIMENSION')
        index_name = current_settings.get('INDEX_NAME')

        typer.echo("✅ Configuration structure is valid")
        typer.echo(f"   Current provider: {embedding_provider}")
        typer.echo(f"   Current model: {embedding_model}")
        typer.echo(f"   Current dimension: {embedding_dimension}")
        typer.echo(f"   Current index: {index_name}")

        # Check if current config matches any profile
        matched_profile = None
        for profile_name, profile_data in profiles.items():
            config = profile_data.get('config', {})
            if (config.get('EMBEDDING_PROVIDER') == embedding_provider and
                config.get('EMBEDDING_MODEL') == embedding_model and
                config.get('EMBEDDING_DIMENSION') == embedding_dimension and
                config.get('INDEX_NAME') == index_name):
                matched_profile = profile_name
                break

        if matched_profile:
            typer.echo(f"✅ Configuration matches profile: {matched_profile}")
        else:
            typer.echo("⚠️  Current configuration doesn't match any defined profile")
            typer.echo("   This might be expected if you've made custom changes")

    except Exception as e:
        typer.echo(f"❌ Validation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("status")
def status():
    """Shows a quick status overview of the project."""
    typer.echo("🔍 Project Status:")

    # Check if services are running
    typer.echo("\n📊 Services:")
    # This would need to be implemented based on your service checking logic

    # Show current config
    typer.echo("\n⚙️  Current Configuration:")
    try:
        try:
            from switch_config import load_settings  # type: ignore
            settings = load_settings()
        except ImportError:
            typer.echo("   ❌ switch_config module not available")
            settings = {}
        typer.echo(f"   Provider: {settings.get('EMBEDDING_PROVIDER', 'Not set')}")
        typer.echo(f"   Model: {settings.get('EMBEDDING_MODEL', 'Not set')}")
        typer.echo(f"   Index: {settings.get('INDEX_NAME', 'Not set')}")
    except Exception as e:
        typer.echo(f"   ❌ Error loading config: {e}")

    # Check data files
    typer.echo("\n📁 Data Files:")
    data_files = [
        "data/documents_ko.jsonl",
        "data/documents_bilingual.jsonl",
        "data/eval.jsonl"
    ]

    for file_path in data_files:
        if Path(file_path).exists():
            typer.echo(f"   ✅ {file_path}")
        else:
            typer.echo(f"   ❌ {file_path} (missing)")


@app.command("help")
def show_help():
    """Shows detailed help and usage examples."""
    help_text = """
🔧 RAG CLI Tool - Quick Reference Guide
========================================

📋 AVAILABLE PROFILES:
  korean      - Korean KR-SBERT (768d) for Korean documents
  english     - English MiniLM (384d) for English documents
  bilingual   - Korean KR-SBERT (768d) for bilingual documents
  solar       - Solar API (4096d) for high-quality embeddings
  polyglot    - Polyglot-Ko-3.8B (3072d) for Korean text
  polyglot-3b - Polyglot-Ko-3.8B (3072d) optimized settings
  polyglot-1b - Polyglot-Ko-1.3B (2048d) lightweight option

🚀 COMMON WORKFLOWS:

  # Check project status
  PYTHONPATH=src uv run python cli.py status

  # List all available profiles
  PYTHONPATH=src uv run python cli.py config list

  # Switch to Korean configuration
  PYTHONPATH=src uv run python cli.py config switch korean

  # Validate current configuration
  PYTHONPATH=src uv run python cli.py config validate

  # Show current settings
  PYTHONPATH=src uv run python cli.py config show

  # Setup project with specific profile
  PYTHONPATH=src uv run python cli.py setup --profile bilingual

📖 FOR DETAILED DOCUMENTATION:
  See: docs/CLI_TOOL_README.md

🔗 COMPATIBILITY:
  This tool works alongside switch_config.py and updates the same settings.yaml
"""
    typer.echo(help_text)


if __name__ == "__main__":
    app()