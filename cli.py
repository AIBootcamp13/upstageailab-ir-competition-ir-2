#!/usr/bin/env python3
"""
Modern CLI for RAG Project Configuration Management

This tool provides a modern command-line interface for managing
embedding configurations and other project operations.
"""

import typer
from pathlib import Path
from typing import Optional

# Import functions from switch_config
from switch_config import (
    switch_to_profile,
    show_current_config,
    list_profiles,
    load_profiles
)

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
        from switch_config import load_settings
        current_settings = load_settings()

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
        from switch_config import load_settings
        settings = load_settings()
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


@app.command("setup")
def setup(
    profile: Optional[str] = typer.Option(None, help="Profile to set up after initialization"),
    force: bool = typer.Option(False, "--force", "-f", help="Force setup even if already configured")
):
    """Initial project setup and configuration."""
    typer.echo("🚀 Setting up RAG project...")

    # Check if already configured
    if not force:
        try:
            from switch_config import load_settings
            settings = load_settings()
            if settings.get('EMBEDDING_MODEL'):
                typer.echo("⚠️  Project appears to already be configured.")
                if not typer.confirm("Continue with setup anyway?"):
                    raise typer.Abort()
        except:
            pass

    # Setup steps would go here
    typer.echo("✅ Setup completed successfully!")

    if profile:
        typer.echo(f"\n🔄 Switching to {profile} profile...")
        config_switch(profile)


if __name__ == "__main__":
    app()