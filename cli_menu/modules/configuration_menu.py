#!/usr/bin/env python3
"""
Configuration Management Menu Module for RAG CLI

This module provides configuration switching and management commands for the CLI menu system.
It integrates with switch_config.py to allow seamless configuration changes.
"""

import os
from pathlib import Path
from typing import Dict, List

from cli_menu.modules import BaseMenuModule


class ConfigurationMenu(BaseMenuModule):
    """Configuration management menu commands for CLI integration."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)

    def get_commands(self) -> Dict[str, List[Dict]]:
        """
        Get all configuration-related commands.

        Returns:
            Dict containing configuration commands organized by category
        """
        return {
            "Configuration Management": [
                {
                    "name": "Switch to Korean Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} korean",
                    "description": "Switch to Korean configuration (KR-SBERT 768d, Korean data, Korean index)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to English Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} english",
                    "description": "Switch to English configuration (MiniLM 384d, Bilingual data, English index)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Bilingual Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} bilingual",
                    "description": "Switch to Bilingual configuration (KR-SBERT 768d, Bilingual data, Bilingual index)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Solar API Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} solar",
                    "description": "Switch to Solar API configuration (4096d embeddings via Upstage API)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Polyglot-Ko Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} polyglot",
                    "description": "Switch to Polyglot-Ko-3.8B configuration (3072d embeddings)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Polyglot-Ko-3.8B Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} polyglot-3b",
                    "description": "Switch to Polyglot-Ko-3.8B configuration (3072d embeddings)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Polyglot-Ko-1.3B Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} polyglot-1b",
                    "description": "Switch to Polyglot-Ko-1.3B configuration (2048d embeddings)",
                    "needs_params": False,
                },
                {
                    "name": "Show Current Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} show",
                    "description": "Display current configuration settings",
                    "needs_params": False,
                },
                {
                    "name": "Create Index for Current Config",
                    "command": f"{self.get_command_path('scripts/maintenance/reindex.py')}",
                    "description": "Create Elasticsearch index using current configuration settings",
                    "needs_params": True,
                    "params": ["data_file", "index_name"],
                },
            ]
        }

    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that configuration management components are properly set up.

        Returns:
            Dict with validation results for different components
        """
        results = {
            "switch_config_exists": False,
            "settings_file_exists": False,
            "data_files_exist": False,
            "reindex_script_exists": False,
        }

        # Check switch_config.py
        switch_config_path = self.project_root / "switch_config.py"
        results["switch_config_exists"] = switch_config_path.exists()

        # Check settings file
        settings_path = self.project_root / "conf" / "settings.yaml"
        results["settings_file_exists"] = settings_path.exists()

        # Check data files
        data_files = [
            "data/documents_ko.jsonl",
            "data/documents_bilingual.jsonl",
        ]
        results["data_files_exist"] = any(
            (self.project_root / data_file).exists() for data_file in data_files
        )

        # Check reindex script
        reindex_path = self.project_root / "scripts" / "maintenance" / "reindex.py"
        results["reindex_script_exists"] = reindex_path.exists()

        return results

    def get_setup_instructions(self) -> str:
        """
        Get setup instructions for configuration management components.

        Returns:
            String with setup instructions
        """
        validation = self.validate_setup()

        instructions = []

        if not validation["switch_config_exists"]:
            instructions.append("❌ switch_config.py missing - configuration switching not available")

        if not validation["settings_file_exists"]:
            instructions.append("❌ Configuration file missing - check conf/settings.yaml")

        if not validation["data_files_exist"]:
            instructions.append("⚠️  Data files missing - check data/ directory")
            instructions.append("   Need: data/documents_ko.jsonl or data/documents_bilingual.jsonl")

        if not validation["reindex_script_exists"]:
            instructions.append("❌ Reindex script missing - check scripts/maintenance/reindex.py")

        if validation["switch_config_exists"] and validation["settings_file_exists"]:
            instructions.append("✅ Core configuration components are available!")

        return "\n".join(instructions)

    def get_usage_examples(self) -> List[str]:
        """
        Get usage examples for configuration commands.

        Returns:
            List of usage examples
        """
        return [
            "# Switch to Korean configuration",
            "PYTHONPATH=src uv run python switch_config.py korean",
            "",
            "# Switch to English configuration",
            "PYTHONPATH=src uv run python switch_config.py english",
            "",
            "# Show current configuration",
            "PYTHONPATH=src uv run python switch_config.py show",
            "",
            "# Create index after configuration switch",
            "PYTHONPATH=src uv run python scripts/maintenance/reindex.py data/documents_ko.jsonl --index documents_ko_with_embeddings_new",
        ]


# Factory function for easy integration
def get_configuration_menu(project_root: Path) -> ConfigurationMenu:
    """Factory function to create ConfigurationMenu instance."""
    return ConfigurationMenu(project_root)