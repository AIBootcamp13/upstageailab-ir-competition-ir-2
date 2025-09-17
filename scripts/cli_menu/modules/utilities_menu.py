#!/usr/bin/env python3
"""
Utilities Menu Module for RAG CLI

This module provides utility and helper commands for the CLI menu system.
It handles testing, monitoring, and maintenance tasks.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List

from cli_menu.modules import BaseMenuModule


class UtilitiesMenu(BaseMenuModule):
    """Utilities menu commands for CLI integration."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)

    def get_commands(self) -> Dict[str, List[Dict]]:
        """
        Get all utility-related commands.

        Returns:
            Dict containing utility commands organized by category
        """
        return {
            "Utilities": [
                {
                    "name": "Run Smoke Tests",
                    "command": f"{self.get_command_path('scripts/evaluation/smoke_test.py')}",
                    "description": "Run smoke tests to verify system health",
                    "needs_params": False,
                },
                {
                    "name": "Test HuggingFace Integration",
                    "command": f"{self.get_command_path('scripts/test_huggingface_integration.py')}",
                    "description": "Test HuggingFace model integration for retrieval and generation",
                    "needs_params": False,
                },
                {
                    "name": "List All Scripts",
                    "command": "poetry run python scripts/list_scripts.py",
                    "description": "List all available scripts with descriptions",
                    "needs_params": False,
                },
                {
                    "name": "Show Current Configuration",
                    "command": "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py show",
                    "description": "Display current RAG system configuration (embeddings, index, etc.)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Korean Configuration",
                    "command": "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py korean",
                    "description": "Switch to Korean configuration (768D embeddings, Korean data)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to English Configuration",
                    "command": "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py english",
                    "description": "Switch to English configuration (384D embeddings, bilingual data)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Bilingual Configuration",
                    "command": "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py bilingual",
                    "description": "Switch to Bilingual configuration (768D embeddings, mixed data)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Solar API Configuration",
                    "command": "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py solar",
                    "description": "Switch to Solar API configuration (4096D embeddings, requires API key)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Polyglot-Ko-3.8B Configuration",
                    "command": "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py polyglot-3b",
                    "description": "Switch to Polyglot-Ko-3.8B configuration (3072D embeddings, local model)",
                    "needs_params": False,
                },
                {
                    "name": "Switch to Polyglot-Ko-1.3B Configuration",
                    "command": "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py polyglot-1b",
                    "description": "Switch to Polyglot-Ko-1.3B configuration (2048D embeddings, local model)",
                    "needs_params": False,
                },
                {
                    "name": "Clean Distributions",
                    "command": "./scripts/infra/cleanup-distros.sh",
                    "description": "Clean up downloaded service distributions",
                    "needs_params": False,
                },
                {
                    "name": "Launch Streamlit UI",
                    "command": "poetry run streamlit run scripts/visualize_submissions.py",
                    "description": "Launch the Streamlit UI for visualizing RAG submission results",
                    "needs_params": False,
                    "run_in_background": True,
                },
            ]
        }

    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that utility components are properly configured.

        Returns:
            Dict with validation results for different components
        """
        results = {
            "test_scripts_exist": False,
            "utility_scripts_exist": False,
            "streamlit_available": False,
            "infrastructure_scripts_exist": False,
            "script_listing_available": False,
            "config_switcher_available": False,
            "settings_file_exists": False,
        }

        # Check test scripts
        test_scripts = [
            "scripts/evaluation/smoke_test.py",
            "scripts/test_huggingface_integration.py",
        ]
        results["test_scripts_exist"] = all(
            (self.project_root / script).exists() for script in test_scripts
        )

        # Check utility scripts
        utility_scripts = [
            "scripts/list_scripts.py",
            "scripts/visualize_submissions.py",
        ]
        results["utility_scripts_exist"] = all(
            (self.project_root / script).exists() for script in utility_scripts
        )

        # Check infrastructure scripts
        infra_scripts = [
            "scripts/infra/cleanup-distros.sh",
        ]
        results["infrastructure_scripts_exist"] = all(
            (self.project_root / script).exists() for script in infra_scripts
        )

        # Check script listing functionality
        list_script = self.project_root / "scripts" / "list_scripts.py"
        results["script_listing_available"] = list_script.exists()

        # Check configuration switcher
        config_script = self.project_root / "scripts" / "indexing" / "switch_config.py"
        results["config_switcher_available"] = config_script.exists()

        # Check settings file
        settings_file = self.project_root / "conf" / "settings.yaml"
        results["settings_file_exists"] = settings_file.exists()

        # Check Streamlit availability
        try:
            import streamlit
            results["streamlit_available"] = True
        except ImportError:
            results["streamlit_available"] = False

        return results

    def get_setup_instructions(self) -> str:
        """
        Get setup instructions for utility components.

        Returns:
            String with setup instructions
        """
        validation = self.validate_setup()

        instructions = []

        if not validation["test_scripts_exist"]:
            instructions.append("⚠️  Test scripts missing - check scripts/evaluation/ and scripts/ directories")

        if not validation["utility_scripts_exist"]:
            instructions.append("⚠️  Utility scripts missing - check scripts/ directory")

        if not validation["infrastructure_scripts_exist"]:
            instructions.append("⚠️  Infrastructure scripts missing - check scripts/infra/ directory")

        if not validation["config_switcher_available"]:
            instructions.append("⚠️  Configuration switcher missing - check switch_config.py")

        if not validation["settings_file_exists"]:
            instructions.append("⚠️  Settings file missing - check conf/settings.yaml")

        if not validation["streamlit_available"]:
            instructions.append("⚠️  Streamlit not available - install for UI functionality")
            instructions.append("   Run: poetry add streamlit")

        if validation["config_switcher_available"] and validation["settings_file_exists"]:
            instructions.append("✅ Configuration switching is available!")
            instructions.append("   Use: PYTHONPATH=src poetry run python scripts/indexing/switch_config.py [korean|english|bilingual|solar|show]")

        if validation["utility_scripts_exist"]:
            instructions.append("✅ Core utility components are available!")

        return "\n".join(instructions)

    def get_usage_examples(self) -> List[str]:
        """
        Get usage examples for utility commands.

        Returns:
            List of usage examples
        """
        return [
            "# Run smoke tests",
            "PYTHONPATH=src poetry run python scripts/evaluation/smoke_test.py",
            "",
            "# Test HuggingFace integration",
            "PYTHONPATH=src poetry run python scripts/test_huggingface_integration.py",
            "",
            "# List all available scripts",
            "poetry run python scripts/list_scripts.py",
            "",
            "# Show current configuration",
            "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py show",
            "",
            "# Switch configurations",
            "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py korean    # Korean (768D)",
            "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py english   # English (384D)",
            "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py bilingual # Bilingual (768D)",
            "PYTHONPATH=src poetry run python scripts/indexing/switch_config.py solar     # Solar API (4096D)",
            "",
            "# Clean up distributions",
            "./scripts/infra/cleanup-distros.sh",
            "",
            "# Launch Streamlit UI",
            "poetry run streamlit run scripts/visualize_submissions.py",
        ]


# Factory function for easy integration
def get_utilities_menu(project_root: Path) -> UtilitiesMenu:
    """Factory function to create UtilitiesMenu instance."""
    return UtilitiesMenu(project_root)