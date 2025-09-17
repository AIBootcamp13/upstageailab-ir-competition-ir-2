#!/usr/bin/env python3
"""
Data Management Menu Module for RAG CLI

This module provi        # Check if data files exist
        data_dir = self.project_root / "data"
        documents_file = data_dir / Path(self.get_current_documents_path()).name
        results["data_files_exist"] = documents_file.exists()        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:9200/_cluster/health"],
                capture_output=True, timeout=5
            )
            results["elasticsearch_connection"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            results["elasticsearch_connection"] = Falseessing and indexing commands for the CLI menu system.
It handles document indexing, data analysis, and duplicate detection.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from cli_menu.modules import BaseMenuModule


class DataManagementMenu(BaseMenuModule):
    """Data management menu commands for CLI integration."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)

    def get_current_documents_path(self) -> str:
        """Get the current documents path from the active data configuration"""
        try:
            # Import here to avoid circular imports
            sys.path.insert(0, str(self.project_root))
            from switch_config import get_current_documents_path
            return get_current_documents_path()
        except ImportError:
            # Fallback if switch_config is not available
            return "data/documents_ko.jsonl"

    def get_commands(self) -> Dict[str, List[Dict]]:
        """
        Get all data management-related commands.

        Returns:
            Dict containing data management commands organized by category
        """
        return {
            "Data Management": [
                {
                    "name": "Reindex Documents",
                    "command": lambda: f"{self.get_command_path('scripts/maintenance/reindex.py')} {self.get_current_documents_path()}",
                    "description": "Reindex documents to Elasticsearch",
                    "needs_params": False,
                },
                {
                    "name": "Analyze Data",
                    "command": f"{self.get_command_path('scripts/data/analyze_data.py')}",
                    "description": "Analyze document datasets for statistics",
                    "needs_params": False,
                },
                {
                    "name": "Check Duplicates",
                    "command": f"{self.get_command_path('scripts/data/check_duplicates.py')}",
                    "description": "Detect duplicate entries in datasets",
                    "needs_params": False,
                },
            ]
        }

    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that data management components are properly configured.

        Returns:
            Dict with validation results for different components
        """
        results = {
            "data_files_exist": False,
            "elasticsearch_connection": False,
            "data_scripts_exist": False,
            "maintenance_scripts_exist": False,
        }

        # Check if data files exist
        data_dir = self.project_root / "data"
        documents_file = data_dir / "documents_ko.jsonl"
        results["data_files_exist"] = documents_file.exists()

        # Check data processing scripts
        data_scripts = [
            "scripts/data/analyze_data.py",
            "scripts/data/check_duplicates.py",
        ]
        results["data_scripts_exist"] = all(
            (self.project_root / script).exists() for script in data_scripts
        )

        # Check maintenance scripts
        maintenance_scripts = [
            "scripts/maintenance/reindex.py",
        ]
        results["maintenance_scripts_exist"] = all(
            (self.project_root / script).exists() for script in maintenance_scripts
        )

        # Check Elasticsearch connection
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:9200/_cluster/health"],
                capture_output=True, timeout=5
            )
            results["elasticsearch_connection"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results["elasticsearch_connection"] = False

        return results

    def get_setup_instructions(self) -> str:
        """
        Get setup instructions for data management components.

        Returns:
            String with setup instructions
        """
        validation = self.validate_setup()

        instructions = []

        if not validation["data_files_exist"]:
            current_docs_path = self.get_current_documents_path()
            instructions.append(f"⚠️  Data files missing - ensure {current_docs_path} exists")
            instructions.append(f"   Check: ls -la {current_docs_path}")

        if not validation["data_scripts_exist"]:
            instructions.append("❌ Data processing scripts missing - check scripts/data/ directory")

        if not validation["maintenance_scripts_exist"]:
            instructions.append("❌ Maintenance scripts missing - check scripts/maintenance/ directory")

        if not validation["elasticsearch_connection"]:
            instructions.append("❌ Elasticsearch not available - start services first")
            instructions.append("   Run: ./scripts/execution/run-local.sh start")

        if all(validation.values()):
            instructions.append("✅ All data management components are properly set up!")

        return "\n".join(instructions)

    def get_usage_examples(self) -> List[str]:
        """
        Get usage examples for data management commands.

        Returns:
            List of usage examples
        """
        return [
            "# Reindex documents to Elasticsearch",
            f"PYTHONPATH=src poetry run python scripts/maintenance/reindex.py {self.get_current_documents_path()}",
            "",
            "# Analyze document datasets",
            "PYTHONPATH=src poetry run python scripts/data/analyze_data.py",
            "",
            "# Check for duplicate entries",
            "PYTHONPATH=src poetry run python scripts/data/check_duplicates.py",
        ]


# Factory function for easy integration
def get_data_management_menu(project_root: Path) -> DataManagementMenu:
    """Factory function to create DataManagementMenu instance."""
    return DataManagementMenu(project_root)