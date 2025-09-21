#!/usr/bin/env python3
"""
Setup & Infrastructure Menu Module for RAG CLI

This module provides setup and infrastructure-related commands for the CLI menu system.
It handles dependency installation, environment setup, and service management.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List

from cli_menu.modules import BaseMenuModule


class SetupMenu(BaseMenuModule):
    """Setup and infrastructure menu commands for CLI integration."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)

    def get_commands(self) -> Dict[str, List[Dict]]:
        """
        Get all setup and infrastructure-related commands.

        Returns:
            Dict containing setup commands organized by category
        """
        return {
            "Setup & Infrastructure": [
                {
                    "name": "Install Dependencies",
                    "command": "uv sync",
                    "description": "Install all project dependencies using uv",
                    "needs_params": False,
                },
                {
                    "name": "Run Smoke Test",
                    "command": f"{self.get_command_path('scripts/evaluation/smoke_test.py')}",
                    "description": "Run smoke tests to verify system health",
                    "needs_params": False,
                },
                # {
                #     "name": "Start Local Services",
                #     "command": "./scripts/execution/run-local.sh start",
                #     "description": "Start Elasticsearch, Redis, and Kibana locally (skips if already running)",
                #     "needs_params": False,
                # },
                {
                    "name": "Check Service Status",
                    "command": "./scripts/execution/run-local.sh status",
                    "description": "Check status of local services (Elasticsearch, Redis, Kibana)",
                    "needs_params": False,
                },
                # {
                #     "name": "Stop Local Services",
                #     "command": "./scripts/execution/run-local.sh stop",
                #     "description": "Stop local Elasticsearch, Redis, and Kibana",
                #     "needs_params": False,
                # }
            ]
        }

    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that the setup and infrastructure components are properly configured.

        Returns:
            Dict with validation results for different components
        """
        results = {
            "poetry_installed": False,
            "dependencies_installed": False,
            "env_file_exists": False,
            "services_available": False,
            "scripts_executable": False,
        }

        # Check uv installation
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            results["poetry_installed"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            results["poetry_installed"] = False

        # Check if dependencies are installed (check for uv.lock and pyproject.toml)
        uv_lock = self.project_root / "uv.lock"
        pyproject_toml = self.project_root / "pyproject.toml"
        results["dependencies_installed"] = uv_lock.exists() and pyproject_toml.exists()

        # Check environment file
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        results["env_file_exists"] = env_file.exists() or env_example.exists()

        # Check service scripts
        run_local_script = self.project_root / "scripts" / "execution" / "run-local.sh"
        results["scripts_executable"] = run_local_script.exists() and os.access(run_local_script, os.X_OK)

        # Check if services are running (basic check)
        try:
            # Check Elasticsearch
            es_result = subprocess.run(
                ["curl", "-s", "http://elasticsearch:9200/_cluster/health"],
                capture_output=True, timeout=5
            )
            es_running = es_result.returncode == 0

            # Check Redis
            redis_running = False
            try:
                redis_result = subprocess.run(
                    ["redis-cli", "-h", "redis", "ping"],
                    capture_output=True, timeout=5
                )
                redis_running = "PONG" in redis_result.stdout.decode()
            except (FileNotFoundError, subprocess.SubprocessError):
                # Fallback to Python redis library if redis-cli not available
                try:
                    import redis
                    r = redis.Redis(host='redis', port=6379)
                    redis_running = r.ping()
                except Exception:
                    redis_running = False

            # Check Kibana
            kibana_result = subprocess.run(
                ["curl", "-s", "http://kibana:5601/api/status"],
                capture_output=True, timeout=5
            )
            kibana_running = kibana_result.returncode == 0

            results["services_available"] = es_running and redis_running and kibana_running
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results["services_available"] = False

        return results

    def get_setup_instructions(self) -> str:
        """
        Get setup instructions for infrastructure components.

        Returns:
            String with setup instructions
        """
        validation = self.validate_setup()

        instructions = []

        if not validation["poetry_installed"]:
            instructions.append("❌ uv not installed - required for dependency management")
            instructions.append("   Install: curl -LsSf https://astral.sh/uv/install.sh | sh")

        if not validation["dependencies_installed"]:
            instructions.append("⚠️  Dependencies not installed - run uv sync")
            instructions.append("   Run: uv sync")

        if not validation["env_file_exists"]:
            instructions.append("⚠️  Environment file missing - copy from template")
            instructions.append("   Run: cp .env.example .env")

        if not validation["scripts_executable"]:
            instructions.append("❌ Service scripts not executable - check scripts/execution/")
            instructions.append("   Run: chmod +x scripts/execution/run-local.sh")

        if not validation["services_available"]:
            instructions.append("⚠️  Local services not running - start them for full functionality")
            instructions.append("   Run: ./scripts/execution/run-local.sh start")

        if all(validation.values()):
            instructions.append("✅ All infrastructure components are properly set up!")

        return "\n".join(instructions)

    def get_usage_examples(self) -> List[str]:
        """
        Get usage examples for setup commands.

        Returns:
            List of usage examples
        """
        return [
            "# Install all dependencies",
            "uv sync",
            "",
            "# Set up environment configuration",
            "cp .env.example .env",
            "",
            "# Start all local services",
            "./scripts/execution/run-local.sh start",
            "",
            "# Check service status",
            "./scripts/execution/run-local.sh status",
            "",
            "# Stop all services",
            "./scripts/execution/run-local.sh stop",
        ]


# Factory function for easy integration
def get_setup_menu(project_root: Path) -> SetupMenu:
    """Factory function to create SetupMenu instance."""
    return SetupMenu(project_root)