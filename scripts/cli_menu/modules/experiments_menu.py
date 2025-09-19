#!/usr/bin/env python3
"""
Experiments & Validation Menu Module for RAG CLI

This module provides experiment running and validation commands for the CLI menu system.
It handles retrieval validation, model testing, and performance evaluation.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List

from cli_menu.modules import BaseMenuModule


class ExperimentsMenu(BaseMenuModule):
    """Experiments and validation menu commands for CLI integration."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)

    def get_commands(self) -> Dict[str, List[Dict]]:
        """
        Get all experiments and validation-related commands.

        Returns:
            Dict containing experiment commands organized by category
        """
        return {
            "Experiments & Validation": [
                {
                    "name": "Check Current Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} show",
                    "description": "Show current configuration before running experiments (IMPORTANT: Ensure correct embedding model and index)",
                    "needs_params": False,
                },
                {
                    "name": "Validate Retrieval (OpenAI)",
                    "command": f"{self.get_command_path('scripts/evaluation/validate_retrieval.py')} --config-dir conf pipeline=default",
                    "description": "Run retrieval validation using OpenAI models. Uses current embedding model/index from configuration.",
                    "needs_params": True,
                    "params": ["model.alpha", "limit"],
                },
                {
                    "name": "Validate Retrieval (Qwen2:7b Full)",
                    "command": f"{self.get_command_path('scripts/evaluation/validate_retrieval.py')} --config-dir conf pipeline=qwen-full",
                    "description": "Run retrieval validation using Qwen2:7b for query rewriting, tool calling, and answer generation",
                    "needs_params": True,
                    "params": ["model.alpha", "limit"],
                },
                {
                    "name": "Validate Retrieval (Llama3.1:8b Full)",
                    "command": f"{self.get_command_path('scripts/evaluation/validate_retrieval.py')} --config-dir conf pipeline=llama-full",
                    "description": "Run retrieval validation using Llama3.1:8b for query rewriting, tool calling, and answer generation",
                    "needs_params": True,
                    "params": ["model.alpha", "limit"],
                },
                {
                    "name": "Validate Retrieval (Ollama Hybrid)",
                    "command": f"{self.get_command_path('scripts/evaluation/validate_retrieval.py')} --config-dir conf pipeline=hybrid-qwen-llama",
                    "description": "Run retrieval validation using Qwen2:7b for query rewriting and tool calling, Llama3.1:8b for answer generation",
                    "needs_params": True,
                    "params": ["model.alpha", "limit"],
                },
                {
                    "name": "Validate Retrieval (Custom)",
                    "command": f"{self.get_command_path('scripts/evaluation/validate_retrieval.py')} --config-dir conf",
                    "description": "Run retrieval validation with custom parameters",
                    "needs_params": True,
                    "params": ["model.alpha", "limit", "experiment", "pipeline.generator_type", "pipeline.generator_model_name"],
                },
                {
                    "name": "Multi-Run Experiments",
                    "command": f"{self.get_command_path('scripts/evaluation/validate_retrieval.py')} --config-dir conf --multirun",
                    "description": "Run multiple experiments in parallel",
                    "needs_params": True,
                    "params": ["experiment", "limit"],
                },
            ]
        }

    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that experiments and validation components are properly configured.

        Returns:
            Dict with validation results for different components
        """
        results = {
            "validation_scripts_exist": False,
            "config_files_exist": False,
            "ollama_available": False,
            "huggingface_available": False,
            "data_files_exist": False,
        }

        # Check validation scripts
        validation_scripts = [
            "scripts/evaluation/validate_retrieval.py",
        ]
        results["validation_scripts_exist"] = all(
            (self.project_root / script).exists() for script in validation_scripts
        )

        # Check configuration files
        config_files = [
            "conf/settings.yaml",
        ]
        results["config_files_exist"] = all(
            (self.project_root / config).exists() for config in config_files
        )

        # Check data files
        data_files = [
            "data/eval.jsonl",
            "data/validation_balanced.jsonl",
        ]
        results["data_files_exist"] = any(
            (self.project_root / data_file).exists() for data_file in data_files
        )

        # Check Ollama availability
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True, timeout=5
            )
            results["ollama_available"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results["ollama_available"] = False

        # Check HuggingFace availability (basic check)
        try:
            import transformers
            results["huggingface_available"] = True
        except ImportError:
            results["huggingface_available"] = False

        return results

    def get_setup_instructions(self) -> str:
        """
        Get setup instructions for experiments and validation components.

        Returns:
            String with setup instructions
        """
        validation = self.validate_setup()

        instructions = []

        if not validation["validation_scripts_exist"]:
            instructions.append("❌ Validation scripts missing - check scripts/evaluation/ directory")

        if not validation["config_files_exist"]:
            instructions.append("❌ Configuration files missing - check conf/ directory")

        if not validation["data_files_exist"]:
            instructions.append("⚠️  Evaluation data files missing - check data/ directory")
            instructions.append("   Need: data/eval.jsonl or data/validation_balanced.jsonl")

        if not validation["ollama_available"]:
            instructions.append("⚠️  Ollama not available - start Ollama service for local models")
            instructions.append("   Run: ollama serve")

        if not validation["huggingface_available"]:
            instructions.append("⚠️  HuggingFace transformers not available - install for HuggingFace models")
            instructions.append("   Run: poetry add transformers torch")

        if validation["validation_scripts_exist"] and validation["config_files_exist"]:
            instructions.append("✅ Core validation components are available!")

        return "\n".join(instructions)

    def get_usage_examples(self) -> List[str]:
        """
        Get usage examples for experiment commands.

        Returns:
            List of usage examples
        """
        return [
            "# Basic retrieval validation",
            "PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py --config-dir conf",
            "",
            "# Qwen2:7b full pipeline validation",
            "PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py \\",
            "  --config-dir conf pipeline=qwen-full model.alpha=0.5 limit=50",
            "",
            "# Custom validation with parameters",
            "PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py \\",
            "  --config-dir conf model.alpha=0.3 limit=100 experiment=my_experiment",
            "",
            "# Multi-run experiments",
            "PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py \\",
            "  --config-dir conf --multirun experiment=my_experiment limit=50",
        ]


# Factory function for easy integration
def get_experiments_menu(project_root: Path) -> ExperimentsMenu:
    """Factory function to create ExperimentsMenu instance."""
    return ExperimentsMenu(project_root)