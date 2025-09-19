#!/usr/bin/env python3
"""
Evaluation & Submission Menu Module for RAG CLI

This module provides evaluation and submission generation commands for the CLI menu system.
It handles submission file creation, evaluation, and result processing.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List

from cli_menu.modules import BaseMenuModule


class EvaluationMenu(BaseMenuModule):
    """Evaluation and submission menu commands for CLI integration."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)

    def get_commands(self) -> Dict[str, List[Dict]]:
        """
        Get all evaluation and submission-related commands.

        Returns:
            Dict containing evaluation commands organized by category
        """
        return {
            "Evaluation & Submission": [
                {
                    "name": "Check Current Configuration",
                    "command": f"{self.get_command_path('switch_config.py')} show",
                    "description": "Show current configuration before generating submissions (IMPORTANT: Ensure correct embedding model and index)",
                    "needs_params": False,
                },
                {
                    "name": "Generate Submission (OpenAI)",
                    "command": f"{self.get_command_path('scripts/evaluation/evaluate.py')} --config-dir conf",
                    "description": "Generate submission using OpenAI GPT models. Uses current embedding model and index from configuration.",
                    "needs_params": True,
                    "params": ["model.alpha", "limit"],
                },
                {
                    "name": "Generate Submission (Qwen2:7b Full)",
                    "command": f"{self.get_command_path('scripts/evaluation/evaluate.py')} --config-dir conf pipeline=qwen-full",
                    "description": "Generate submission using Qwen2:7b for all pipeline stages. Uses current embedding model from settings.yaml.",
                    "needs_params": True,
                    "params": ["model.alpha", "limit", "qwen_submission_output_file"],
                },
                {
                    "name": "Generate Submission (Llama3.1:8b Full)",
                    "command": f"{self.get_command_path('scripts/evaluation/evaluate.py')} --config-dir conf pipeline=llama-full",
                    "description": "Generate submission using Llama3.1:8b for all pipeline stages. Uses current embedding model from settings.yaml.",
                    "needs_params": True,
                    "params": ["model.alpha", "limit"],
                },
                {
                    "name": "Generate Submission (Ollama Hybrid)",
                    "command": f"{self.get_command_path('scripts/evaluation/evaluate.py')} --config-dir conf pipeline=hybrid-qwen-llama",
                    "description": "Hybrid pipeline: Qwen2:7b for query processing, Llama3.1:8b for generation. Uses current embedding model.",
                    "needs_params": True,
                    "params": ["model.alpha", "limit"],
                },
                {
                    "name": "Generate Submission (Custom Output)",
                    "command": f"{self.get_command_path('scripts/evaluation/evaluate.py')} --config-dir conf",
                    "description": "Generate submission with custom output file name. Specify evaluate.custom_output_file parameter.",
                    "needs_params": True,
                    "params": ["model.alpha", "limit", "evaluate.custom_output_file"],
                },
                {
                    "name": "Quick Generate (JSONL)",
                    "command": f"{self.get_command_path('scripts/evaluation/evaluate.py')} --config-dir conf evaluate.custom_output_file=outputs/quick_submission.jsonl",
                    "description": "Quick generation with JSONL output format. Uses default settings.",
                    "needs_params": True,
                    "params": ["model.alpha", "limit"],
                },
                {
                    "name": "Quick Generate (JSONL - Alternative)",
                    "command": f"{self.get_command_path('scripts/evaluation/evaluate.py')} --config-dir conf evaluate.custom_output_file=outputs/quick_submission_alt.jsonl",
                    "description": "Alternative quick generation with JSONL output format. Uses default settings.",
                    "needs_params": True,
                    "params": ["model.alpha", "limit"],
                },
                {
                    "name": "Create Validation Set",
                    "command": f"{self.get_command_path('scripts/data/create_validation_set.py')} --config-dir conf",
                    "description": "Create new validation dataset",
                    "needs_params": True,
                    "params": ["create_validation_set.sample_size"],
                },
                {
                    "name": "Trim Submission",
                    "command": f"{self.get_command_path('scripts/data/trim_submission.py')}",
                    "description": "Trim submission file content",
                    "needs_params": True,
                    "params": ["input_file", "output_file", "max_length"],
                },
                {
                    "name": "Transform Submission",
                    "command": f"{self.get_command_path('scripts/data/transform_submission.py')}",
                    "description": "Transform submission to evaluation logs",
                    "needs_params": True,
                    "params": ["eval_file", "submission_file", "output_file"],
                },
            ]
        }

    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that evaluation and submission components are properly configured.

        Returns:
            Dict with validation results for different components
        """
        results = {
            "evaluation_scripts_exist": False,
            "data_processing_scripts_exist": False,
            "config_files_exist": False,
            "output_directory_exists": False,
            "openai_available": False,
        }

        # Check evaluation scripts
        evaluation_scripts = [
            "scripts/evaluation/evaluate.py",
        ]
        results["evaluation_scripts_exist"] = all(
            (self.project_root / script).exists() for script in evaluation_scripts
        )

        # Check data processing scripts
        data_scripts = [
            "scripts/data/create_validation_set.py",
            "scripts/data/trim_submission.py",
            "scripts/data/transform_submission.py",
        ]
        results["data_processing_scripts_exist"] = all(
            (self.project_root / script).exists() for script in data_scripts
        )

        # Check configuration files
        config_files = [
            "conf/settings.yaml",
        ]
        results["config_files_exist"] = all(
            (self.project_root / config).exists() for config in config_files
        )

        # Check output directory
        output_dir = self.project_root / "outputs"
        results["output_directory_exists"] = output_dir.exists()

        # Check OpenAI availability (basic check)
        try:
            import openai
            results["openai_available"] = True
        except ImportError:
            results["openai_available"] = False

        return results

    def get_setup_instructions(self) -> str:
        """
        Get setup instructions for evaluation and submission components.

        Returns:
            String with setup instructions
        """
        validation = self.validate_setup()

        instructions = []

        if not validation["evaluation_scripts_exist"]:
            instructions.append("❌ Evaluation scripts missing - check scripts/evaluation/ directory")

        if not validation["data_processing_scripts_exist"]:
            instructions.append("❌ Data processing scripts missing - check scripts/data/ directory")

        if not validation["config_files_exist"]:
            instructions.append("❌ Configuration files missing - check conf/ directory")

        if not validation["output_directory_exists"]:
            instructions.append("⚠️  Output directory missing - create outputs/ directory")
            instructions.append("   Run: mkdir -p outputs")

        if not validation["openai_available"]:
            instructions.append("⚠️  OpenAI package not available - install for OpenAI models")
            instructions.append("   Run: poetry add openai")

        if validation["evaluation_scripts_exist"] and validation["config_files_exist"]:
            instructions.append("✅ Core evaluation components are available!")

        return "\n".join(instructions)

    def get_usage_examples(self) -> List[str]:
        """
        Get usage examples for evaluation commands.

        Returns:
            List of usage examples
        """
        return [
            "# Generate submission with OpenAI",
            "PYTHONPATH=src poetry run python scripts/evaluation/evaluate.py \\",
            "  --config-dir conf model.alpha=0.5 limit=100",
            "",
            "# Generate submission with Qwen2:7b full pipeline",
            "PYTHONPATH=src poetry run python scripts/evaluation/evaluate.py \\",
            "  --config-dir conf pipeline=qwen-full model.alpha=0.5 limit=100",
            "",
            "# Create validation dataset",
            "PYTHONPATH=src poetry run python scripts/data/create_validation_set.py \\",
            "  --config-dir conf create_validation_set.sample_size=200",
            "",
            "# Trim submission file",
            "PYTHONPATH=src poetry run python scripts/data/trim_submission.py \\",
            "  inputs/submission.jsonl outputs/submission_trimmed.jsonl 500",
            "",
            "# Transform submission to evaluation logs",
            "PYTHONPATH=src poetry run python scripts/data/transform_submission.py \\",
            "  data/eval.jsonl outputs/submission.jsonl outputs/evaluation_logs.jsonl",
        ]


# Factory function for easy integration
def get_evaluation_menu(project_root: Path) -> EvaluationMenu:
    """Factory function to create EvaluationMenu instance."""
    return EvaluationMenu(project_root)