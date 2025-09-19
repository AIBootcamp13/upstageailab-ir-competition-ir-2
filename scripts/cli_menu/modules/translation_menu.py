#!/usr/bin/env python3
"""
Translation Menu Module for RAG CLI

This module provides translation-related commands for the CLI menu system.
It integrates various translation methods (Google Translate, Ollama, caching) into
a unified interface for the RAG validation pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from cli_menu.modules import BaseMenuModule

console = Console()


class TranslationMenu(BaseMenuModule):
    """Translation menu commands for CLI integration."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.translation_dir = project_root / "scripts" / "translation"

    def get_commands(self) -> Dict[str, List[Dict]]:
        """
        Get all translation-related commands organized by category.

        Returns:
            Dict containing translation commands for CLI menu integration
        """
        return {
            "Translation": [
                {
                    "name": "Translate Validation Data",
                    "command": f"bash {self.get_command_path('scripts/translation/translate_validation.sh')}",
                    "description": "Translate Korean validation queries to English using cached translation system",
                    "needs_params": False,
                },
                {
                    "name": "Translate Validation (Advanced)",
                    "command": f"{self.get_command_path('scripts/translation/integrate_translation.py')}",
                    "description": "Advanced translation with custom input/output files and caching options",
                    "needs_params": True,
                    "params": ["--input", "--output", "--cache"],
                },
                {
                    "name": "Validate with Translation",
                    "command": f"{self.get_command_path('scripts/translation/validate_with_translation.py')}",
                    "description": "Run validation pipeline with automatic query translation",
                    "needs_params": False,
                },
                {
                    "name": "Translate Documents (Ollama)",
                    "command": f"{self.get_command_path('scripts/translation/translate_documents_ollama.py')}",
                    "description": "Translate documents using local Ollama models (high quality, offline)",
                    "needs_params": True,
                    "params": ["--input", "--output", "--model", "--batch-size"],
                },
                {
                    "name": "Translate Documents (Google)",
                    "command": f"{self.get_command_path('scripts/translation/translate_documents_google.py')}",
                    "description": "Translate documents using Google Translate API (requires API access)",
                    "needs_params": True,
                    "params": ["--input", "--output", "--batch-size"],
                },
                {
                    "name": "Test Translation Setup",
                    "command": f"{self.get_command_path('scripts/translation/test_translation.py')}",
                    "description": "Test translation functionality with sample data",
                    "needs_params": False,
                },
                {
                    "name": "Check Translation Cache",
                    "command": "redis-cli keys 'translation:*' | wc -l",
                    "description": "Check number of cached translations in Redis",
                    "needs_params": False,
                },
                {
                    "name": "Clear Translation Cache",
                    "command": "redis-cli del $(redis-cli keys 'translation:*' | tr '\\n' ' ')",
                    "description": "Clear all cached translations from Redis",
                    "needs_params": False,
                },
                {
                    "name": "Monitor Translation Cache",
                    "command": "redis-cli monitor | grep translation",
                    "description": "Monitor translation cache operations in real-time",
                    "needs_params": False,
                    "run_in_background": True,
                },
            ]
        }

    def validate_translation_setup(self) -> Dict[str, bool]:
        """
        Validate that translation system is properly set up.

        Returns:
            Dict with validation results for different components
        """
        results = {
            "translation_scripts": False,
            "ollama_available": False,
            "googletrans_available": False,
            "redis_available": False,
            "cache_functional": False,
        }

        # Check if translation scripts exist
        required_scripts = [
            "integrate_translation.py",
            "translate_validation.sh",
            "validate_with_translation.py",
            "translate_documents_ollama.py",
            "translate_documents_google.py",
        ]

        results["translation_scripts"] = all(
            (self.translation_dir / script).exists() for script in required_scripts
        )

        # Check Ollama availability
        try:
            import aiohttp
            import asyncio

            async def check_ollama():
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get("http://localhost:11434/api/tags", timeout=5) as resp:
                            return resp.status == 200
                    except:
                        return False

            results["ollama_available"] = asyncio.run(check_ollama())
        except ImportError:
            results["ollama_available"] = False

        # Check Google Translate availability
        try:
            from googletrans import Translator
            results["googletrans_available"] = True
        except ImportError:
            results["googletrans_available"] = False

        # Check Redis availability
        try:
            import redis
            r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            r.ping()
            results["redis_available"] = True

            # Check if cache has any entries
            cache_keys = r.keys("translation:*")
            results["cache_functional"] = isinstance(cache_keys, list) and len(cache_keys) >= 0
        except:
            results["redis_available"] = False
            results["cache_functional"] = False

        return results

    def get_setup_instructions(self) -> str:
        """
        Get setup instructions for translation system.

        Returns:
            String with setup instructions
        """
        validation = self.validate_translation_setup()

        instructions = []

        if not validation["translation_scripts"]:
            instructions.append("❌ Translation scripts missing - check scripts/translation/ directory")

        if not validation["ollama_available"]:
            instructions.append("⚠️  Ollama not available - start Ollama service for local translation")
            instructions.append("   Run: ollama serve")

        if not validation["googletrans_available"]:
            instructions.append("⚠️  Google Translate not available - install googletrans for API translation")
            instructions.append("   Run: poetry add googletrans==4.0.2")

        if not validation["redis_available"]:
            instructions.append("❌ Redis not available - start Redis service for caching")
            instructions.append("   Run: ./scripts/execution/run-local.sh start")

        if validation["translation_scripts"] and validation["redis_available"]:
            instructions.append("✅ Basic translation setup complete")
            if validation["ollama_available"] or validation["googletrans_available"]:
                instructions.append("✅ Translation methods available - ready to use")

        return "\n".join(instructions)

    def get_usage_examples(self) -> List[str]:
        """
        Get usage examples for translation commands.

        Returns:
            List of usage examples
        """
        return [
            "# Quick validation translation",
            "bash scripts/translation/translate_validation.sh",
            "",
            "# Advanced translation with custom files",
            "poetry run python scripts/translation/integrate_translation.py \\",
            "  --input data/validation_balanced.jsonl \\",
            "  --output data/validation_balanced_en.jsonl \\",
            "  --cache",
            "",
            "# Run validation with automatic translation",
            "poetry run python scripts/translation/validate_with_translation.py",
            "",
            "# Translate documents with Ollama",
            "poetry run python scripts/translation/translate_documents_ollama.py \\",
            "  --input data/documents.jsonl \\",
            "  --output data/documents_en.jsonl \\",
            "  --model qwen2:7b \\",
            "  --batch-size 10",
        ]


# Factory function for easy integration
def get_translation_menu(project_root: Path) -> TranslationMenu:
    """Factory function to create TranslationMenu instance."""
    return TranslationMenu(project_root)


# CLI integration helper
def integrate_translation_menu(commands_dict: Dict[str, List[Dict]], project_root: Path) -> Dict[str, List[Dict]]:
    """
    Integrate translation commands into existing CLI menu structure.

    Args:
        commands_dict: Existing commands dictionary
        project_root: Path to project root

    Returns:
        Updated commands dictionary with translation commands
    """
    translation_menu = get_translation_menu(project_root)
    translation_commands = translation_menu.get_commands()

    # Merge translation commands into existing structure
    for category, cmds in translation_commands.items():
        if category in commands_dict:
            commands_dict[category].extend(cmds)
        else:
            commands_dict[category] = cmds

    return commands_dict