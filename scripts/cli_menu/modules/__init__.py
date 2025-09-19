#!/usr/bin/env python3
"""
Base Menu Classes for RAG CLI Modular Architecture

This module provides the foundation classes and interfaces for the modular CLI menu system.
All menu modules should inherit from these base classes to ensure consistency.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

console = Console()


class BaseMenuModule(ABC):
    """Abstract base class for all CLI menu modules."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    @abstractmethod
    def get_commands(self) -> Dict[str, List[Dict]]:
        """
        Get all commands provided by this module.

        Returns:
            Dict containing commands organized by category
        """
        pass

    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that the module's dependencies are properly set up.

        Returns:
            Dict with validation results for different components
        """
        return {}

    def get_setup_instructions(self) -> str:
        """
        Get setup instructions for this module.

        Returns:
            String with setup instructions
        """
        return "No setup instructions available."

    def get_command_path(self, script_path: str) -> str:
        """
        Get the full command path for a script.

        Args:
            script_path: Relative path to the script from project root

        Returns:
            Full command string with correct paths
        """
        script_full_path = self.project_root / script_path
        return f"PYTHONPATH={self.project_root}/src poetry run python {script_full_path}"


class CommandRegistry:
    """Registry for managing CLI commands from multiple modules."""

    def __init__(self):
        self._modules: List[BaseMenuModule] = []
        self._commands: Dict[str, List[Dict]] = {}

    def register_module(self, module: BaseMenuModule):
        """Register a menu module."""
        self._modules.append(module)

    def get_all_commands(self) -> Dict[str, List[Dict]]:
        """
        Get all commands from registered modules.

        Returns:
            Dict containing all commands organized by category
        """
        # Reset commands
        self._commands = {}

        # Collect commands from all modules
        for module in self._modules:
            module_commands = module.get_commands()
            for category, commands in module_commands.items():
                if category not in self._commands:
                    self._commands[category] = []
                self._commands[category].extend(commands)

        return self._commands

    def get_validation_summary(self) -> Dict[str, Dict[str, bool]]:
        """
        Get validation summary for all registered modules.

        Returns:
            Dict with module names as keys and validation results as values
        """
        summary = {}
        for module in self._modules:
            module_name = module.__class__.__name__.replace('Menu', '').lower()
            summary[module_name] = module.validate_setup()
        return summary

    def get_setup_instructions(self) -> Dict[str, str]:
        """
        Get setup instructions for all registered modules.

        Returns:
            Dict with module names as keys and setup instructions as values
        """
        instructions = {}
        for module in self._modules:
            module_name = module.__class__.__name__.replace('Menu', '').lower()
            instructions[module_name] = module.get_setup_instructions()
        return instructions


class MenuBuilder:
    """Builder class for constructing CLI menus dynamically."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.registry = CommandRegistry()

    def add_module(self, module_class):
        """Add a menu module to the builder."""
        module_instance = module_class(self.project_root)
        self.registry.register_module(module_instance)
        return self

    def build(self) -> Dict[str, List[Dict]]:
        """Build the complete command structure."""
        return self.registry.get_all_commands()

    def get_validation_summary(self) -> Dict[str, Dict[str, bool]]:
        """Get validation summary for all modules."""
        return self.registry.get_validation_summary()

    def get_setup_instructions(self) -> Dict[str, str]:
        """Get setup instructions for all modules."""
        return self.registry.get_setup_instructions()


# Export key classes for use by other modules
__all__ = [
    'BaseMenuModule',
    'CommandRegistry',
    'MenuBuilder',
]