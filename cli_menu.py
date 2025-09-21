#!/usr/bin/env python3
"""
Modular CLI Menu for RAG Project Operations

This script provides a user-friendly interface to execute common project commands
using a modular architecture with separate menu modules for different functionalities.
"""

from cli_menu.modules.command_executor import CommandExecutor
from cli_menu.modules import MenuBuilder
import os
import sys
from pathlib import Path
from typing import Dict, List

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import modular components
sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent))

console = Console()


class ModularCLIMenu:
    """Modular CLI menu for RAG project operations."""

    def __init__(self):
        self.project_root = Path(os.path.realpath(__file__)).parent
        self.executor = CommandExecutor(self.project_root)
        self._commands = None  # Lazy load commands

    @property
    def commands(self):
        """Lazy load commands only when accessed."""
        if self._commands is None:
            self._commands = self._build_commands()
        return self._commands

    def _build_commands(self) -> dict:
        """Build the complete command structure using MenuBuilder."""
        # Lazy import menu modules
        from cli_menu.modules.configuration_menu import get_configuration_menu
        from cli_menu.modules.translation_menu import integrate_translation_menu
        from cli_menu.modules.utilities_menu import get_utilities_menu
        from cli_menu.modules.evaluation_menu import get_evaluation_menu
        from cli_menu.modules.experiments_menu import get_experiments_menu
        from cli_menu.modules.data_management_menu import get_data_management_menu
        from cli_menu.modules.setup_menu import get_setup_menu

        builder = MenuBuilder(self.project_root)

        # Add all menu modules
        builder.add_module(get_setup_menu)
        builder.add_module(get_data_management_menu)
        builder.add_module(get_experiments_menu)
        builder.add_module(get_evaluation_menu)
        builder.add_module(get_utilities_menu)
        builder.add_module(get_configuration_menu)

        commands = builder.build()
        integrate_translation_menu(commands, self.project_root)
        return commands

    def run_command(self, command_name: str, *args, **kwargs):
        """Run a command by its name."""
        command = self.commands.get(command_name)
        if command:
            return command.execute(*args, **kwargs)
        else:
            console.print(f"[red]Command '{command_name}' not found![/red]")

    def show_menu(self):
        """Display the modular CLI menu."""
        table = Table(title="Modular CLI Menu", show_header=True,
                      header_style="bold magenta")
        table.add_column("Module", style="dim")
        table.add_column("Description")

        for command_name, command in self.commands.items():
            table.add_row(command_name, command.description)

        console.print(
            Panel(table, title="Available Commands", border_style="cyan"))


class Menu:
    """Interactive menu system for CLI operations."""

    def __init__(self, cli_menu: 'ModularCLIMenu'):
        self.cli_menu = cli_menu

    def run(self):
        """Run the interactive menu loop."""
        while True:
            try:
                if not self._show_main_menu():
                    break
            except KeyboardInterrupt:
                console.print(
                    "\n[bold yellow]Interrupted by user.[/bold yellow]")
                break
            except Exception as e:
                console.print(f"[red]Unexpected error: {str(e)}[/red]")
                console.print("[dim]Returning to main menu...[/dim]")
                continue

    def _show_main_menu(self) -> bool:
        """Show main category menu. Returns False to exit."""
        try:
            categories = list(self.cli_menu.commands.keys()) + ["Exit"]

            if not self.cli_menu.commands:
                console.print(
                    "[red]No commands available. Check menu configuration.[/red]")
                return False

            category = questionary.select(
                "Select a category:",
                choices=categories,
                use_shortcuts=True,
            ).ask()

            if category is None:
                return False  # User cancelled
            elif category == "Exit":
                console.print("[bold]Exiting the CLI Menu.[/bold]")
                return False

            return self._show_category_menu(category)

        except Exception as e:
            console.print(f"[red]Error displaying main menu: {str(e)}[/red]")
            return False

    def _show_category_menu(self, category: str) -> bool:
        """Show commands for a category. Returns True to continue."""
        try:
            if category not in self.cli_menu.commands:
                console.print(f"[red]Category '{category}' not found.[/red]")
                return True

            commands = self.cli_menu.commands[category]
            if not commands:
                console.print(
                    f"[yellow]No commands available in category '{category}'.[/yellow]")
                return True

            command_names = [cmd["name"]
                             for cmd in commands] + ["Exit to Main Menu"]

            command_name = questionary.select(
                f"Select a command from {category}:",
                choices=command_names,
                use_shortcuts=True,
            ).ask()

            if command_name is None:
                return True  # User cancelled, back to main
            elif command_name == "Exit to Main Menu":
                return True

            # Find and execute the command
            command = next(
                (cmd for cmd in commands if cmd["name"] == command_name), None)
            if command is None:
                console.print(
                    f"[red]Command '{command_name}' not found in category.[/red]")
                return True

            self._execute_command_with_feedback(command)

            # Ask if user wants to continue in this category or go back
            return self._ask_continue_in_category(category)

        except Exception as e:
            console.print(f"[red]Error in category menu: {str(e)}[/red]")
            return True

    def _execute_command_with_feedback(self, command: dict):
        """Execute a command with proper feedback and error handling."""
        try:
            # Check if command needs confirmation
            if self._command_needs_confirmation(command):
                if not self._confirm_command_execution(command):
                    console.print(
                        "[yellow]Command execution cancelled.[/yellow]")
                    return

            success = self.cli_menu.executor.execute_command_dict(command)
            if success:
                console.print(
                    "[green]✓ Command completed successfully![/green]")
            else:
                console.print("[red]✗ Command failed![/red]")
        except Exception as e:
            console.print(f"[red]✗ Error executing command: {str(e)}[/red]")

    def _command_needs_confirmation(self, command: dict) -> bool:
        """Check if a command needs user confirmation before execution."""
        destructive_keywords = [
            'delete', 'remove', 'drop', 'clear', 'reset', 'stop', 'kill',
            'uninstall', 'purge', 'format', 'wipe'
        ]

        # Commands that are read-only and safe
        safe_command_names = [
            'analyze data', 'run smoke test', 'check service status',
            'list all scripts', 'show current configuration',
            'check current configuration', 'validate retrieval (openai)', 'validate retrieval (qwen2:7b full)',
            'validate retrieval (llama3.1:8b full)', 'validate retrieval (ollama hybrid)',
            'validate retrieval (custom)', 'multi-run experiments', 'test huggingface integration'
        ]

        command_name = command.get("name", "").lower()
        if command_name in safe_command_names:
            return False

        command_text = (command.get("command", "") + " " +
                        command.get("description", "")).lower()
        return any(keyword in command_text for keyword in destructive_keywords)

    def _confirm_command_execution(self, command: dict) -> bool:
        """Ask user to confirm execution of a command."""
        console.print(
            f"[bold yellow]⚠️  This command may be destructive:[/bold yellow]")
        console.print(
            f"[yellow]Name: {command.get('name', 'Unknown')}[/yellow]")
        console.print(
            f"[yellow]Description: {command.get('description', 'No description')}[/yellow]")
        console.print(
            f"[yellow]Command: {command.get('command', 'Unknown')}[/yellow]")
        console.print()

        confirmed = questionary.confirm(
            "Are you sure you want to execute this command?",
            default=False,
        ).ask()

        return confirmed if confirmed is not None else False

    def _ask_continue_in_category(self, category: str) -> bool:
        """Ask user if they want to continue in the current category or go back."""
        try:
            choice = questionary.select(
                f"Do you want to continue in the '{category}' category?",
                choices=["Continue in this category", "Exit to Main Menu"],
                use_shortcuts=True,
            ).ask()

            if choice is None:
                return False  # User cancelled, go back
            elif choice == "Continue in this category":
                return True
            else:
                return False
        except Exception as e:
            console.print(f"[red]Error asking to continue: {str(e)}[/red]")
            return False


def main():
    """Main entry point for the CLI menu."""
    try:
        menu = ModularCLIMenu()
        console.print(
            "[bold blue]Welcome to the RAG Project CLI Menu![/bold blue]")
        console.print(
            "[dim]Use arrow keys to navigate, Enter to select, Ctrl+C to exit[/dim]\n")

        menu_interface = Menu(menu)
        menu_interface.run()

    except KeyboardInterrupt:
        console.print("\n[bold yellow]CLI Menu interrupted.[/bold yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
