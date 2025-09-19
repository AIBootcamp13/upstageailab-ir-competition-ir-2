#!/usr/bin/env python3
"""
Modular CLI Menu for RAG Project Operations

This script provides a user-friendly interface to execute common project commands
using a modular architecture with separate menu modules for different functionalities.
"""

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
from cli_menu.modules import MenuBuilder
from cli_menu.modules.command_executor import CommandExecutor
from cli_menu.modules.setup_menu import get_setup_menu
from cli_menu.modules.data_management_menu import get_data_management_menu
from cli_menu.modules.experiments_menu import get_experiments_menu
from cli_menu.modules.evaluation_menu import get_evaluation_menu
from cli_menu.modules.utilities_menu import get_utilities_menu
from cli_menu.modules.translation_menu import integrate_translation_menu
from cli_menu.modules.configuration_menu import get_configuration_menu

console = Console()


class ModularCLIMenu:
    """Modular CLI menu for RAG project operations."""

    def __init__(self):
        self.project_root = Path(os.path.realpath(__file__)).parent.parent
        self.executor = CommandExecutor(self.project_root)
        self.commands = self._build_commands()

    def _build_commands(self) -> dict:
        """Build the complete command structure using MenuBuilder."""
        builder = MenuBuilder(self.project_root)

        # Add all menu modules
        builder.add_module(get_setup_menu)
        builder.add_module(get_data_management_menu)
        builder.add_module(get_experiments_menu)
        builder.add_module(get_evaluation_menu)
        builder.add_module(get_utilities_menu)
        builder.add_module(get_configuration_menu)

        # Build base commands
        commands = builder.build()

        # Integrate translation commands (legacy integration)
        commands = integrate_translation_menu(commands, self.project_root)

        return commands

    def _show_category_menu(self, category: str, commands: List[Dict]):
        """Display menu for a specific category."""
        while True:
            # Create table for commands
            table = Table(title=f"[bold cyan]{category}[/bold cyan]")
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Command", style="magenta")
            table.add_column("Description", style="white")

            for i, cmd in enumerate(commands, 1):
                table.add_row(str(i), cmd["name"], cmd["description"])

            console.print(table)

            # Add back option
            console.print(f"\n[dim]Enter 0 to go back to main menu[/dim]")

            choice = questionary.text("Select an option:").ask()

            if choice == "0" or choice is None:
                break

            if not choice.strip():
                console.print("[yellow]Please enter a valid option.[/yellow]")
                continue

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(commands):
                    selected_cmd = commands[idx]

                    # Get parameters if needed
                    if selected_cmd.get("needs_params", False):
                        params = self.executor.get_parameters(selected_cmd.get("params", []))
                        command = self.executor.build_command_with_params(
                            selected_cmd["command"], params
                        )
                    else:
                        command = selected_cmd["command"]

                    # Execute command
                    run_in_background = selected_cmd.get("run_in_background", False)
                    success = self.executor.execute_command(command, selected_cmd["description"], run_in_background)

                    if success:
                        if run_in_background:
                            console.print("\n[green]✓ Command started successfully in background![/green]")
                        else:
                            console.print("\n[green]✓ Command completed successfully![/green]")
                    else:
                        console.print("\n[red]✗ Command failed. Check the output above.[/red]")

                    # Wait for user to continue (only for non-background commands)
                    if not run_in_background:
                        questionary.press_any_key_to_continue().ask()
                else:
                    console.print("[red]Invalid option. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")

    def run(self):
        """Run the main CLI menu."""
        console.print(Panel.fit(
            "[bold green]🚀 Modular RAG Project CLI Menu[/bold green]\n\n"
            "Welcome to the modular CLI for managing your RAG project!\n"
            "Select a category to explore available commands.",
            title="Modular RAG CLI"
        ))

        while True:
            # Main menu
            categories = list(self.commands.keys())

            # Create main menu table
            table = Table(title="[bold cyan]Main Menu[/bold cyan]")
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Category", style="magenta")
            table.add_column("Commands", style="white")

            for i, category in enumerate(categories, 1):
                cmd_count = len(self.commands[category])
                table.add_row(str(i), category, f"{cmd_count} commands")

            console.print(table)
            console.print(f"\n[dim]Enter 0 to exit[/dim]")

            choice = questionary.text("Select a category:").ask()

            if choice == "0" or choice is None:
                console.print("\n[bold green]Goodbye! 👋[/bold green]")
                break

            if not choice.strip():
                console.print("[yellow]Please enter a valid option.[/yellow]")
                continue

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(categories):
                    category = categories[idx]
                    self._show_category_menu(category, self.commands[category])
                else:
                    console.print("[red]Invalid option. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")


def main():
    """Main entry point."""
    try:
        menu = ModularCLIMenu()
        menu.run()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]An error occurred: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()