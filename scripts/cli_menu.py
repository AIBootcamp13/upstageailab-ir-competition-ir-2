#!/usr/bin/env python3
"""
Interactive CLI Menu for RAG Project Ope                {
                    "name": "Reinde                {
                    "name": "Generate Submission",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/evaluation/evaluate.py",
                    "description": "Generate submission file for evaluation",
                    "needs_params": True,
                    "params": ["model.alpha"],
                },
                {
                    "name": "Create Validation Set",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/data/create_validation_set.py",
                    "description": "Create new validation dataset",
                    "needs_params": True,
                    "params": ["create_validation_set.sample_size"],
                },                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/maintenance/reindex.py",
                    "description": "Reindex documents to Elasticsearch",
                    "needs_params": False,
                },s

This script provides a user-friendly interface to execute common project commands
with colored output and interactive prompts for parameters.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


console = Console()


class CLIMenu:
    """Interactive CLI menu for RAG project operations."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.commands = self._get_commands()

    def _get_commands(self) -> Dict[str, List[Dict]]:
        """Get all available commands organized by category."""
        return {
            "Setup & Infrastructure": [
                {
                    "name": "Install Dependencies",
                    "command": "poetry install",
                    "description": "Install all project dependencies using Poetry",
                    "needs_params": False,
                },
                {
                    "name": "Setup Environment",
                    "command": "cp .env.example .env",
                    "description": "Copy environment template file",
                    "needs_params": False,
                },
                {
                    "name": "Start Local Services",
                    "command": "./scripts/execution/run-local.sh start",
                    "description": "Start Elasticsearch and Redis locally",
                    "needs_params": False,
                },
                {
                    "name": "Check Service Status",
                    "command": "./scripts/execution/run-local.sh status",
                    "description": "Check status of local services",
                    "needs_params": False,
                },
                {
                    "name": "Stop Local Services",
                    "command": "./scripts/execution/run-local.sh stop",
                    "description": "Stop local Elasticsearch and Redis",
                    "needs_params": False,
                },
            ],
            "Data Management": [
                {
                    "name": "Reindex Documents",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/maintenance/reindex.py data/documents.jsonl",
                    "description": "Reindex documents to Elasticsearch",
                    "needs_params": False,
                },
                {
                    "name": "Analyze Data",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/data/analyze_data.py",
                    "description": "Analyze document datasets for statistics",
                    "needs_params": False,
                },
                {
                    "name": "Check Duplicates",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/data/check_duplicates.py",
                    "description": "Detect duplicate entries in datasets",
                    "needs_params": False,
                },
            ],
            "Experiments & Validation": [
                {
                    "name": "Validate Retrieval (Basic)",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/evaluation/validate_retrieval.py --config-dir conf",
                    "description": "Run basic retrieval validation",
                    "needs_params": False,
                },
                {
                    "name": "Validate Retrieval (Ollama)",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/evaluation/validate_retrieval.py --config-dir conf pipeline=ollama",
                    "description": "Run retrieval validation using local Ollama model",
                    "needs_params": True,
                    "params": ["model.alpha", "limit", "pipeline.generator_model_name"],
                },
                {
                    "name": "Validate Retrieval (Custom)",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/evaluation/validate_retrieval.py --config-dir conf",
                    "description": "Run retrieval validation with custom parameters",
                    "needs_params": True,
                    "params": ["model.alpha", "limit", "experiment", "pipeline.generator_type", "pipeline.generator_model_name"],
                },
                {
                    "name": "Multi-Run Experiments",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/evaluation/validate_retrieval.py --config-dir conf --multirun",
                    "description": "Run multiple experiments in parallel",
                    "needs_params": True,
                    "params": ["experiment", "limit"],
                },
                {
                    "name": "Domain Classification Validation",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/evaluation/validate_domain_classification.py --config-dir conf",
                    "description": "Validate domain classification accuracy",
                    "needs_params": False,
                },
            ],
            "Evaluation & Submission": [
                {
                    "name": "Generate Submission",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/evaluation/evaluate.py --config-dir conf",
                    "description": "Generate submission file for evaluation",
                    "needs_params": True,
                    "params": ["model.alpha"],
                },
                {
                    "name": "Create Validation Set",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/data/create_validation_set.py --config-dir conf",
                    "description": "Create new validation dataset",
                    "needs_params": True,
                    "params": ["create_validation_set.sample_size"],
                },
                {
                    "name": "Trim Submission",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/data/trim_submission.py",
                    "description": "Trim submission file content",
                    "needs_params": True,
                    "params": ["input_file", "output_file", "max_length"],
                },
                {
                    "name": "Transform Submission",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/data/transform_submission.py",
                    "description": "Transform submission to evaluation logs",
                    "needs_params": True,
                    "params": ["eval_file", "submission_file", "output_file"],
                },
            ],
            "Utilities": [
                {
                    "name": "Run Smoke Tests",
                    "command": f"PYTHONPATH={self.project_root}/src poetry run python scripts/evaluation/smoke_test.py",
                    "description": "Run smoke tests to verify system health",
                    "needs_params": False,
                },
                {
                    "name": "List All Scripts",
                    "command": "python scripts/list_scripts.py",
                    "description": "List all available scripts with descriptions",
                    "needs_params": False,
                },
                {
                    "name": "Clean Distributions",
                    "command": "./scripts/infra/cleanup-distros.sh",
                    "description": "Clean up downloaded service distributions",
                    "needs_params": False,
                },
            ],
        }

    def _run_command(self, command: str, description: str) -> bool:
        """Execute a command and display results."""
        console.print(f"\n[bold blue]Executing:[/bold blue] {description}")
        console.print(f"[dim]{command}[/dim]\n")

        try:
            # Change to project root directory
            os.chdir(self.project_root)

            # Set PYTHONPATH and other environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "src")

            # Run the command
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                env=env
            )

            if result.returncode == 0:
                console.print("[green]✓ Command completed successfully![/green]")
                return True
            else:
                console.print("[red]✗ Command failed![/red]")
                return False

        except Exception as e:
            console.print(f"[red]✗ Error executing command: {e}[/red]")
            return False

    def _get_parameters(self, params: List[str]) -> Dict[str, str]:
        """Prompt user for command parameters."""
        param_values = {}

        for param in params:
            if param == "model.alpha":
                value = questionary.text(
                    f"Enter value for {param} (default: 0.7):",
                    default="0.7"
                ).ask()
            elif param == "limit":
                value = questionary.text(
                    f"Enter value for {param} (default: 50):",
                    default="50"
                ).ask()
            elif param == "experiment":
                value = questionary.text(
                    f"Enter experiment name (e.g., prompt_tuning):",
                    default="prompt_tuning"
                ).ask()
            elif param == "input_file":
                value = questionary.text(
                    "Enter input file path:",
                    default="outputs/submission.csv"
                ).ask()
            elif param == "output_file":
                value = questionary.text(
                    "Enter output file path:",
                    default="outputs/submission_processed.csv"
                ).ask()
            elif param == "max_length":
                value = questionary.text(
                    "Enter max length (default: 500):",
                    default="500"
                ).ask()
            elif param == "eval_file":
                value = questionary.text(
                    "Enter eval file path:",
                    default="data/eval.jsonl"
                ).ask()
            elif param == "submission_file":
                value = questionary.text(
                    "Enter submission file path:",
                    default="outputs/submission.csv"
                ).ask()
            elif param == "create_validation_set.sample_size":
                value = questionary.text(
                    "Enter sample size (default: 100):",
                    default="100"
                ).ask()
            elif param == "pipeline.generator_type":
                value = questionary.select(
                    "Select generator type:",
                    choices=["openai", "ollama"],
                    default="openai"
                ).ask()
            elif param == "pipeline.generator_model_name":
                value = questionary.text(
                    "Enter generator model name (e.g., gpt-3.5-turbo, qwen2:7b, llama3.1:8b):",
                    default="qwen2:7b"
                ).ask()
            else:
                value = questionary.text(f"Enter value for {param}:").ask()

            param_values[param] = value

        return param_values

    def _build_command_with_params(self, base_command: str, params: Dict[str, str]) -> str:
        """Build command string with parameters."""
        command_parts = [base_command]

        for param, value in params.items():
            if param == "create_validation_set.sample_size":
                command_parts.append(f"{param}={value}")
            elif param in ["input_file", "output_file", "eval_file", "submission_file"]:
                # These are positional arguments, not Hydra parameters
                continue
            else:
                command_parts.append(f"{param}={value}")

        # Handle positional arguments for specific commands
        if "trim_submission.py" in base_command:
            input_file = params.get("input_file", "outputs/submission.csv")
            output_file = params.get("output_file", "outputs/submission_trimmed.csv")
            max_length = params.get("max_length", "500")
            command_parts.extend([input_file, output_file, max_length])
        elif "transform_submission.py" in base_command:
            eval_file = params.get("eval_file", "data/eval.jsonl")
            submission_file = params.get("submission_file", "outputs/submission.csv")
            output_file = params.get("output_file", "outputs/evaluation_logs.jsonl")
            command_parts.extend([eval_file, submission_file, output_file])

        return " ".join(command_parts)

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
                        params = self._get_parameters(selected_cmd.get("params", []))
                        command = self._build_command_with_params(
                            selected_cmd["command"], params
                        )
                    else:
                        command = selected_cmd["command"]

                    # Execute command
                    success = self._run_command(command, selected_cmd["description"])

                    if success:
                        console.print("\n[green]✓ Command completed successfully![/green]")
                    else:
                        console.print("\n[red]✗ Command failed. Check the output above.[/red]")

                    # Wait for user to continue
                    questionary.press_any_key_to_continue().ask()
                else:
                    console.print("[red]Invalid option. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")

    def run(self):
        """Run the main CLI menu."""
        console.print(Panel.fit(
            "[bold green]🚀 RAG Project CLI Menu[/bold green]\n\n"
            "Welcome to the interactive CLI for managing your RAG project!\n"
            "Select a category to explore available commands.",
            title="RAG CLI"
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
        menu = CLIMenu()
        menu.run()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]An error occurred: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
