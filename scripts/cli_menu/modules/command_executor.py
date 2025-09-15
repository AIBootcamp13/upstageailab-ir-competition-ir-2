#!/usr/bin/env python3
"""
Command Executor for RAG CLI Modular Architecture

This module provides unified command execution functionality for the CLI menu system.
It handles parameter prompting, command execution, and background process management.
"""

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import questionary
from rich.console import Console

console = Console()


class CommandExecutor:
    """Unified command executor for CLI menu operations."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def execute_command(self, command: str, description: str, run_in_background: bool = False) -> bool:
        """
        Execute a command and display results.

        Args:
            command: The command to execute
            description: Description of the command for display
            run_in_background: Whether to run the command in background

        Returns:
            True if command executed successfully, False otherwise
        """
        console.print(f"\n[bold blue]Executing:[/bold blue] {description}")
        console.print(f"[dim]{command}[/dim]\n")

        try:
            # Set PYTHONPATH and other environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "src")

            # Don't change working directory - let Poetry find pyproject.toml in the original location
            # os.chdir(self.project_root)

            if run_in_background:
                return self._run_background_command(command, description, env)
            else:
                return self._run_foreground_command(command, env)

        except Exception as e:
            console.print(f"[red]✗ Error executing command: {e}[/red]")
            return False

    def _run_background_command(self, command: str, description: str, env: Dict[str, str]) -> bool:
        """Run a command in the background."""
        # First, kill any existing processes that might conflict
        try:
            kill_result = subprocess.run(
                "pkill -f streamlit",
                shell=True,
                cwd=self.project_root,
                env=env,
                capture_output=True
            )
            if kill_result.returncode in [0, 1]:  # 0 = processes killed, 1 = no processes found
                console.print("[dim]Stopped any existing conflicting processes[/dim]")
        except Exception:
            pass  # Ignore errors from pkill

        # Create a temporary file to capture output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as temp_file:
            temp_log = temp_file.name

        # Start the command with output redirected to temp file
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self.project_root,
                env=env,
                stdout=open(temp_log, 'w'),
                stderr=subprocess.STDOUT
            )

            # Wait a bit for the command to start up and print URLs
            time.sleep(3)

            # Read and display the captured output
            try:
                with open(temp_log, 'r') as f:
                    output = f.read()
                    if output.strip():
                        console.print("[cyan]Command output:[/cyan]")
                        console.print(output)
                        # Extract and highlight URLs
                        import re
                        url_pattern = r'https?://[^\s]+'
                        urls = re.findall(url_pattern, output)
                        if urls:
                            console.print("[green]🌐 Access URLs:[/green]")
                            for url in urls:
                                console.print(f"  [bold blue]{url}[/bold blue]")
            except Exception as e:
                console.print(f"[yellow]Could not read command output: {e}[/yellow]")

            # Clean up temp file in background
            def cleanup_temp_file():
                time.sleep(1)  # Give process time to fully start
                try:
                    os.unlink(temp_log)
                except:
                    pass
            cleanup_thread = threading.Thread(target=cleanup_temp_file, daemon=True)
            cleanup_thread.start()

            console.print("[green]✓ Command started successfully in background![/green]")
            return True

        except Exception as e:
            console.print(f"[red]✗ Error starting background command: {e}[/red]")
            return False

    def _run_foreground_command(self, command: str, env: Dict[str, str]) -> bool:
        """Run a command in the foreground."""
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

    def get_parameters(self, params: List[str]) -> Dict[str, str]:
        """
        Prompt user for command parameters.

        Args:
            params: List of parameter names to prompt for

        Returns:
            Dict mapping parameter names to values
        """
        param_values = {}

        for param in params:
            if param == "model.alpha":
                value = questionary.text(
                    f"Enter value for {param} (default: 0.0):",
                    default="0.0"
                ).ask()
                if value is None:
                    value = "0.0"
            elif param == "limit":
                value = questionary.text(
                    f"Enter value for {param} (default: 50):",
                    default="50"
                ).ask()
                if value is None:
                    value = "50"
            elif param == "pipeline":
                value = questionary.select(
                    "Select pipeline:",
                    choices=["ollama-full", "ollama", "ollama-llama", "hybrid-ollama"],
                    default="ollama-full"
                ).ask()
                if value is None:
                    value = "ollama-full"
            elif param == "experiment":
                value = questionary.text(
                    f"Enter experiment name (e.g., prompt_tuning):",
                    default="prompt_tuning"
                ).ask()
                if value is None:
                    value = "prompt_tuning"
            elif param == "input_file":
                value = questionary.text(
                    "Enter input file path:",
                    default="outputs/submission.csv"
                ).ask()
                if value is None:
                    value = "outputs/submission.csv"
            elif param == "output_file":
                value = questionary.text(
                    "Enter output file path:",
                    default="outputs/submission_processed.csv"
                ).ask()
                if value is None:
                    value = "outputs/submission_processed.csv"
            elif param == "max_length":
                value = questionary.text(
                    "Enter max length (default: 500):",
                    default="500"
                ).ask()
                if value is None:
                    value = "500"
            elif param == "eval_file":
                value = questionary.text(
                    "Enter eval file path:",
                    default="data/eval.jsonl"
                ).ask()
                if value is None:
                    value = "data/eval.jsonl"
            elif param == "submission_file":
                value = questionary.text(
                    "Enter submission file path:",
                    default="outputs/submission.csv"
                ).ask()
                if value is None:
                    value = "outputs/submission.csv"
            elif param == "create_validation_set.sample_size":
                value = questionary.text(
                    "Enter sample size (default: 100):",
                    default="100"
                ).ask()
                if value is None:
                    value = "100"
            elif param == "pipeline.generator_type":
                value = questionary.select(
                    "Select generator type:",
                    choices=["openai", "ollama"],
                    default="openai"
                ).ask()
                if value is None:
                    value = "openai"
            elif param == "pipeline.generator_model_name":
                value = questionary.text(
                    "Enter generator model name (e.g., gpt-3.5-turbo, qwen2:7b, llama3.1:8b):",
                    default="llama3.1:8b"
                ).ask()
                if value is None:
                    value = "llama3.1:8b"
            else:
                value = questionary.text(f"Enter value for {param}:").ask()
                if value is None:
                    value = ""

            param_values[param] = value

        return param_values

    def build_command_with_params(self, base_command: str, params: Dict[str, str]) -> str:
        """
        Build command string with parameters.

        Args:
            base_command: Base command string
            params: Parameter values

        Returns:
            Complete command string with parameters
        """
        command_parts = [base_command]

        for param, value in params.items():
            # Skip empty values (when user cancels input)
            if not value or value.strip() == "":
                continue

            if param == "create_validation_set.sample_size":
                command_parts.append(f"{param}={value}")
            elif param == "pipeline":
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