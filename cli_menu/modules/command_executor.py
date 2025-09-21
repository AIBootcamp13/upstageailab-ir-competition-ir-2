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

# Import settings for centralized configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
# from ir_core.config import settings  # Lazy import


def get_settings():
    """Lazy import of settings to avoid loading models on CLI startup."""
    from ir_core.config import settings
    return settings


console = Console()


class CommandExecutor:
    """Unified command executor for CLI menu operations."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def execute_command(self, command: str, description: str, run_in_background: bool = False, timeout: int = 300) -> bool:
        """
        Execute a command and display results.

        Args:
            command: The command to execute
            description: Description of the command for display
            run_in_background: Whether to run the command in background
            timeout: Timeout in seconds for foreground commands

        Returns:
            True if command executed successfully, False otherwise
        """
        if not command or not command.strip():
            console.print("[red]✗ Error: Empty command[/red]")
            return False

        console.print(f"\n[bold blue]Executing:[/bold blue] {description}")
        console.print(f"[dim]{command}[/dim]\n")

        try:
            # Validate command safety (basic check)
            if self._is_potentially_dangerous_command(command):
                if not self._confirm_dangerous_command(command, description):
                    console.print(
                        "[yellow]Command execution cancelled by user.[/yellow]")
                    return False

            # Set PYTHONPATH and other environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "src")

            if run_in_background:
                return self._run_background_command(command, description, env)
            else:
                return self._run_foreground_command(command, env, timeout)

        except subprocess.TimeoutExpired:
            console.print(
                f"[red]✗ Command timed out after {timeout} seconds[/red]")
            return False
        except FileNotFoundError as e:
            console.print(f"[red]✗ Command not found: {e.filename}[/red]")
            return False
        except PermissionError:
            console.print("[red]✗ Permission denied executing command[/red]")
            return False
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
            # 0 = processes killed, 1 = no processes found
            if kill_result.returncode in [0, 1]:
                console.print(
                    "[dim]Stopped any existing conflicting processes[/dim]")
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
                                console.print(
                                    f"  [bold blue]{url}[/bold blue]")
            except Exception as e:
                console.print(
                    f"[yellow]Could not read command output: {e}[/yellow]")

            # Clean up temp file in background
            def cleanup_temp_file():
                time.sleep(1)  # Give process time to fully start
                try:
                    os.unlink(temp_log)
                except:
                    pass
            cleanup_thread = threading.Thread(
                target=cleanup_temp_file, daemon=True)
            cleanup_thread.start()

            console.print(
                "[green]✓ Command started successfully in background![/green]")
            return True

        except Exception as e:
            console.print(
                f"[red]✗ Error starting background command: {e}[/red]")
            return False

    def _run_foreground_command(self, command: str, env: Dict[str, str], timeout: int = 300) -> bool:
        """Run a command in the foreground with timeout."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                env=env,
                timeout=timeout,
                capture_output=False,  # Let output go to console
                text=True
            )

            if result.returncode == 0:
                console.print(
                    "[green]✓ Command completed successfully![/green]")
                return True
            else:
                console.print(
                    f"[red]✗ Command failed with exit code {result.returncode}![/red]")
                return False
        except subprocess.TimeoutExpired:
            console.print(
                f"[red]✗ Command timed out after {timeout} seconds![/red]")
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
                value = input(
                    f"Enter value for {param} (default: 0.0): ").strip() or "0.0"
            elif param == "limit":
                value = questionary.text(
                    f"Enter value for {param} (default: 50):",
                    default="50"
                ).ask()
                if value is None:
                    value = "50"
            elif param == "pipeline":
                print(
                    "Select pipeline: ollama-full, ollama, ollama-llama, hybrid-ollama")
                value = input(
                    "Enter pipeline (default: ollama-full): ").strip() or "ollama-full"
            elif param == "experiment":
                value = input(
                    f"Enter experiment name (e.g., prompt_tuning): ").strip() or "prompt_tuning"
            elif param == "input_file":
                # Use centralized output path as default
                default_input = getattr(get_settings(), 'data', {}).get(
                    'output_path', 'outputs/submission.jsonl')
                value = input(
                    f"Enter input file path (default: {default_input}): ").strip() or default_input
            elif param == "output_file":
                value = input("Enter output file path (default: outputs/submission_processed.jsonl): ").strip(
                ) or "outputs/submission_processed.jsonl"
            elif param == "max_length":
                value = questionary.text(
                    "Enter max length (default: 500):",
                    default="500"
                ).ask()
                if value is None:
                    value = "500"
            elif param == "eval_file":
                # Use centralized evaluation path as default
                default_eval = getattr(get_settings(), 'data', {}).get(
                    'evaluation_path', 'data/eval.jsonl')
                value = questionary.text(
                    "Enter eval file path:",
                    default=default_eval
                ).ask()
                if value is None:
                    value = default_eval
            elif param == "submission_file":
                # Use centralized output path as default
                default_submission = getattr(get_settings(), 'data', {}).get(
                    'output_path', 'outputs/submission.jsonl')
                value = input(f"Enter submission file path (default: {default_submission}): ").strip(
                ) or default_submission
            elif param == "create_validation_set.sample_size":
                value = input(
                    "Enter sample size (default: 100): ").strip() or "100"
            elif param == "pipeline.generator_type":
                print("Select generator type: openai, ollama")
                value = input(
                    "Enter generator type (default: openai): ").strip() or "openai"
            elif param == "pipeline.generator_model_name":
                value = questionary.text(
                    "Enter generator model name (e.g., gpt-3.5-turbo, qwen2:7b, llama3.1:8b):",
                    default="llama3.1:8b"
                ).ask()
                if value is None:
                    value = "llama3.1:8b"
            elif param == "data_file":
                print("Select data file: data/documents_ko.jsonl, data/documents_bilingual.jsonl, data/documents_ko_with_metadata.jsonl")
                value = input(
                    "Enter data file (default: data/documents_ko.jsonl): ").strip() or "data/documents_ko.jsonl"
            elif param == "evaluate.custom_output_file":
                # Generate timestamp-based default filename
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                default_output = f"outputs/submission_qwen2_7b_{timestamp}.jsonl"
                value = questionary.text(
                    f"Enter output file path (default: {default_output}):",
                    default=default_output
                ).ask()
                if value is None or value.strip() == "":
                    value = default_output
            elif param == "qwen_submission_output_file":
                # Generate timestamp-based default filename for Qwen2:7b submissions
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                default_output = f"outputs/submission_qwen2_7b_full_{timestamp}.jsonl"
                console.print(
                    f"\n[bold cyan]📝 Generate Submission (Qwen2:7b Full)[/bold cyan]")
                console.print(
                    f"[dim]This will generate a submission file using Qwen2:7b for all pipeline stages.[/dim]")
                value = questionary.text(
                    f"Enter submission file name (press Enter for default: {default_output}):",
                    default=default_output
                ).ask()
                if value is None or value.strip() == "":
                    value = default_output
                    console.print(
                        f"[green]✓ Using default filename: {value}[/green]")
                else:
                    console.print(
                        f"[green]✓ Using custom filename: {value}[/green]")
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
            elif param == "qwen_submission_output_file":
                # Convert to the expected parameter name for the evaluation script
                command_parts.append(f"evaluate.custom_output_file={value}")
            elif param in ["input_file", "output_file", "eval_file", "submission_file"]:
                # These are positional arguments, not Hydra parameters
                continue
            else:
                command_parts.append(f"{param}={value}")

        # Handle positional arguments for specific commands
        if "trim_submission.py" in base_command:
            input_file = params.get("input_file", getattr(get_settings(), 'data', {}).get(
                'output_path', 'outputs/submission.jsonl')) or getattr(get_settings(), 'data', {}).get('output_path', 'outputs/submission.jsonl')
            output_file = params.get(
                "output_file", "outputs/submission_trimmed.jsonl") or "outputs/submission_trimmed.jsonl"
            max_length = params.get("max_length", "500") or "500"
            command_parts.extend([input_file, output_file, max_length])
        elif "transform_submission.py" in base_command:
            eval_file = params.get("eval_file", getattr(get_settings(), 'data', {}).get(
                'evaluation_path', 'data/eval.jsonl')) or getattr(get_settings(), 'data', {}).get('evaluation_path', 'data/eval.jsonl')
            submission_file = params.get("submission_file", getattr(get_settings(), 'data', {}).get(
                'output_path', 'outputs/submission.jsonl')) or getattr(get_settings(), 'data', {}).get('output_path', 'outputs/submission.jsonl')
            output_file = params.get(
                "output_file", "outputs/evaluation_logs.jsonl") or "outputs/evaluation_logs.jsonl"
            command_parts.extend([eval_file, submission_file, output_file])

        return " ".join(command_parts)

    def execute_command_dict(self, command_dict: Dict) -> bool:
        """
        Execute a command from a command dictionary, handling parameters if needed.

        Args:
            command_dict: Dictionary containing command details

        Returns:
            True if successful
        """
        command = command_dict["command"]
        description = command_dict["description"]

        if command_dict.get("needs_params", False):
            # Prompt for parameters
            params = self._prompt_for_params(command_dict.get("params", []))
            command = self._build_command_with_params(command, params)

        return self.execute_command(command, description)

    def _prompt_for_params(self, param_names: List[str]) -> Dict[str, str]:
        """
        Prompt for parameters by name.

        Args:
            param_names: List of parameter names to prompt for

        Returns:
            Dictionary of parameter values
        """
        params = {}

        for name in param_names:
            value = questionary.text(f"Enter value for {name}:").ask()
            if value is None:
                value = ""
            params[name] = value

        return params

    def _build_command_with_params(self, base_command: str, params: Dict[str, str]) -> str:
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
            elif param == "qwen_submission_output_file":
                # Convert to the expected parameter name for the evaluation script
                command_parts.append(f"evaluate.custom_output_file={value}")
            elif param in ["input_file", "output_file", "eval_file", "submission_file"]:
                # These are positional arguments, not Hydra parameters
                continue
            else:
                command_parts.append(f"{param}={value}")

        # Handle positional arguments for specific commands
        if "trim_submission.py" in base_command:
            input_file = params.get("input_file", getattr(get_settings(), 'data', {}).get(
                'output_path', 'outputs/submission.jsonl')) or getattr(get_settings(), 'data', {}).get('output_path', 'outputs/submission.jsonl')
            output_file = params.get(
                "output_file", "outputs/submission_trimmed.jsonl") or "outputs/submission_trimmed.jsonl"
            max_length = params.get("max_length", "500") or "500"
            command_parts.extend([input_file, output_file, max_length])
        elif "transform_submission.py" in base_command:
            eval_file = params.get("eval_file", getattr(get_settings(), 'data', {}).get(
                'evaluation_path', 'data/eval.jsonl')) or getattr(get_settings(), 'data', {}).get('evaluation_path', 'data/eval.jsonl')
            submission_file = params.get("submission_file", getattr(get_settings(), 'data', {}).get(
                'output_path', 'outputs/submission.jsonl')) or getattr(get_settings(), 'data', {}).get('output_path', 'outputs/submission.jsonl')
            output_file = params.get(
                "output_file", "outputs/evaluation_logs.jsonl") or "outputs/evaluation_logs.jsonl"
            command_parts.extend([eval_file, submission_file, output_file])

        return " ".join(command_parts)

    def _is_potentially_dangerous_command(self, command: str) -> bool:
        """Check if a command might be dangerous to execute."""
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf /*",
            "dd if=",
            "mkfs",
            "fdisk",
            "format",
            "del /",
            "deltree",
            "shutdown",
            "reboot",
            "halt",
            "poweroff",
            "sudo rm",
            "sudo dd",
            "sudo mkfs",
            "> /dev/",
            "chmod 777",
        ]

        command_lower = command.lower()
        return any(pattern in command_lower for pattern in dangerous_patterns)

    def _confirm_dangerous_command(self, command: str, description: str) -> bool:
        """Ask user to confirm execution of a potentially dangerous command."""
        console.print(
            f"[bold red]⚠️  WARNING: This command appears potentially dangerous![/bold red]")
        console.print(f"[red]Command: {command}[/red]")
        console.print(f"[red]Description: {description}[/red]")
        console.print()

        confirmed = questionary.confirm(
            "Are you absolutely sure you want to execute this command?",
            default=False,
        ).ask()

        return confirmed if confirmed is not None else False
