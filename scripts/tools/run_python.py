#!/usr/bin/env python3
"""
Poetry Python Runner

This script demonstrates the proper way to run Python commands in this Poetry-managed project.
It automatically uses the correct Poetry virtual environment.

Usage:
    python run_python.py <script_name> [args...]

Example:
    python run_python.py test_confidence_logging.py
    python run_python.py -c "import sys; print(sys.executable)"
"""

import subprocess
import sys
import os
from pathlib import Path


def run_poetry_python(args):
    """Run Python command using Poetry."""
    try:
        # Build the poetry run python command
        cmd = ['poetry', 'run', 'python'] + args

        # Run the command
        result = subprocess.run(cmd, cwd=os.getcwd())

        # Return the exit code
        return result.returncode

    except FileNotFoundError:
        print("❌ Error: Poetry is not installed or not in PATH")
        print("💡 Install Poetry: curl -sSL https://install.python-poetry.org | python3 -")
        return 1
    except Exception as e:
        print(f"❌ Error running Poetry command: {e}")
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("❌ Usage: python run_python.py <script_name> [args...]")
        print("💡 Example: python run_python.py test_confidence_logging.py")
        return 1

    # Get the script arguments (excluding this script name)
    script_args = sys.argv[1:]

    print("🚀 Running with Poetry Python...")
    print(f"📝 Command: poetry run python {' '.join(script_args)}")
    print("-" * 50)

    # Run the command
    exit_code = run_poetry_python(script_args)

    print("-" * 50)
    if exit_code == 0:
        print("✅ Command completed successfully")
    else:
        print(f"❌ Command failed with exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())