#!/bin/bash
# Project-specific aliases for Information Retrieval RAG
# Source this file to get convenient shortcuts for frequently used commands
#
# Usage: source scripts/project-aliases.sh
# Or add to your ~/.bashrc: source /path/to/project/scripts/project-aliases.sh

# Check if uv is available before setting aliases
if command -v uv >/dev/null 2>&1; then
    # CLI menu - main project interface
    alias menu='uv run cli_menu.py'

    # CLI script - command-line interface
    alias cli='uv run scripts/cli.py'

    # Configuration switching utilities
    alias sc='uv run switch_config.py'
    alias sd='uv run switch_data_config.py'

    echo "✅ Project aliases loaded:"
    echo "   menu  → uv run cli_menu.py"
    echo "   cli   → uv run scripts/cli.py"
    echo "   sc    → uv run switch_config.py"
    echo "   sd    → uv run switch_data_config.py"
else
    echo "⚠️  uv not found - project aliases not loaded"
    echo "   Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi