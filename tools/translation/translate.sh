#!/bin/bash
# Translation wrapper script using uv
# Usage: ./translate.sh <input_file> <output_file> [text_field]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Navigate to the uv project directory
cd "$SCRIPT_DIR/translation-tools"

# Run the translation script with uv
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_file> <output_file> [text_field]"
    echo "Example: $0 ../../../data/validation_balanced.jsonl ../../../outputs/validation_en.jsonl content"
    exit 1
fi

echo "Running translation with uv..."
# Convert relative paths to absolute paths
INPUT_FILE="$(cd "$PROJECT_ROOT" && realpath "$1")"
OUTPUT_FILE="$(cd "$PROJECT_ROOT" && realpath "$2")"
shift 2

uv run python translate.py "$INPUT_FILE" "$OUTPUT_FILE" "$@"

echo "Translation completed successfully!"