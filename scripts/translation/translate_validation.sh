#!/bin/bash
# Translate validation data to English
# This script uses the new integration system with caching

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_FILE="data/validation_balanced.jsonl"
OUTPUT_FILE="data/validation_balanced_en.jsonl"

echo "Translating validation data..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

cd "$REPO_DIR"

# Run the integration script with caching enabled
python scripts/translation/integrate_translation.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --cache

echo "Translation completed successfully!"
echo "Translated file: $OUTPUT_FILE"