#!/usr/bin/env python3
"""
Enhanced Validation Script with Translation Support

This script demonstrates how to integrate translation into the validation pipeline.
It can automatically translate Korean queries to English before running validation.

Usage:
    python scripts/translation/validate_with_translation.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add src to path
scripts_dir = Path(__file__).parent
repo_dir = scripts_dir.parent.parent
src_dir = repo_dir / "src"
sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def translate_validation_data(input_file: str, output_file: str, use_cache: bool = True) -> bool:
    """Translate validation data using the integration script."""
    try:
        integration_script = repo_dir / "scripts" / "translation" / "integrate_translation.py"

        if not integration_script.exists():
            logger.error(f"Integration script not found: {integration_script}")
            return False

        cmd = [
            sys.executable,
            str(integration_script),
            "--input", input_file,
            "--output", output_file
        ]

        if use_cache:
            cmd.append("--cache")

        logger.info(f"Running translation: {' '.join(cmd)}")

        import subprocess
        # Set up environment with PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src_dir)
        
        result = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True, env=env)

        if result.returncode == 0:
            logger.info("Translation completed successfully")
            return True
        else:
            logger.error(f"Translation failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error during translation: {e}")
        return False

def run_validation_with_translation(use_translation: bool = True):
    """Run validation pipeline with optional translation."""

    # Configuration
    original_validation = "data/validation_balanced.jsonl"
    translated_validation = "data/validation_balanced_en.jsonl"

    # Check if translation is enabled
    if use_translation:
        logger.info("Translation enabled - checking for translated validation data...")

        # Check if translated file exists
        if not Path(translated_validation).exists():
            logger.info("Translated validation file not found - creating it...")
            success = translate_validation_data(original_validation, translated_validation)

            if not success:
                logger.warning("Translation failed - falling back to original data")
                validation_file = original_validation
            else:
                validation_file = translated_validation
        else:
            logger.info("Using existing translated validation file")
            validation_file = translated_validation
    else:
        logger.info("Translation disabled - using original validation data")
        validation_file = original_validation

    # Now run the standard validation pipeline
    logger.info(f"Running validation with file: {validation_file}")

    # Import and run the validation script
    try:
        # Modify sys.argv to pass the translated file to the validation script
        original_argv = sys.argv[:]
        sys.argv = [
            'validate_retrieval.py',
            f'data.validation_path={validation_file}',
            'limit=10'  # Small limit for testing
        ]

        # Import the validation module
        scripts_dir = repo_dir / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        
        from evaluation.validate_retrieval import run

        # Run validation (this would normally be called via Hydra)
        logger.info("Starting validation pipeline...")

        # Actually run the validation command
        import subprocess
        cmd = [
            sys.executable, "scripts/evaluation/validate_retrieval.py",
            f"data.validation_path={validation_file}",
            "limit=10"  # Small limit for testing
        ]
        
        logger.info(f"Running validation: {' '.join(cmd)}")
        
        # Set up environment with PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(src_dir)
        
        result = subprocess.run(cmd, cwd=str(repo_dir), env=env)
        
        if result.returncode == 0:
            logger.info("Validation completed successfully")
        else:
            logger.error("Validation failed")

        # Reset sys.argv
        sys.argv = original_argv

    except Exception as e:
        logger.error(f"Error running validation: {e}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run validation with optional translation")
    parser.add_argument("--translate", action="store_true", default=True,
                       help="Enable translation (default: True)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable translation caching")

    args = parser.parse_args()

    use_cache = not args.no_cache
    run_validation_with_translation(use_translation=args.translate)

if __name__ == "__main__":
    main()