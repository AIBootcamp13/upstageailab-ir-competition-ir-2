"""
Utility functions for scripts.
"""

import os
import sys
from pathlib import Path


def add_src_to_path() -> None:
    """
    Add the src directory to the Python path to allow for project imports.

    This is used in scripts to ensure they can import modules from src/.
    """
    scripts_dir = Path(__file__).parent.parent.parent.parent / "src"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
