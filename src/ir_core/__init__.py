"""ir_core package — minimal public API surface.

This package intentionally keeps a small, side-effect-free import surface.
Importing ``ir_core`` will only expose the top-level ``api`` facade. Heavy
submodules should be imported explicitly (for example, ``from ir_core import
retrieval``) to avoid import-time side effects and fallback/compat logic.
"""

from . import api as api

__all__ = ["api"]
