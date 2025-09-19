# src/ir_core/analysis/config/__init__.py

"""
Configuration module for the analysis package.

Provides centralized configuration management for all analysis components.
"""

from .config_loader import ConfigLoader, get_config_loader, get_config, get_typed_config

__all__ = [
    'ConfigLoader',
    'get_config_loader',
    'get_config',
    'get_typed_config'
]