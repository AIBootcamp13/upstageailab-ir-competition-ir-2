#!/usr/bin/env python3
"""
Path Utilities for RAG Project

This module provides centralized path resolution utilities for the RAG project.
It handles common path operations and ensures consistent path resolution across all scripts.
"""

import os
import sys
from pathlib import Path
from typing import Optional


class PathUtils:
    """Centralized path utilities for the RAG project."""

    # Environment variable for project root
    PROJECT_ROOT_ENV = "RAG_PROJECT_ROOT"

    @classmethod
    def get_project_root(cls) -> Path:
        """
        Get the project root directory.

        Priority order:
        1. Environment variable RAG_PROJECT_ROOT
        2. Auto-detect by searching for a marker file/folder (e.g., .git, pyproject.toml, src/)
        """
        # Check environment variable first
        env_root = os.getenv(cls.PROJECT_ROOT_ENV)
        if env_root:
            return Path(env_root).resolve()

        # Auto-detect by walking up the directory tree
        marker_files = [".git", "pyproject.toml", "src"]
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            for marker in marker_files:
                if (parent / marker).exists():
                    return parent
        # Fallback: previous brittle method (for legacy support)
        return current.parent.parent.parent.parent

    @classmethod
    def get_src_path(cls) -> Path:
        """Get the src directory path."""
        return cls.get_project_root() / "src"

    @classmethod
    def get_conf_path(cls) -> Path:
        """Get the conf directory path."""
        return cls.get_project_root() / "conf"

    @classmethod
    def get_scripts_path(cls) -> Path:
        """Get the scripts directory path."""
        return cls.get_project_root() / "scripts"

    @classmethod
    def get_data_path(cls) -> Path:
        """Get the data directory path."""
        return cls.get_project_root() / "data"

    @classmethod
    def get_outputs_path(cls) -> Path:
        """Get the outputs directory path."""
        return cls.get_project_root() / "outputs"

    @classmethod
    def get_settings_path(cls) -> Path:
        """Get the settings.yaml file path."""
        return cls.get_conf_path() / "settings.yaml"

    @classmethod
    def get_embedding_profiles_path(cls) -> Path:
        """Get the embedding_profiles.yaml file path."""
        return cls.get_conf_path() / "embedding_profiles.yaml"

    @classmethod
    def get_data_config_path(cls, config_name: str) -> Path:
        """Get a data config file path."""
        return cls.get_conf_path() / "data" / f"{config_name}.yaml"

    @classmethod
    def add_src_to_sys_path(cls) -> None:
        """Add the src directory to sys.path if not already present."""
        src_path = cls.get_src_path()
        src_str = str(src_path)

        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    @classmethod
    def ensure_project_root_env(cls) -> None:
        """Ensure the PROJECT_ROOT environment variable is set."""
        if not os.getenv(cls.PROJECT_ROOT_ENV):
            project_root = cls.get_project_root()
            os.environ[cls.PROJECT_ROOT_ENV] = str(project_root)

    @classmethod
    def validate_paths(cls) -> dict:
        """
        Validate that all expected paths exist.

        Returns:
            Dict with path validation results
        """
        paths_to_check = {
            "project_root": cls.get_project_root(),
            "src": cls.get_src_path(),
            "conf": cls.get_conf_path(),
            "scripts": cls.get_scripts_path(),
            "data": cls.get_data_path(),
            "outputs": cls.get_outputs_path(),
            "settings": cls.get_settings_path(),
            "embedding_profiles": cls.get_embedding_profiles_path(),
        }

        results = {}
        for name, path in paths_to_check.items():
            results[name] = {
                "path": path,
                "exists": path.exists(),
                "is_dir": path.is_dir() if path.exists() else False,
                "is_file": path.is_file() if path.exists() else False,
            }

        return results


# Convenience functions for backward compatibility and easy importing
def get_project_root() -> Path:
    """Get the project root directory."""
    return PathUtils.get_project_root()


def get_src_path() -> Path:
    """Get the src directory path."""
    return PathUtils.get_src_path()


def get_conf_path() -> Path:
    """Get the conf directory path."""
    return PathUtils.get_conf_path()


def get_scripts_path() -> Path:
    """Get the scripts directory path."""
    return PathUtils.get_scripts_path()


def get_data_path() -> Path:
    """Get the data directory path."""
    return PathUtils.get_data_path()


def get_outputs_path() -> Path:
    """Get the outputs directory path."""
    return PathUtils.get_outputs_path()


def get_settings_path() -> Path:
    """Get the settings.yaml file path."""
    return PathUtils.get_settings_path()


def get_embedding_profiles_path() -> Path:
    """Get the embedding_profiles.yaml file path."""
    return PathUtils.get_embedding_profiles_path()


def get_data_config_path(config_name: str) -> Path:
    """Get a data config file path."""
    return PathUtils.get_data_config_path(config_name)


def add_src_to_sys_path() -> None:
    """Add the src directory to sys.path if not already present."""
    PathUtils.add_src_to_sys_path()


def ensure_project_root_env() -> None:
    """Ensure the PROJECT_ROOT environment variable is set."""
    PathUtils.ensure_project_root_env()


def validate_paths() -> dict:
    """Validate that all expected paths exist."""
    return PathUtils.validate_paths()