# src/ir_core/analysis/config/config_loader.py

"""
Configuration loader for the analysis module.

Provides a centralized way to load and access configuration values
from YAML files with validation and type safety.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import yaml
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf


@dataclass
class AnalysisConfig:
    """Typed configuration container for analysis module."""

    # Core settings
    version: str
    description: str

    # Analysis thresholds
    analysis_thresholds: Dict[str, float]

    # Metrics configuration
    default_k_values: List[int]
    metrics_config: Dict[str, Any]

    # Parallel processing
    parallel_config: Dict[str, Any]

    # Error analysis
    error_thresholds: Dict[str, float]
    error_categories: Dict[str, Dict[str, Any]]

    # Pattern detection
    pattern_detection_config: Dict[str, Any]

    # Query analysis
    query_analysis_config: Dict[str, Any]

    # Domain classification
    domain_keywords: Dict[str, List[str]]

    # Query type patterns
    query_type_patterns: Dict[str, List[str]]

    # Chunking configuration
    chunking_config: Dict[str, Dict[str, Any]]

    # Profiling configuration
    profiling_config: Dict[str, Any]

    # Reporting configuration
    reporting_config: Dict[str, Any]

    # Validation configuration
    validation_config: Dict[str, Any]

    # Logging configuration
    logging_config: Dict[str, Any]

    # Performance monitoring
    performance_config: Dict[str, Any]


class ConfigLoader:
    """
    Loads and validates configuration from YAML files.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default path relative to this file
            current_dir = Path(__file__).parent
            config_path = current_dir / "analysis_config.yaml"

        self.config_path = Path(config_path)
        self._config = None
        self._typed_config = None

    def load_config(self) -> DictConfig:
        """
        Load configuration from YAML file.

        Returns:
            DictConfig: Loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            # Convert to OmegaConf for better nested access
            self._config = OmegaConf.create(config_dict)
            return self._config  # type: ignore

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    def get_typed_config(self) -> AnalysisConfig:
        """
        Get typed configuration object.

        Returns:
            AnalysisConfig: Typed configuration container
        """
        if self._typed_config is None:
            config = self.load_config()
            self._typed_config = self._create_typed_config(config)

        return self._typed_config

    def _create_typed_config(self, config: DictConfig) -> AnalysisConfig:
        """Create typed configuration from OmegaConf object."""
        return AnalysisConfig(
            version=config.version,
            description=config.description,
            analysis_thresholds=dict(config.analysis.thresholds),
            default_k_values=list(config.metrics.default_k_values),
            metrics_config=dict(config.metrics),
            parallel_config=dict(config.parallel),
            error_thresholds=dict(config.error_analysis.thresholds),
            error_categories=dict(config.error_analysis.categories),
            pattern_detection_config=dict(config.pattern_detection),
            query_analysis_config=dict(config.query_analysis),
            domain_keywords=dict(config.domain_classification.keywords),
            query_type_patterns=dict(config.query_type_patterns),
            chunking_config=dict(config.chunking.recommendations),
            profiling_config=dict(config.profiling),
            reporting_config=dict(config.reporting),
            validation_config=dict(config.validation),
            logging_config=dict(config.logging),
            performance_config=dict(config.performance)
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Dot-separated key path (e.g., 'analysis.thresholds.map_score_low')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if self._config is None:
            self.load_config()

        try:
            if self._config is not None:
                return OmegaConf.select(self._config, key, default=default)
            return default
        except Exception:
            return default

    def get_threshold(self, name: str, default: float = 0.0) -> float:
        """
        Get analysis threshold value.

        Args:
            name: Threshold name
            default: Default value

        Returns:
            float: Threshold value
        """
        return float(self.get(f"analysis.thresholds.{name}", default))

    def get_parallel_setting(self, name: str, default: Any = None) -> Any:
        """
        Get parallel processing setting.

        Args:
            name: Setting name
            default: Default value

        Returns:
            Setting value
        """
        return self.get(f"parallel.{name}", default)

    def get_error_threshold(self, name: str, default: float = 0.0) -> float:
        """
        Get error analysis threshold.

        Args:
            name: Threshold name
            default: Default value

        Returns:
            float: Threshold value
        """
        return float(self.get(f"error_analysis.thresholds.{name}", default))

    def get_domain_keywords(self, domain: str) -> List[str]:
        """
        Get keywords for a specific domain.

        Args:
            domain: Domain name

        Returns:
            List[str]: Domain keywords
        """
        return list(self.get(f"domain_classification.keywords.{domain}", []))

    def get_query_pattern(self, pattern_type: str) -> List[str]:
        """
        Get query type patterns.

        Args:
            pattern_type: Pattern type (what, how, why, etc.)

        Returns:
            List[str]: Pattern strings
        """
        return list(self.get(f"query_type_patterns.{pattern_type}", []))

    def get_chunking_config(self, source: str) -> Dict[str, Any]:
        """
        Get chunking configuration for a source.

        Args:
            source: Source name

        Returns:
            Dict[str, Any]: Chunking configuration
        """
        config = dict(self.get(f"chunking.recommendations.{source}", {}))
        if not config:
            # Return default configuration
            config = dict(self.get("chunking.recommendations.default", {}))
        return config

    def validate_config(self) -> List[str]:
        """
        Validate configuration for required fields and consistency.

        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []

        # Check required sections
        required_sections = [
            'analysis.thresholds',
            'metrics',
            'parallel',
            'error_analysis.thresholds',
            'domain_classification.keywords'
        ]

        for section in required_sections:
            if self.get(section) is None:
                errors.append(f"Missing required section: {section}")

        # Validate threshold ranges
        thresholds = self.get('analysis.thresholds', {})
        for name, value in thresholds.items():
            if not isinstance(value, (int, float)):
                errors.append(f"Threshold {name} must be numeric, got {type(value)}")
            elif not 0 <= value <= 1:
                errors.append(f"Threshold {name} must be between 0 and 1, got {value}")

        # Validate parallel processing settings
        max_workers = self.get('parallel.max_workers', {})
        for component, workers in max_workers.items():
            if not isinstance(workers, int) or workers < 1:
                errors.append(f"Max workers for {component} must be positive integer, got {workers}")

        return errors

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._config = None
        self._typed_config = None
        self.load_config()


# Global configuration instance
_config_loader = None

def get_config_loader() -> ConfigLoader:
    """Get global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def get_config() -> DictConfig:
    """Get global configuration."""
    return get_config_loader().load_config()

def get_typed_config() -> AnalysisConfig:
    """Get typed configuration object."""
    return get_config_loader().get_typed_config()