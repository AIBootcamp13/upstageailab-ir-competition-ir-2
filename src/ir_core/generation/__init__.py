# src/ir_core/generation/__init__.py
from ..config import settings
from .base import BaseGenerator
from .openai import OpenAIGenerator
from .ollama import OllamaGenerator

def get_generator() -> BaseGenerator:
    """
    Factory function to get the appropriate generator based on settings.
    This function reads the `GENERATOR_TYPE` from the settings and returns
    an instantiated generator object.

    Returns:
        An instance of a class that inherits from BaseGenerator.

    Raises:
        ValueError: If the generator_type in settings is unknown.
    """
    generator_type = settings.GENERATOR_TYPE.lower()

    if generator_type == "openai":
        return OpenAIGenerator()
    elif generator_type == "ollama":
        return OllamaGenerator()
    else:
        raise ValueError(f"Unknown generator type: '{generator_type}'")

# By adding __all__, we explicitly define the public API of this package.
# This fixes the "unknown import symbol" error for linters and other tools.
__all__ = ["get_generator", "BaseGenerator"]