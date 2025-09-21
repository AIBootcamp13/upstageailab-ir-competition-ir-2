# src/ir_core/query_enhancement/llm_client.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import openai
import requests
import json


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate a chat completion."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, client: Optional[openai.OpenAI] = None):
        self.client = client or openai.OpenAI()

    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate a chat completion using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                **kwargs
            )

            return {
                'success': True,
                'content': response.choices[0].message.content,
                'usage': getattr(response, 'usage', None)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'content': None
            }

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        try:
            # Try a simple ping
            self.client.models.list()
            return True
        except Exception:
            return False


class OllamaClient(LLMClient):
    """Ollama API client for local models like Qwen."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "qwen2:7b"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name

    def chat_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate a chat completion using Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }

            # Add any additional kwargs
            for key, value in kwargs.items():
                if key not in ['model', 'messages', 'stream']:
                    payload[key] = value

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60  # Longer timeout for local models
            )
            response.raise_for_status()

            result = response.json()
            message = result.get('message', {})

            return {
                'success': True,
                'content': message.get('content', ''),
                'usage': None  # Ollama doesn't provide usage stats in the same way
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'content': None
            }

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                return self.model_name in model_names
            return False
        except Exception:
            return False


def create_llm_client(client_type: str = "openai", **kwargs) -> LLMClient:
    """
    Factory function to create LLM clients.

    Args:
        client_type: Type of client ("openai" or "ollama")
        **kwargs: Additional arguments for client initialization

    Returns:
        LLMClient instance
    """
    if client_type.lower() == "openai":
        # OpenAIClient doesn't accept model_name in constructor
        kwargs.pop('model_name', None)
        return OpenAIClient(**kwargs)
    elif client_type.lower() == "ollama":
        return OllamaClient(**kwargs)
    else:
        raise ValueError(f"Unsupported client type: {client_type}")


def detect_client_type(model_name: str) -> str:
    """
    Detect the appropriate client type based on model name.

    Args:
        model_name: Name of the model

    Returns:
        Client type ("openai" or "ollama")
    """
    # If model name contains colon, it's likely an Ollama model (e.g., "qwen2:7b")
    if ":" in model_name:
        return "ollama"
    else:
        return "openai"