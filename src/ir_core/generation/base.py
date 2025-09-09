from abc import ABC, abstractmethod
from typing import List

class BaseGenerator(ABC):
    """
    Abstract Base Class for a generation model.

    This class defines the standard interface for all generation models,
    ensuring they can be used interchangeably throughout the application.
    Any concrete generator (e.g., for OpenAI, Ollama) must implement
    the `generate` method.
    """

    @abstractmethod
    def generate(self, query: str, context_docs: List[str]) -> str:
        """
        Generates a final answer based on a user's query and a list of
        retrieved context documents.

        Args:
            query: The original user question.
            context_docs: A list of strings, where each string is the
                          content of a relevant document retrieved from
                          the search pipeline.

        Returns:
            A string containing the final, synthesized answer.
        """
        pass
