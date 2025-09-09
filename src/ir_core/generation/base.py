# src/ir_core/generation/base.py
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseGenerator(ABC):
    """
    Abstract Base Class for a generation model.
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        context_docs: List[str],
        prompt_template_path: Optional[str] = None
    ) -> str:
        """
        Generates a final answer based on a query, context, and an optional
        prompt template.

        Args:
            query: The original user question.
            context_docs: A list of strings of relevant document content.
            prompt_template_path: An optional path to a specific Jinja2 template
                                  to use for this generation call. If None, the
                                  generator's default template is used.

        Returns:
            A string containing the final, synthesized answer.
        """
        pass

