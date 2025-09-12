# src/ir_core/orchestration/rewriter_ollama.py

import os
from typing import Optional
import requests
import jinja2

class OllamaQueryRewriter:
    """
    Query rewriter using Ollama models with custom prompt engineering.
    This is an alternative to OpenAI-based rewriting for full OSS stack.
    """

    def __init__(
        self,
        model_name: str = "qwen2:7b",
        prompt_template_path: str = "prompts/rewrite_query2.jinja2",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the Ollama query rewriter.

        Args:
            model_name: Ollama model name (e.g., 'qwen2:7b', 'llama3.1:8b')
            prompt_template_path: Path to the Jinja2 prompt template
            ollama_base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = ollama_base_url
        self.prompt_template_path = prompt_template_path

        # Set up Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.getcwd()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a query using Ollama model.

        Args:
            query: Original user query

        Returns:
            Rewritten query optimized for search
        """
        try:
            # Load and render the prompt template
            template = self.jinja_env.get_template(self.prompt_template_path)
            full_prompt = template.render(query=query)

            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent rewriting
                        "top_p": 0.9,
                        "num_predict": 200   # Limit response length
                    }
                },
                timeout=30
            )
            response.raise_for_status()

            rewritten_query = response.json().get("response", "").strip()

            # Clean up the response (remove extra text, keep only the rewritten query)
            # This is a simple approach - could be enhanced with better parsing
            if rewritten_query:
                # Try to extract just the rewritten query
                lines = rewritten_query.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('Here', 'The', 'I', 'Query')):
                        return line

            return query  # Fallback to original query

        except Exception as e:
            print(f"Ollama rewriting failed: {e}")
            return query  # Fallback to original query
