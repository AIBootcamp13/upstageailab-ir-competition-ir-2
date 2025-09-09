import os
from typing import List
import requests
import jinja2
from .base import BaseGenerator

class OllamaGenerator(BaseGenerator):
    """
    A concrete implementation of the BaseGenerator for local Ollama models.

    This class interfaces with a running Ollama server to generate answers.
    It uses a Jinja2 template to construct the prompt.
    """
    def __init__(
        self,
        model_name: str = "llama3",
        prompt_template_path: str = "prompts/scientific_qa_v1.jinja2",
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initializes the Ollama generator.

        Args:
            model_name: The name of the Ollama model to use (e.g., 'llama3', 'mistral').
            prompt_template_path: Path to the Jinja2 prompt template file.
            ollama_base_url: The base URL of the running Ollama server.
        """
        self.model_name = model_name
        self.base_url = ollama_base_url
        self.prompt_template_path = prompt_template_path

        # Set up Jinja2 environment to load templates from the project root
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.getcwd()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _render_prompt(self, query: str, context_docs: List[str]) -> str:
        """Loads and renders the Jinja2 prompt template."""
        try:
            template = self.jinja_env.get_template(self.prompt_template_path)
            return template.render(query=query, context_docs=context_docs)
        except jinja2.TemplateNotFound:
            raise FileNotFoundError(
                f"Prompt template not found at '{self.prompt_template_path}'. "
                f"Ensure the path is correct relative to the project root."
            )

    def generate(self, query: str, context_docs: List[str]) -> str:
        """
        Generates an answer using the Ollama API with a templated prompt.
        """
        # 1. Construct the prompt from the template
        try:
            full_prompt = self._render_prompt(query, context_docs)
        except FileNotFoundError as e:
            print(e)
            return "Error: Could not generate an answer due to a missing prompt template."

        # 2. Call the Ollama API
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                },
                timeout=60 # Add a timeout for safety
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # 3. Extract and return the answer
            response_data = response.json()
            return response_data.get("response", "No answer was generated.").strip()

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while calling the Ollama API: {e}")
            return "Error: Could not connect to the Ollama server."
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "Error: An unexpected error occurred during generation."

