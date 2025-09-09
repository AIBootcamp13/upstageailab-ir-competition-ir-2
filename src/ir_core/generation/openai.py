import os
from typing import List, Optional
import openai
import jinja2
from .base import BaseGenerator

class OpenAIGenerator(BaseGenerator):
    """
    A concrete implementation of the BaseGenerator for OpenAI models.

    This class interfaces with the OpenAI API to generate answers.
    It uses a Jinja2 template to construct the prompt.
    """
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        prompt_template_path: str = "prompts/scientific_qa_v1.jinja2",
        client: Optional[openai.OpenAI] = None,
    ):
        """
        Initializes the OpenAI generator.

        Args:
            model_name: The name of the OpenAI model to use.
            prompt_template_path: Path to the Jinja2 prompt template file.
            client: An optional pre-configured OpenAI client instance.
        """
        self.model_name = model_name
        self.client = client or openai.OpenAI()
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
            # Provide a fallback or a clearer error
            raise FileNotFoundError(
                f"Prompt template not found at '{self.prompt_template_path}'. "
                f"Ensure the path is correct relative to the project root."
            )

    def generate(self, query: str, context_docs: List[str]) -> str:
        """
        Generates an answer using the OpenAI Chat Completions API with a templated prompt.
        """
        # 1. Construct the prompt from the template
        try:
            full_prompt = self._render_prompt(query, context_docs)
        except FileNotFoundError as e:
            print(e)
            return "Error: Could not generate an answer due to a missing prompt template."

        # 2. Call the OpenAI API
        try:
            # We send the entire rendered prompt as a single "user" message
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.2,
            )
            # 3. Extract and return the answer
            if response.choices:
                return response.choices[0].message.content or "No answer was generated."
            else:
                return "The model did not return a valid response."

        except Exception as e:
            print(f"An error occurred while calling the OpenAI API: {e}")
            return "Error: Could not generate an answer from the model."

