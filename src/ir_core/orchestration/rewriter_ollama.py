# src/ir_core/orchestration/rewriter_ollama.py

import os
from typing import Optional
import requests
import jinja2
from .rewriter_openai import BaseQueryRewriter

class OllamaQueryRewriter(BaseQueryRewriter):
    """
    Query rewriter using Ollama models with custom prompt engineering.
    This is an alternative to OpenAI-based rewriting for full OSS stack.
    """

    def __init__(
        self,
        model_name: str = "qwen2:7b",
        prompt_template_path: str = "prompts/rewrite_query3.jinja2",
        ollama_base_url: str = "http://localhost:11434",
        max_tokens: int = 150,
        temperature: float = 0.1,
    ):
        """
        Initialize the Ollama query rewriter.

        Args:
            model_name: Ollama model name (e.g., 'qwen2:7b', 'llama3.1:8b')
            prompt_template_path: Path to the Jinja2 prompt template
            ollama_base_url: Ollama server URL
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
        """
        self.model_name = model_name
        self.base_url = ollama_base_url
        self.prompt_template_path = prompt_template_path
        self.max_tokens = max_tokens
        self.temperature = temperature

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
            conversation_history = [{"role": "user", "content": query}]
            full_prompt = template.render(conversation_history=conversation_history)

            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=30
            )
            response.raise_for_status()

            rewritten_query = response.json().get("response", "").strip()

            # Clean up the response (remove extra text, keep only the rewritten query)
            if rewritten_query:
                # Try to extract just the rewritten query
                lines = rewritten_query.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('Here', 'The', 'I', 'Query', '```')) and not line.endswith('```'):
                        rewritten_query = line
                        break
                else:
                    rewritten_query = query  # Fallback if no suitable line found

            # Remove extra quotes if present
            rewritten_query = rewritten_query.strip().strip('"')

            # 기본적인 검증: 재작성된 쿼리가 너무 짧거나 원본과 너무 다르면 원본 사용
            if len(rewritten_query) < 5 or self._is_query_too_different(query, rewritten_query):
                print(f"재작성된 쿼리가 부적절하여 원본 쿼리를 사용합니다: '{rewritten_query}'")
                return query

            return rewritten_query

        except Exception as e:
            print(f"Ollama rewriting failed: {e}")
            return query  # Fallback to original query
