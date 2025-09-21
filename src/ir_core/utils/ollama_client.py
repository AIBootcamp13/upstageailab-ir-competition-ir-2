# src/ir_core/utils/ollama_client.py

"""
Ollama client utilities for local LLM integration.

This module provides utilities for interacting with local Ollama models,
enabling cost-effective and private LLM usage for the Scientific QA system.
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Response from Ollama API."""
    response: str
    model: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


class OllamaClient:
    """
    Client for interacting with local Ollama models.

    Provides methods for text generation, chat completion, and model management.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 120
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            model: Default model to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = model
        self.timeout = timeout
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        raw: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> OllamaResponse:
        """
        Generate text using Ollama model.

        Args:
            prompt: Input prompt
            model: Model to use (defaults to instance default)
            system: System message
            template: Template to use
            context: Context from previous conversation
            stream: Whether to stream response
            raw: Return raw response
            format: Response format
            options: Additional model options
            **kwargs: Additional parameters

        Returns:
            OllamaResponse object
        """
        model = model or self.default_model

        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        if system:
            data["system"] = system
        if template:
            data["template"] = template
        if context:
            data["context"] = context
        if format:
            data["format"] = format
        if options:
            data["options"] = options

        # Add any additional kwargs
        data.update(kwargs)

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            return OllamaResponse(
                response=result["response"],
                model=result["model"],
                total_duration=result.get("total_duration", 0),
                load_duration=result.get("load_duration", 0),
                prompt_eval_count=result.get("prompt_eval_count", 0),
                prompt_eval_duration=result.get("prompt_eval_duration", 0),
                eval_count=result.get("eval_count", 0),
                eval_duration=result.get("eval_duration", 0)
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> OllamaResponse:
        """
        Chat completion using Ollama model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use
            stream: Whether to stream response
            format: Response format
            options: Additional model options
            **kwargs: Additional parameters

        Returns:
            OllamaResponse object
        """
        model = model or self.default_model

        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        if format:
            data["format"] = format
        if options:
            data["options"] = options
        data.update(kwargs)

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            if stream:
                # For streaming, return the raw result
                return OllamaResponse(
                    response=result.get("message", {}).get("content", ""),
                    model=result.get("model", model),
                    total_duration=result.get("total_duration", 0),
                    load_duration=result.get("load_duration", 0),
                    prompt_eval_count=result.get("prompt_eval_count", 0),
                    prompt_eval_duration=result.get("prompt_eval_duration", 0),
                    eval_count=result.get("eval_count", 0),
                    eval_duration=result.get("eval_duration", 0)
                )

            return OllamaResponse(
                response=result["message"]["content"],
                model=result["model"],
                total_duration=result.get("total_duration", 0),
                load_duration=result.get("load_duration", 0),
                prompt_eval_count=result.get("prompt_eval_count", 0),
                prompt_eval_duration=result.get("prompt_eval_duration", 0),
                eval_count=result.get("eval_count", 0),
                eval_duration=result.get("eval_duration", 0)
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama chat API error: {e}")
            raise

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json()["models"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def pull_model(self, model: str) -> bool:
        """Pull a model from the registry."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False

    def check_health(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False


# Convenience functions for common use cases
def rewrite_query_ollama(
    original_query: str,
    client: Optional[OllamaClient] = None,
    model: str = "llama3.1:8b"
) -> str:
    """
    Rewrite a query using Ollama for better retrieval.

    Args:
        original_query: Original user query
        client: Ollama client instance
        model: Model to use

    Returns:
        Rewritten query string
    """
    if client is None:
        client = OllamaClient(model=model)

    system_prompt = """You are a scientific query rewriter. Your task is to rewrite user queries to be more effective for information retrieval in a scientific context.

Guidelines:
- Keep the core meaning but make it more precise
- Use scientific terminology appropriately
- Make it concise but comprehensive
- Focus on key concepts and entities
- If the query is already well-formed, return it unchanged
- Do not add extra information or assumptions

Return only the rewritten query, no explanations."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Original query: {original_query}\n\nRewritten query:"}
    ]

    try:
        response = client.chat(messages)
        rewritten = response.response.strip()

        # Clean up the response
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]

        return rewritten if rewritten else original_query

    except Exception as e:
        logger.warning(f"Ollama rewrite failed: {e}")
        return original_query


def generate_answer_ollama(
    query: str,
    context_docs: List[str],
    client: Optional[OllamaClient] = None,
    model: str = "llama3.1:8b"
) -> str:
    """
    Generate an answer using retrieved documents as context.

    Args:
        query: User query
        context_docs: List of retrieved document contents
        client: Ollama client instance
        model: Model to use

    Returns:
        Generated answer
    """
    if client is None:
        client = OllamaClient(model=model)

    # Combine context documents
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])

    system_prompt = """You are a scientific QA assistant. Answer questions based on the provided context documents.

Guidelines:
- Be accurate and precise
- Cite specific information from the documents when possible
- If information is not in the context, say so clearly
- Use scientific terminology appropriately
- Keep answers concise but comprehensive
- Structure answers clearly"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    ]

    try:
        response = client.chat(messages)
        return response.response.strip()
    except Exception as e:
        logger.error(f"Ollama answer generation failed: {e}")
        return "I apologize, but I encountered an error while generating the answer."


def generate_validation_queries_ollama(
    domain: str,
    description: str,
    num_queries: int = 5,
    client: Optional[OllamaClient] = None,
    model: str = "llama3.1:8b"
) -> List[str]:
    """
    Generate validation queries for a scientific domain using Ollama.

    Args:
        domain: Scientific domain name
        description: Domain description
        num_queries: Number of queries to generate
        client: Ollama client instance
        model: Model to use

    Returns:
        List of generated queries
    """
    if client is None:
        client = OllamaClient(model=model)

    prompt = f"""
Generate {num_queries} Korean questions about {description}.

Requirements:
1. Each question should be about core concepts in {domain}
2. Questions should be appropriate for a scientific QA system
3. Length: 10-30 characters per question
4. Include variety in difficulty and question types
5. Focus on factual, scientific content

Return only the questions, one per line, no numbering or extra text.
"""

    try:
        response = client.generate(prompt)
        queries = [q.strip() for q in response.response.split('\n') if q.strip()]
        return queries[:num_queries]
    except Exception as e:
        logger.error(f"Ollama validation query generation failed: {e}")
        return []


# Performance monitoring
def benchmark_ollama_model(
    client: OllamaClient,
    test_prompts: List[str],
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark Ollama model performance.

    Args:
        client: Ollama client instance
        test_prompts: List of test prompts
        num_runs: Number of runs per prompt

    Returns:
        Performance metrics
    """
    results = []

    for prompt in test_prompts:
        prompt_results = []

        for _ in range(num_runs):
            start_time = time.time()
            try:
                response = client.generate(prompt, stream=False)
                end_time = time.time()

                prompt_results.append({
                    "success": True,
                    "response_time": end_time - start_time,
                    "response_length": len(response.response),
                    "tokens_per_second": response.eval_count / (response.eval_duration / 1e9) if response.eval_duration > 0 else 0
                })
            except Exception as e:
                prompt_results.append({
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time
                })

        # Aggregate results for this prompt
        successful_runs = [r for r in prompt_results if r["success"]]
        if successful_runs:
            avg_time = sum(r["response_time"] for r in successful_runs) / len(successful_runs)
            avg_tokens_per_sec = sum(r["tokens_per_second"] for r in successful_runs) / len(successful_runs)
        else:
            avg_time = 0
            avg_tokens_per_sec = 0

        results.append({
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "success_rate": len(successful_runs) / num_runs,
            "avg_response_time": avg_time,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "total_runs": num_runs
        })

    return {
        "model": client.default_model,
        "benchmark_results": results,
        "overall_success_rate": sum(r["success_rate"] for r in results) / len(results) if results else 0
    }
