# src/ir_core/orchestration/pipeline.py

from typing import List, Dict, Any, cast, Optional
import os
import json
import openai
import requests
from openai.types.chat import ChatCompletionMessageToolCall
from omegaconf import DictConfig
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text

# Simple toggle for rich logging - set RAG_SIMPLE_LOGGING=1 to disable rich output
USE_RICH_LOGGING = os.getenv('RAG_SIMPLE_LOGGING', '0') != '1'

from ..generation.base import BaseGenerator
from ..generation import get_generator
from ..tools.dispatcher import default_dispatcher, ToolDispatcher
from ..tools.retrieval_tool import get_tool_definition
from ..query_enhancement.manager import QueryEnhancementManager
from .rewriter_openai import BaseQueryRewriter # Keep for backward compatibility

class RAGPipeline:
    """
    Enhanced RAG pipeline with query enhancement capabilities.
    """
    def __init__(
        self,
        generator: Optional[BaseGenerator] = None,
        model_name: Optional[str] = None,
        tool_prompt_description: str = "",
        tool_calling_model: str = "gpt-3.5-turbo",
        query_rewriter: Optional[BaseQueryRewriter] = None, # Made optional for backward compatibility
        dispatcher: ToolDispatcher = default_dispatcher,
        use_query_enhancement: bool = True
    ):
        """
        Initialize the RAG pipeline.

        Args:
            generator: Pre-configured generator instance (optional)
            model_name: Model name to create generator from (optional)
            tool_prompt_description: Description for tool calling
            tool_calling_model: Model to use for tool calling
            query_rewriter: Legacy query rewriter (optional)
            dispatcher: Tool dispatcher instance
            use_query_enhancement: Whether to use query enhancement
        """
        # Create generator if model_name provided
        if generator is None and model_name is not None:
            # Load configuration from consolidated settings.yaml
            import os
            from pathlib import Path
            from omegaconf import OmegaConf

            # Register environment resolver (only if not already registered)
            try:
                OmegaConf.register_new_resolver("env", os.getenv)
            except ValueError:
                # Resolver already registered, continue
                pass

            # Load consolidated configuration
            config_path = Path(__file__).parent.parent.parent.parent / "conf" / "settings.yaml"
            cfg = cast(DictConfig, OmegaConf.load(config_path))

            # Override generator settings based on model_name
            cfg.GENERATOR_TYPE = "ollama" if ":" in model_name else "openai"
            cfg.GENERATOR_MODEL_NAME = model_name

            self.generator = get_generator(cfg)
        elif generator is not None:
            self.generator = generator
        else:
            raise ValueError("Either 'generator' or 'model_name' must be provided")

        self.query_rewriter = query_rewriter # Keep for backward compatibility
        self.dispatcher = dispatcher
        self.tool_prompt_description = tool_prompt_description
        self.tool_calling_model = tool_calling_model
        self.use_query_enhancement = use_query_enhancement

        # Initialize query enhancement manager if enabled
        if use_query_enhancement:
            self.enhancement_manager = QueryEnhancementManager(model_name=model_name)
        else:
            self.enhancement_manager = None

        # Determine if using OpenAI or Ollama for tool calling
        if ":" in tool_calling_model:  # Assume Ollama model (e.g., qwen2:7b)
            self.use_ollama = True
            self.ollama_url = "http://localhost:11434/api/chat"
        else:  # Assume OpenAI
            self.use_ollama = False
            self.client = openai.OpenAI()

    def run_retrieval_only(self, query: str) -> List[Dict[str, Any]]:
        """
        평가 목적으로 파이프라인의 검색 부분만 실행합니다.
        """
        technique_used: str = 'none'
        # --- 1단계: 쿼리 개선 ---
        if self.enhancement_manager:
            # Use new query enhancement system
            enhancement_result = self.enhancement_manager.enhance_query(query)

            # Check if enhancement was bypassed (conversational queries)
            if enhancement_result.get('technique_used') == 'CONVERSATIONAL_SKIP':
                reason = enhancement_result.get('reason', 'unknown')
                if USE_RICH_LOGGING:
                    rprint(f"[dim cyan]쿼리 개선 우회:[/dim cyan] [yellow]'{query}'[/yellow] [dim](유형: {reason})[/dim]")
                else:
                    print(f"Query enhancement bypassed: {query} (type: {reason})")
                technique_used = 'CONVERSATIONAL_SKIP'
                return [{"standalone_query": query, "docs": [], "technique_used": technique_used}]

            enhanced_query = enhancement_result.get('enhanced_query', query)
            technique_used = enhancement_result.get('technique_used', 'none')

            # Check if HyDE already provided retrieval results
            if technique_used == 'hyde' and 'retrieval_results' in enhancement_result:
                hyde_results = enhancement_result['retrieval_results']
                if hyde_results:
                    if USE_RICH_LOGGING:
                        rprint(f"[bold green]HyDE 검색 결과 사용:[/bold green] [yellow]'{query}'[/yellow] -> [green]{len(hyde_results)}개 문서 발견[/green]")
                    else:
                        print(f"HyDE retrieval results used: {query} -> {len(hyde_results)} documents found")
                    return [{"standalone_query": enhanced_query, "docs": hyde_results, "technique_used": technique_used}]

            # Create structured output for query enhancement
            # original_text = Text(f"원본 쿼리: {query}", style="bold red")
            # enhanced_text = Text(f"개선된 쿼리: {enhanced_query}", style="bold green")
            # technique_text = Text(f"기법: {technique_used}", style="bold blue")

            # panel = Panel.fit(
            #     f"{original_text}\n{enhanced_text}\n{technique_text}",
            #     title="[bold magenta]쿼리 개선 결과[/bold magenta]",
            #     border_style="magenta"
            # )
            # rprint(panel)
            # --- NEW: Create separate, color-coded panels for better readability ---
            # Panel for the Original Query

            original_panel = Panel.fit(
                Text(query, style="white"),
                title="[bold red]Original Query[/bold red]",
                border_style="red"
            )

            # Panel for the Enhanced Query and Technique Used
            enhancement_content = Text()
            enhancement_content.append("Enhanced Query: ", style="bold green")
            enhancement_content.append(enhanced_query, style="white")
            enhancement_content.append("\nTechnique Used: ", style="bold blue")
            enhancement_content.append(technique_used, style="white")

            # Add confidence score if available
            confidence = enhancement_result.get('confidence')
            if confidence is not None:
                enhancement_content.append("\nConfidence Score: ", style="bold yellow")
                # Format confidence with color coding
                if confidence >= 0.8:
                    confidence_style = "green"
                elif confidence >= 0.5:
                    confidence_style = "yellow"
                else:
                    confidence_style = "red"
                enhancement_content.append(f"{confidence:.2f}", style=confidence_style)

            enhanced_panel = Panel.fit(
                enhancement_content,
                title="[bold green]Enhancement Result[/bold green]",
                border_style="green"
            )

            # Display enhancement results
            if USE_RICH_LOGGING:
                rprint(original_panel)
                rprint(enhanced_panel)
            else:
                # Simple LM-friendly logging
                print(f"Original Query: {query}")
                print(f"Enhanced Query: {enhanced_query}")
                print(f"Technique Used: {technique_used}")
                if confidence is not None:
                    print(f"Confidence Score: {confidence:.2f}")


        elif self.query_rewriter:
            # Fallback to old rewriter for backward compatibility
            enhanced_query = self.query_rewriter.rewrite_query(query)
            technique_used = 'rewriting'
            if USE_RICH_LOGGING:
                original_text = Text(f"원본 쿼리: {query}", style="bold red")
                enhanced_text = Text(f"재작성된 쿼리: {enhanced_query}", style="bold green")

                panel = Panel.fit(
                    f"{original_text}\n{enhanced_text}",
                    title="[bold cyan]쿼리 재작성 결과[/bold cyan]",
                    border_style="cyan"
                )
                rprint(panel)
            else:
                print(f"Original Query: {query}")
                print(f"Rewritten Query: {enhanced_query}")
                print(f"Technique Used: rewriting")
        else:
            # No enhancement
            enhanced_query = query
            technique_used = 'none'
            if USE_RICH_LOGGING:
                rprint(f"[dim yellow]쿼리 개선 없이 사용:[/dim yellow] [yellow]'{query}'[/yellow]")
            else:
                print(f"No query enhancement used: {query}")

        tools = [cast(Any, get_tool_definition(self.tool_prompt_description))]

        try:
            if self.use_ollama:
                # Use Ollama for tool calling
                tools = [get_tool_definition(self.tool_prompt_description)]
                payload = {
                    "model": self.tool_calling_model,
                    "messages": [{"role": "user", "content": enhanced_query}],
                    "tools": tools,
                    "stream": False
                }
                response = requests.post(self.ollama_url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                response_message = result.get("message", {})
                tool_calls = response_message.get("tool_calls", [])
                if tool_calls:
                    tool_call = tool_calls[0]
                    tool_args_json = json.dumps(tool_call["function"]["arguments"])
                    tool_name = tool_call["function"]["name"]
                else:
                    tool_name = None
                    tool_args_json = "{}"
                    tool_args_json = "{}"
            else:
                # Use OpenAI for tool calling
                tools = [cast(Any, get_tool_definition(self.tool_prompt_description))]
                response = self.client.chat.completions.create(
                    model=self.tool_calling_model,
                    messages=[{"role": "user", "content": enhanced_query}],
                    tools=tools,
                    tool_choice="auto"
                )
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                if tool_calls:
                    tool_call = cast(ChatCompletionMessageToolCall, tool_calls[0])
                    tool_args_json = tool_call.function.arguments or "{}"
                    tool_name = tool_call.function.name
                else:
                    tool_name = None
                    tool_args_json = "{}"

            if tool_name:
                tool_result = self.dispatcher.execute_tool(
                    tool_name=tool_name,
                    tool_args_json=tool_args_json
                )
                return [{"standalone_query": enhanced_query, "docs": tool_result, "technique_used": technique_used}]
            else:
                return [{"standalone_query": enhanced_query, "docs": [], "technique_used": technique_used}]
        except Exception as e:
            print(f"'{query}' 쿼리에 대한 검색 중 오류 발생: {e}")
            return [{"standalone_query": enhanced_query, "docs": [], "technique_used": technique_used}]

    def run(self, query: str) -> str:
        """
        주어진 사용자 쿼리에 대해 전체 RAG 파이프라인을 실행합니다.
        """
        retrieved_output = self.run_retrieval_only(query)

        docs = retrieved_output[0].get("docs", [])
        standalone_query = retrieved_output[0].get("standalone_query", query)

# src/ir_core/orchestration/pipeline.py

        # IMPROVED: Handle case where docs might be a string (error message) or None
        if not isinstance(docs, list):
            error_message = docs if isinstance(docs, str) else "an unknown error."
            print(f"🐛 DEBUG Retrieval failed. Error: {error_message}")
            # Generate a response indicating failure
            final_answer = self.generator.generate(
                query=standalone_query,
                context_docs=[],
                prompt_template_path="prompts/error_fallback/error_fallback_v1.jinja2" # Use a conversational template for error messages
            )
            return final_answer

        # Check if this is a conversational query that should bypass retrieval
        if self.enhancement_manager and not docs:
            # Check if the enhancement result indicates bypass
            enhancement_result = self.enhancement_manager.enhance_query(query)
            if enhancement_result.get('technique_used') == 'CONVERSATIONAL_SKIP':
                # Generate a direct conversational response
                conversational_prompt = f"""
                사용자가 다음과 같은 질문을 했습니다: "{query}"

                이는 과학 검색이 필요하지 않은 대화형 질문입니다.
                친절하고 도움이 되는 방식으로 답변해주세요.
                """
                return self.generator.generate(query=conversational_prompt, context_docs=[])

        # Always run generation; if no docs, generate with empty context
        context_docs_content = [item.get('content', '') for item in docs if isinstance(item, dict)] if isinstance(docs, list) else []
        final_answer = self.generator.generate(query=standalone_query, context_docs=context_docs_content)

        return final_answer