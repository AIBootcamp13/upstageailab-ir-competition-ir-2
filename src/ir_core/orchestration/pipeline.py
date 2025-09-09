# src/ir_core/orchestration/pipeline.py
from typing import List, Dict, Any, cast
import os
import openai
from openai.types.chat import ChatCompletionMessageToolCall

# Import the core components we've built
from ..generation.base import BaseGenerator
from ..tools.dispatcher import default_dispatcher, ToolDispatcher
from ..tools.retrieval_tool import get_tool_definition

class RAGPipeline:
    """
    Orchestrates the entire RAG process, from receiving a query to generating a final answer.

    This pipeline uses a generator model (like OpenAI's GPT series) to decide
    whether to answer directly or to first use a tool (like scientific_search).
    If a tool is required, it uses the ToolDispatcher to execute it and then
    feeds the result back to the generator to produce a final, context-aware answer.
    """
    def __init__(self, generator: BaseGenerator, dispatcher: ToolDispatcher = default_dispatcher):
        """
        Initializes the RAG Pipeline.

        Args:
            generator: An instance of a generator class (e.g., OpenAIGenerator).
                       This is the primary LLM used for decision-making and generation.
            dispatcher: An instance of the ToolDispatcher to execute tools.
        """
        self.generator = generator
        self.dispatcher = dispatcher

        # This implementation is tightly coupled with the OpenAI client for tool calling.
        # Ensure the API key is available in the environment.
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY environment variable not set. Tool calling will fail.")
        self.client = openai.OpenAI()

    def run(self, query: str) -> str:
        """
        Executes the full RAG pipeline for a given user query.

        The process is as follows:
        1.  Make an initial call to the LLM with the query and available tools.
        2.  If the LLM decides to use a tool, parse its response.
        3.  Use the ToolDispatcher to execute the requested tool.
        4.  Take the tool's output (e.g., retrieved documents) as context.
        5.  Call the generator a second time with the original query and the new context
            to produce the final answer.
        6.  If the LLM decides not to use a tool, generate a direct response.

        Returns:
            The final generated answer as a string.
        """
        print(f"\n--- Running RAG Pipeline for query: '{query}' ---")

        # For now, we only have one tool. In the future, this could be a list of tools.
        # get_tool_definition returns a plain dict describing the tool; the OpenAI SDK
        # typings are strict so cast to Any to avoid type errors at runtime.
        tools = [cast(Any, get_tool_definition())]

        try:
            # First call: The LLM decides whether to use a tool or answer directly.
            print("Step 1: Making decision call to LLM...")
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # A cost-effective model for tool-use decisions
                messages=[{"role": "user", "content": query}],
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Step 2: Check if the LLM requested a tool.
            if tool_calls:
                # For this implementation, we'll only handle the first tool call.
                tool_call = cast(ChatCompletionMessageToolCall, tool_calls[0])
                # Directly access the function attribute assuming SDK object structure
                tool_name = tool_call.function.name
                tool_args_json = tool_call.function.arguments or "{}"
                tool_args_json = tool_call.function.arguments or "{}"

                # Step 3: Execute the tool using the dispatcher.
                print(f"Step 3: Executing tool '{tool_name}' via dispatcher...")
                tool_result = self.dispatcher.execute_tool(
                    tool_name=tool_name,
                    tool_args_json=tool_args_json
                )

                # The tool result (retrieved docs) needs to be a list of strings for the prompt.
                if isinstance(tool_result, list) and all(isinstance(item, dict) and 'content' in item for item in tool_result):
                        context_docs = [item['content'] for item in tool_result]
                else:
                    # Handle cases where the tool returns an error string or unexpected format.
                    context_docs = [str(tool_result)]

                print(f"Step 4: Received context from tool. Number of documents: {len(context_docs)}")

                # Step 5: Call the generator with the context to get the final answer.
                print("Step 5: Generating final answer with context...")
                final_answer = self.generator.generate(query=query, context_docs=context_docs)

            else:
                # Step 6: No tool was called. Generate a direct response.
                print("Step 2: LLM decided to answer directly.")
                final_answer = self.generator.generate(query=query, context_docs=[])

            print("--- Pipeline finished ---")
            return final_answer

        except Exception as e:
            print(f"An error occurred during the RAG pipeline execution: {e}")
            return "Error: Could not complete the request due to an internal error."
