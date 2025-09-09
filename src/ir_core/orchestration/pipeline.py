# src/ir_core/orchestration/pipeline.py
from typing import List, Dict, Any, Optional, cast
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

    def run_retrieval_only(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes only the retrieval part of the pipeline for evaluation purposes.

        This method asks the LLM to decide if a tool is needed. If so, it
        executes the tool and returns the results. If not, it returns an
        empty list. This aligns with the competition's evaluation logic.

        Args:
            query: The user query.

        Returns:
            A list of retrieved document dictionaries, or an empty list.
        """
        tools = [cast(Any, get_tool_definition())]

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}],
                tools=tools,
                tool_choice="auto"
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                tool_call = cast(ChatCompletionMessageToolCall, tool_calls[0])
                tool_name = tool_call.function.name
                tool_args_json = tool_call.function.arguments or "{}"

                tool_result = self.dispatcher.execute_tool(
                    tool_name=tool_name,
                    tool_args_json=tool_args_json
                )
                # Ensure the result is a list for consistent return type
                return tool_result if isinstance(tool_result, list) else []
            else:
                # No tool call was made, so return no documents.
                return []
        except Exception as e:
            print(f"An error occurred during retrieval for query '{query}': {e}")
            return []

    def run(self, query: str) -> str:
        """
        Executes the full RAG pipeline for a given user query.
        """
        print(f"\n--- Running RAG Pipeline for query: '{query}' ---")

        try:
            # Step 1 & 2: Decide if a tool is needed and get the results.
            # We now use our dedicated retrieval method for this.
            retrieved_docs_list = self.run_retrieval_only(query)

            # Step 3: Generate the final answer based on the retrieval result.
            if retrieved_docs_list:
                print(f"Step 4: Received context from tool. Number of documents: {len(retrieved_docs_list)}")
                context_docs = [item.get('content', '') for item in retrieved_docs_list]

                print("Step 5: Generating final answer with context...")
                final_answer = self.generator.generate(query=query, context_docs=context_docs)
            else:
                print("Step 2: LLM decided to answer directly or retrieval failed.")
                final_answer = self.generator.generate(query=query, context_docs=[])

            print("--- Pipeline finished ---")
            return final_answer

        except Exception as e:
            print(f"An error occurred during the RAG pipeline execution: {e}")
            return "Error: Could not complete the request due to an internal error."

