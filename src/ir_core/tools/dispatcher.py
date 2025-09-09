# src/ir_core/tools/dispatcher.py
from typing import Dict, Callable, Any
import json

# Import the specific tool functions that the dispatcher will manage
from .retrieval_tool import scientific_search

class ToolDispatcher:
    """
    Manages and executes available tools.

    This class holds a registry of all callable tools and provides a single
    interface to execute them by name. This decouples the LLM's decision
    (which tool to call) from the actual implementation of the tool,
    improving modularity and making the system easier to debug and extend.
    """
    def __init__(self):
        """Initializes the dispatcher and registers the available tools."""
        self._tool_registry: Dict[str, Callable[..., Any]] = {}
        self._register_all_tools()

    def _register_all_tools(self):
        """
        Registers all known tools into the dispatcher's registry.
        To make a new tool available to the RAG system, it must be
        imported and registered here.
        """
        self.register("scientific_search", scientific_search)

    def register(self, tool_name: str, function: Callable[..., Any]):
        """Adds a single tool to the registry."""
        self._tool_registry[tool_name] = function
        print(f"Registered tool: '{tool_name}'")

    def execute_tool(self, tool_name: str, tool_args_json: str) -> Any:
        """
        Executes a registered tool by its name with JSON string arguments.

        Args:
            tool_name: The name of the tool to execute (e.g., "scientific_search").
            tool_args_json: A JSON string of the arguments for the tool
                             (e.g., '{"query": "What is the biggest ocean?"}').

        Returns:
            The result of the tool's execution, or a formatted error message
            if the execution fails.
        """
        if tool_name not in self._tool_registry:
            return f"Error: Tool '{tool_name}' is not a valid or registered tool."

        try:
            tool_function = self._tool_registry[tool_name]
            # Deserialize the JSON string into a Python dictionary
            args = json.loads(tool_args_json)

            if not isinstance(args, dict):
                 raise TypeError("Tool arguments must be a valid JSON object, which becomes a Python dictionary.")

            print(f"Executing tool '{tool_name}' with args: {args}")
            # Use dictionary unpacking to pass arguments to the function
            return tool_function(**args)

        except json.JSONDecodeError:
            return f"Error: Invalid JSON provided for arguments of tool '{tool_name}'."
        except TypeError as e:
            # This catches errors if the arguments in the JSON don't match the
            # function's signature (e.g., wrong names, missing required args).
            return f"Error: Mismatched arguments for tool '{tool_name}'. Details: {e}"
        except Exception as e:
            # A general catch-all for any other errors during tool execution.
            return f"An unexpected error occurred while executing tool '{tool_name}': {e}"

# A default, pre-configured instance that can be imported and used elsewhere
# in the application, ensuring there is a single source of truth for tools.
default_dispatcher = ToolDispatcher()
