
-----

## ✅ Phase 2: Testing, Validation, and Maintainability

This phase focuses on making the existing retrieval codebase robust, reliable, and easy for new contributors to understand and work with.

### **Step 1: Improve Developer Experience (DX) & Onboarding**

The goal here is to make it incredibly fast for anyone to get the project running and understand its structure.

  - `[x]` **Create a "Quickstart" Guide:** Add a new section to `README.md` right after the table of contents. It should contain the three most essential commands to see the system work:

    1.  Start the necessary infrastructure: `poetry run ./scripts/run-local.sh start`
    2.  Index the sample data: `PYTHONPATH=src poetry run python scripts/reindex.py data/documents.jsonl --index test`
    3.  Run a sample query script (you'll need to create this, e.g., `scripts/run_query.py`) that takes a question and prints the retrieved documents.

  - `[x]` **Add Core Documentation:** Ensure the `docs/` folder contains key information. The project already has a good foundation.

      - `[x]` **Verify Architecture Diagram:** Review `docs/assets/diagrams/rag-pipeline.md` and ensure it accurately reflects the current codebase.
      - `[x]` **Verify Sequence Diagram:** Check `docs/assets/diagrams/system-flow-detailed.md` to confirm the query flow is correct.
      - `[x]` **Add Troubleshooting Section:** Create a small section in the `README.md` for common errors like `ConnectionRefusedError` or `index_not_found_exception` and their solutions.

### **Step 2: Solidify the Code Structure**

This step focuses on internal code quality, making the logic cleaner and preventing subtle bugs.

  - `[x]` **Decouple Runtime Initializers:** Move any code in `__init__.py` files that loads models or connects to databases into explicit functions.

      - **Reasoning:** Importing a module should never have side effects like loading a large model into memory or establishing a network connection. This makes the code predictable and easier to test.
      - **Action:** In `src/ir_core/embeddings/core.py`, ensure `load_model()` is called by the `encode_` functions, not automatically when the module is imported. The current structure already does this well, so this is more of a check to enforce the pattern.

  - `[x]` **Define Module Contracts:** For each core module, create or update a `README.md` within its directory (e.g., `src/ir_core/retrieval/README.md`). This document should clearly state:

      - **Purpose:** What is this module's responsibility?
      - **Key Functions:** List the main public functions (e.g., `hybrid_retrieve`).
      - **Inputs & Outputs:** What are the data types for arguments and return values? (e.g., Input: `query: str`, Output: `list[dict]`).
      - **Error Handling:** How does it behave on failure? (e.g., "Returns an empty list if the index does not exist.").

### **Step 3: Implement Robust Testing**

This is the most critical step for long-term maintainability.

  - `[x]` **Write Unit Tests:** Use `pytest` to test pure logic that doesn't require external services.

      - `[x]` Test `evaluation` functions (`precision_at_k`, `mrr`) with known inputs and expected outputs.
      - `[x]` Test `utils` functions to ensure file reading/writing works as expected.
      - `[x]` Test the scoring logic within `hybrid_retrieve` by providing it with sample BM25 scores and cosine similarities to check the final blended score calculation.

  - `[x]` **Write an Integration Test:** Create a single, powerful test that validates the entire retrieval pipeline.

      - **Action:** Create `tests/test_integration_retrieval.py`.
      - This test should use your `scripts/run-local.sh` to automatically start fresh instances of Elasticsearch and Redis.
      - It will then programmatically index a small, controlled set of documents (e.g., 5-10 sample texts).
      - Finally, it will run a query that is known to match one of the documents and assert that this document is returned in the top results.
      - The test should clean up after itself by stopping the services.

-----

## 🧠 Phase 3: Building a Flexible and Experimental Generation Engine

This phase focuses on implementing the "G" in RAG. The architecture is designed for extensive experimentation and to avoid the tool-calling issues you've faced before.

### **Step 1: Design an Abstracted Generation Layer**

To support different models (OpenAI, Ollama, HuggingFace), we need a common interface.

  - `[x]` **Create the Generator Interface:**

      - Define an abstract base class in a new file `src/ir_core/generation/base.py`.

    <!-- end list -->

    ```python
    from abc import ABC, abstractmethod

    class BaseGenerator(ABC):
        @abstractmethod
        def generate(self, prompt: str, context_docs: list[str]) -> str:
            """Generates an answer based on the prompt and context."""
            pass
    ```

  - `[x]` **Create Concrete Implementations:**

      - Create `src/ir_core/generation/openai.py` with a class `OpenAIGenerator(BaseGenerator)`.
      - Create `src/ir_core/generation/ollama.py` with a class `OllamaGenerator(BaseGenerator)`.
      - Create `src/ir_core/generation/huggingface.py` for a local transformer model.

  - `[x]` **Add a Factory Function:** Create a function in `src/ir_core/generation/__init__.py` that reads from your project's configuration and returns the correct generator instance. This allows you to switch models just by changing a config file.

### **Step 2: Implement a Scalable Prompt Management System**

To facilitate experimentation, prompts should be treated as templates, not hardcoded strings.

  - `[x]` **Create a `prompts` Directory:** Create a new top-level directory named `prompts`.
  - `[x]` **Use a Templating Engine:** Integrate a lightweight templating engine like **Jinja2**.
  - `[x]` **Structure Your Prompts:** Store prompts in files within the `prompts` directory. For example, `prompts/scientific_qa/v1.jinja`.
    ```jinja
    You are a helpful assistant specializing in scientific facts.
    Using the following context, please answer the user's question.

    Context:
    {% for doc in context_docs %}
    - {{ doc }}
    {% endfor %}

    Question: {{ query }}

    Answer:
    ```
  - `[x]` **Load Prompts Dynamically:** Your generation logic should load and render these templates, passing in the query and retrieved documents. This makes it easy to create `v2.jinja`, `v3.jinja`, etc., and switch between them in your configuration to test different prompt engineering techniques.

### **Step 3: Engineer an Efficient and Observable Tool Calling System**

This design decouples the LLM's decision-making from the actual execution of tools, which is key to solving slowness and observability issues.

  - `[x]` **Centralize Tool Definition:**

      - Create a `src/ir_core/tools/` directory.
      - Define your tools here, such as `retrieval_tool.py`. Each tool should be a simple Python function with a clear definition (name, description, arguments) that can be easily converted to the format your LLM needs (e.g., OpenAI Function Calling JSON schema).
      - The key tool for this project is `scientific_search(query: str)`.

  - `[x]` **Implement a Tool Dispatcher:**

      - Create a class or module called the `ToolDispatcher`. Its only job is to receive a tool name and arguments (as decided by the LLM) and execute the corresponding Python function.
      - **Benefit:** The LLM's role is limited to generating a structured request (e.g., `{"tool": "scientific_search", "arguments": {"query": "What is the capital of South Korea?"}}`). It doesn't wait for the tool to run.

  - `[ ]` **Establish an Observable Workflow:** This is how you'll monitor what's happening.

      - **Generate a Trace ID:** For every incoming user request, generate a unique ID (e.g., `trace_id = "req_123"`).
      - **Use Structured Logging:** Use Python's `logging` module to log every step of the process as a structured JSON object, always including the `trace_id`.

    <!-- end list -->

    ```json
    {"timestamp": "...", "level": "INFO", "trace_id": "req_123", "event": "llm_call_start", "prompt": "..."}
    {"timestamp": "...", "level": "INFO", "trace_id": "req_123", "event": "llm_decision", "tool_to_call": "scientific_search"}
    {"timestamp": "...", "level": "INFO", "trace_id": "req_123", "event": "tool_execution_start", "tool": "scientific_search"}
    {"timestamp": "...", "level": "INFO", "trace_id": "req_123", "event": "tool_execution_complete", "result": "..."}
    ```

      - **Benefit:** You can now easily filter your logs for a single `trace_id` to see the entire lifecycle of a request, measure the time spent in each step (LLM vs. tool), and debug failures precisely.

  - `[ ]` **(Advanced) Optimize Tool Execution:**

      - **Asynchronous Execution:** If an LLM could decide to call multiple independent tools, design the `ToolDispatcher` to run them in parallel using `asyncio`.
      - **Caching:** Use the existing Redis infrastructure to cache the results of deterministic tool calls. If the `scientific_search` tool is called with the exact same query twice, the second result can be served instantly from the cache instead of hitting Elasticsearch.