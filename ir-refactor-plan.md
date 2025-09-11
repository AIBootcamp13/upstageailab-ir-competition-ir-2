# RAG System Improvement Plan

## 1. Overview

This document outlines the plan to enhance the Information Retrieval (RAG) system. The primary goals are to improve performance on multi-turn conversational queries, increase the precision of document retrieval, and establish a more robust framework for experimentation and evaluation. This plan builds upon the existing modular architecture and incorporates advanced techniques for conversation analysis and semantic understanding.

## 2. Guiding Principles

The following principles, adapted from best practices in conversational AI, will guide our development:

* **Explicit Pipeline Stages:** We will move from a single tool-calling decision to a multi-stage pipeline: **1. Intent Classification & Rewriting** -> **2. Topic-Aware Retrieval** -> **3. Contextual Answer Generation**. This improves modularity, debuggability, and performance.
* **Intent-Driven Control Flow:** The system's behavior will be explicitly determined by the user's intent. Scientific queries will trigger the RAG pipeline, while chit-chat will be handled by a separate, conversational response generator.
* **Data-Driven Evaluation:** All significant changes will be measured against a dedicated validation set. We will use MAP for end-to-end retrieval performance and supplement it with component-specific metrics where appropriate.
* **Graceful Fallbacks:** The system will be designed to handle ambiguity and non-scientific queries gracefully, providing helpful, conversational responses rather than failing or returning empty results.

## 3. Actionable Roadmap

### Session 1: Foundational Tooling & Conversation Core

**Goal:** Establish a rigorous experimentation framework and implement the core conversation handling module.

1.  **[ ] Integrate Hydra for Configuration Management:**
    * **Task:** Refactor the configuration loading in scripts like `evaluate.py` and `validate_retrieval.py` to use the Hydra framework.
    * **Benefit:** Enables easy command-line overrides for hyperparameters (`alpha`, `rerank_k`), prompt versions, and model names. Simplifies managing and reproducing experiments.

2.  **[ ] Integrate WandB for Experiment Tracking:**
    * **Task:** Adapt the `wandb_utils.py` script. Integrate `wandb` logging into `validate_retrieval.py` to track hyperparameters, MAP scores, and output artifacts (like the validation results).
    * **Benefit:** Provides a centralized dashboard for comparing experiment results, visualizing performance trends, and ensuring full reproducibility.

3.  **[ ] Implement Explicit Query Analysis Module:**
    * **Task:**
        1.  Create a new prompt, `prompts/query_analyzer.jinja2`, designed to perform two tasks: **Intent Classification** (is it `scientific` or `chit-chat`?) and **Query Rewriting** (if scientific, rewrite into a standalone query).
        2.  Modify the `RAGPipeline`. Before tool calling, make a preliminary call to the LLM with this new prompt.
        3.  Implement control flow based on the intent classification:
            * If `scientific`, use the rewritten query for the retrieval step.
            * If `chit-chat`, bypass retrieval entirely and use the `OllamaGenerator` or `OpenAIGenerator` with the `prompts/conversational_v1.jinja2` template to generate a friendly response.
    * **Benefit:** This directly addresses the core challenge of handling mixed-intent, multi-turn conversations and provides a robust mechanism for generating high-quality standalone queries.

### Session 2: Enhancing Retrieval with Semantic Understanding

**Goal:** Improve retrieval precision by enriching the search index with semantic topic information.

1.  **[ ] Implement Topic Modeling for Document Corpus:**
    * **Task:**
        1.  Create a new script, `scripts/generate_topics.py`, that uses a library like `BERTopic` to analyze all documents in `data/documents.jsonl`.
        2.  The script will output a file mapping each `docid` to a `topic_id` and `topic_keywords`.
        3.  Modify `scripts/reindex.py` to read this mapping and add the topic information as new fields to each document during Elasticsearch indexing.
    * **Benefit:** Creates a semantically structured search index.

2.  **[ ] Implement Topic-Filtered Retrieval:**
    * **Task:**
        1.  In the `RAGPipeline`, after the Query Analysis module generates a standalone query, add a step to classify this query into one of the previously generated topics.
        2.  Modify the `sparse_retrieve` and `dense_retrieve` functions in `ir_core/retrieval/core.py` to accept an optional `topic_id` parameter.
        3.  Update the Elasticsearch queries to use a `filter` clause, restricting the search to only documents with the matching `topic_id`.
    * **Benefit:** Dramatically improves search precision and speed by reducing the search space to only the most relevant subset of documents.

### Session 3: Advanced Evaluation & Refinement

**Goal:** Build a dedicated evaluation suite for the new conversation analysis capabilities and use it to refine the system.

1.  **[ ] Create a Multi-Turn Validation Set:**
    * **Task:** Manually create `data/validation_multurn.jsonl`. Each entry will contain a multi-turn `msg` history, a human-authored "golden" `standalone_query`, and the `ground_truth_doc_id`.
    * **Benefit:** Provides the necessary data to quantitatively measure the performance of the new Query Analysis module.

2.  **[ ] Enhance the Validation Script:**
    * **Task:** Create a new script, `scripts/validate_conversation.py`, that evaluates the system against the multi-turn dataset.
    * **Clarification on Metrics:**
        * **Module-level (Query Rewriter):** The script can optionally compare the generated standalone query to the "golden" query using metrics like sentence similarity (embedding cosine similarity) as a diagnostic tool. This is a "unit test" for the rewriter.
        * **End-to-End (System Level):** The primary success metric remains **MAP**. The script will use the generated standalone query to perform retrieval and calculate the MAP score against the `ground_truth_doc_id`. This measures the real-world impact of the rewriter on retrieval performance.
    * **Benefit:** Provides a robust framework for testing, debugging, and improving the entire conversational RAG pipeline.

### Session 4: Performance Optimization with Asynchronous Execution

**Goal:** Dramatically reduce the runtime of batch processes and lay the groundwork for a high-performance, real-time API.

1.  **[ ] Phase 1: Asynchronous Batch Processing:**
    * **Task:** Refactor `evaluate.py` and `validate_retrieval.py` for concurrent execution.
        1.  Install necessary async libraries (e.g., `httpx` for OpenAI, `elasticsearch-async`, `redis-py`'s async support).
        2.  Convert the core processing logic for a single query into an `async def process_item(item)` function.
        3.  In the main execution block, create a list of concurrent tasks and run them all using `asyncio.gather`.
    * **Benefit:** Massive speed improvements for evaluation and validation runs, enabling faster iteration cycles. This is the highest-impact, lowest-effort async change.

2.  **[ ] Phase 2 (Advanced): Internal Pipeline Asynchronization:**
    * **Task:**
        1.  Refactor the methods within `RAGPipeline` and `hybrid_retrieve` to be `async def`.
        2.  Replace the standard `openai`, `elasticsearch`, and `redis` clients with their async-compatible counterparts throughout the `src/ir_core` library.
        3.  Identify and implement opportunities for concurrent I/O *within* a single pipeline run (e.g., parallel sparse and dense retrieval).
    * **Benefit:** Reduces single-query latency, which is essential for any user-facing application (like a real-time API or chatbot).