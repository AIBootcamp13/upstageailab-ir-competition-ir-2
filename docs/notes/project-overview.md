# Information Retrieval Project Overview

## 1. Competition Overview

This project focuses on implementing **Retrieval Augmented Generation (RAG)**. The core task is to process multi-turn questions about scientific common sense, extract relevant reference documents using retrieval technology, and then generate appropriate answers based on these references.

### 1.1 Key Aspects

*   **Dialogue Style**: The system should handle colloquial, conversational dialogue formats.
*   **Dialogue Topic**: The scope of conversations is limited to scientific common sense topics.
*   **LLM Used**: The project utilizes the **GPT-3.5** version, with the baseline specifically employing **GPT-3.5 Turbo 1106**.
*   **Search Engine**: **Elastic Search** is used for building the search engine component to extract references.

### 1.2 Competition Goals

The competition aims to provide participants with experience in:

*   **Document Extraction**: Implementing components that can extract suitable documents using commercial vector encoders, vector databases, or search engines.
*   **Query Analysis and Answer Generation**: Utilizing LLM features like **prompt engineering** and **function calling** to analyze queries and produce accurate answers based on the retrieved references.
*   **RAG Pipeline Construction**: Building a complete RAG pipeline to develop a backend system for a conversational interface.

### 1.3 RAG System Architecture Example

A typical RAG system, as exemplified in this project, consists of two main flows:

*   **Ingest Flow**: This pipeline handles document indexing. For this competition, it's a one-time indexing process. It involves preprocessing documents, generating embeddings using a vector encoder, and storing them in a vector database or search engine. The choice of search engine (sparse or dense retrieval) is flexible, prioritizing performance.
*   **Query Flow**: This flow processes user queries and generates responses.
    *   When a user inputs a query, the LLM first determines if it's a scientific common sense question requiring document retrieval.
    *   If not, the LLM generates a direct response without consulting the search engine (e.g., for general small talk).
    *   If it is a scientific question, the LLM creates a **standalone query** suitable for searching.
    *   This standalone query is then used by the search engine to extract relevant documents.
    *   (Optional) The extracted documents may undergo post-processing or reranking.
    *   Finally, the LLM uses these retrieved documents as references, along with the original query, to generate the final answer.

### 1.4 Evaluation Method

The primary evaluation metric for this competition is **Mean Average Precision (MAP)**.

*   **MAP Definition**: MAP is the average of Average Precision (AP) across multiple queries. AP, in turn, represents the area under the Precision-Recall curve.
*   **Reasons for MAP Selection**:
    *   It effectively assigns higher weight to errors in top-ranked results, enhancing evaluation precision.
    *   Relevance in the project's evaluation set is predominantly binary (a document is either relevant or not).
    *   It considers scores for all relevant documents within the top N results and accounts for their position, making it suitable when there might be multiple relevant documents for a single query.
*   **Special Consideration for Query Analysis Performance**:
    *   Since the final output is an answer, not just extracted documents, the MAP calculation logic is slightly modified to assess the LLM's query intent understanding.
    *   For questions outside the scientific common sense domain, document extraction is unnecessary. If the LLM correctly identifies such a query and the search result list is empty (matching an empty ground truth), it receives full points (1 point); otherwise, 0 points. This encourages accurate classification of query intent.

## 2. Data Context

The project utilizes two main datasets: one for indexing and another for evaluation.

### 2.1 Indexing Target Documents

*   **Quantity**: **4,272** documents.
*   **Content**: These documents cover educational and scientific common sense topics.
*   **Fields**:
    *   `docid`: A unique identifier for each document, in UUID format.
    *   `src`: The original source or origin of the document.
    *   `content`: The actual text content of the document.
*   **Origin**: The `content` is derived from **MMLU** and **ARC** datasets, which are part of the Ko-H4 data available on Hugging Face's Open Ko LLM Leaderboard.
    *   **MMLU**: Developed by Facebook AI Research, this dataset includes diverse knowledge from humanities, social sciences, and professional fields, featuring multiple-choice questions to measure reasoning and knowledge comprehension.
    *   **ARC**: Developed by the Allen Institute for AI, this dataset comprises multiple-choice questions based on science education content, designed to measure multi-stage reasoning abilities.
*   **Preprocessing**: Originally structured as question-and-answer pairs, these were transformed into single, combined documents using **GPT-4** for this competition, effectively paraphrasing the Q&A into a coherent text.

### 2.11 Sample Data

```json
documents.jsonl

{"docid": "42508ee0-c543-4338-878e-d98c6babee66", "src": "ko_mmlu__nutrition__test", "content": "건강...다."}
...
---
eval.jsonl
{"eval_id": 78, "msg": [{"role": "user", "content": "나무의 분류에 대해 조사해 보기 위한 방법은?"}]}
...
```

### 2.2 Evaluation Data

*   **Quantity**: **220** conversational natural language messages.
*   **Fields**:
    *   `eval_id`: A unique identifier for each evaluation item.
    *   `msg`: A list representing the conversation history between the user and the assistant.
*   **Types of Queries**:
    *   **20 multi-turn conversational questions**: These involve multiple exchanges to arrive at the core query.
    *   **20 everyday conversational messages**: These are non-scientific questions or general small talk (e.g., "Who are you?" or "I'm excited you answered well") for which document retrieval is not expected.
    *   The remaining messages are general natural language questions primarily related to scientific common sense.

## 3. Other Useful Information (Major Approach Methods)

To enhance performance in the information retrieval project, three main approaches are suggested:

### 3.1 Prompt Engineering

This approach focuses on **optimizing the queries** sent to the search engine by improving the LLM's ability to analyze query intent and extract a standalone query.

*   **Refining Instructions**: The initial instructions and function calling definitions in the baseline code are basic. Significant improvements can be made by creating more precise and effective definitions.
*   **Language Considerations**: While English often shows strong LLM performance, considering that the target documents are in Korean, converting the function calling descriptions (especially for the `standalone_query` parameter) to Korean might yield better results.
*   **Search Engine Specific Optimization**: The definition of function calling, particularly for generating the standalone query, should be tailored to the specific search engine being used. For instance, if a sparse retrieval (inverse index-based) engine is used, a standalone query that is more keyword-centric might be more effective than a natural language sentence.

### 3.2 Retrieval Model Enhancement

This involves **optimizing document extraction performance** by improving the search engine itself or the models it uses.

*   **Dense Retrieval**: Explore using **dense retrieval** (e.g., embedding-based search) as an alternative to the baseline's sparse retrieval. Although sparse retrieval performed better with the provided baseline data, a more advanced or higher-performing embedding generation model (vector encoder) could change this outcome.
*   **High-Performance Embedding Models**: Implement and utilize state-of-the-art embedding generation models (vector encoders) for both indexing and searching.
*   **ColBERT for Domain Optimization**:
    *   Consider using the **ColBERT model** to optimize performance specifically for the scientific common sense domain.
    *   This involves constructing a **pseudo-train dataset** from the provided document collection and then training the ColBERT model. While this incurs additional model training costs and time, successful training can lead to a highly performant search engine.

### 3.3 Reranking and Post-processing

This method aims to **optimize search results** by applying post-processing logic to the initially retrieved candidates.

*   **Extracting Ample Candidates**:
    *   **Increasing Top-K**: Instead of directly extracting the final required number of documents (e.g., 3), initially retrieve a larger pool of candidates (e.g., 5, 10, or 20 documents).
    *   **Multiple Search Engines**: Utilize multiple search engines simultaneously, such as combining **sparse and dense retrieval**, to leverage the strengths of each.
    *   **Generating Multiple Queries**: Create several variations of a query and send them to the search engine in parallel, then consolidate and filter the results.
*   **LLM for Reranking or Filtering**:
    *   **Limitations of Search Engines**: Standard search engines rank documents primarily based on similarity, which might not perfectly capture the contextual relevance to the query.
    *   **LLM Advantage**: An LLM, with its strong semantic understanding, can be used to **rerank** the initial set of candidates or **filter out** irrelevant documents.
    *   **Cost-Benefit Analysis**: While effective, using an LLM for reranking can be computationally intensive and time-consuming if the candidate pool is too large. It's crucial to select an appropriate size for the candidate set (e.g., 5 to 10 documents) to manage time and cost efficiently.

By exploring these advanced approaches, participants can further enhance the performance and accuracy of their RAG pipelines for the information retrieval project.

