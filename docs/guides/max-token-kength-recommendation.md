### The Core Principle

The main trade-off is **Cost/Speed vs. Completeness/Quality**.
* **Shorter `max_tokens`**: Cheaper, faster, less detailed. Ideal for internal processing steps.
* **Longer `max_tokens`**: More expensive, slower, more comprehensive. Best for the final user-facing output.

---

### Summary Table of Recommendations

| RAG Stage | Task | Recommended `max_tokens` | Rationale |
| :--- | :--- | :--- | :--- |
| **1. Query Processing** | **Query Rewriting** | **50 - 100** | Output should be a concise, focused search query. |
| | **HyDE Generation** | **150 - 300** | Generate a dense, hypothetical paragraph for embedding. |
| | **Sub-Question Generation**| **150 - 250** | Create a list of 3-5 short, distinct questions. |
| **2. Final Answering** | **Standard RAG Answer** | **500 - 1,000** | A balanced, multi-paragraph answer for most Q&A. |
| | **Conversational Chat** | **150 - 400** | A shorter, more natural response to keep dialogue flowing. |
| | **Summarization** | **300 - 700** | Condense retrieved docs into a detailed summary. |
| | **Long-Form Generation** | **2,000 - 4,000+** | For tasks like generating a report section or detailed analysis. |

---

### Detailed Breakdown

#### Stage 1: Query Understanding & Expansion

The goal here is to refine the user's input for better retrieval. The outputs should be short and precise. Using a fast, cheap model like `gpt-4o-mini` is highly recommended for these tasks.

##### 1. Query Rewriting
* **Purpose**: To convert a vague or conversational user query ("tell me about the economy here in Incheon") into a clear, keyword-rich query suitable for a vector database ("What are the key economic indicators and major industries in Incheon, South Korea?").
* **Recommended `max_tokens`**: **50 - 100**
* **Why**: This is more than enough space for a well-formed question or two. A higher limit risks the model adding conversational fluff, which is counterproductive for retrieval.

##### 2. HyDE (Hypothetical Document Embeddings)
* **Purpose**: To generate a fictional, ideal document that answers the user's query. This hypothetical document is then embedded to find real documents that are semantically similar.
* **Recommended `max_tokens`**: **150 - 300**
* **Why**: You need a full paragraph that sounds like a real document chunk. It should be dense with relevant terms and concepts. 150-300 tokens is the sweet spot—long enough to be meaningful but short enough to be fast and cheap.

#### Stage 2: Synthesis & Answering

The goal here is to use the retrieved context to generate the final, user-facing response. The ideal length depends entirely on your application's purpose. It's often best to use a high-quality model like `gpt-4o` for this stage.

##### 1. Standard RAG Answering (General Purpose)
* **Purpose**: To provide a comprehensive answer based on the retrieved documents for a standard Q&A or "explain this" task.
* **Recommended `max_tokens`**: **500 - 1,000**
* **Why**: This is a great default for many applications. It allows for a detailed, multi-paragraph response that can cite sources and explain nuances without being excessively long.

##### 2. Conversational Chatbot Response
* **Purpose**: To provide an answer as part of an ongoing, interactive conversation.
* **Recommended `max_tokens`**: **150 - 400**
* **Why**: Shorter responses feel more natural in a chat. A 1,000-token monologue can feel jarring. This range keeps the conversation snappy and interactive.

##### 3. Summarization
* **Purpose**: The primary task is to summarize the retrieved documents.
* **Recommended `max_tokens`**: **300 - 700**
* **Why**: The length depends on the desired level of detail in the summary. 300 is good for a brief overview, while 700 can provide a more structured summary with multiple key points.

### Important: Managing Your Total Context Window

Remember, `max_tokens` only limits the **generated output**. You must also manage the **input tokens** (system prompt + user query + retrieved context) to ensure the total does not exceed your model's context window (e.g., 128,000 for GPT-4 Turbo).

**Best Practice**: Calculate your input tokens and reserve your desired `max_tokens` for the output. If the input is too long, you must truncate or summarize your retrieved context before sending it to the final generation model.

`Total Tokens = (Prompt Tokens + Context Tokens) + Completion Tokens (max_tokens)`