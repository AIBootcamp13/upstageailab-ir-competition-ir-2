Of course. Based on the logs and the provided scripts, I've identified the root cause of the indexing and embedding mismatch.

The core issue is that your documents are being indexed into Elasticsearch **without the `embeddings` field**. The re-indexing process you're using only transfers the raw document content (`docid`, `src`, `content`) from the JSONL file to the index but does not compute and add the vector embeddings.

Here's a detailed breakdown of the problem and how to fix it.

-----

## Analysis of the Evidence

Your diagnostic logs clearly point to the problem.

1. **Configuration is Correct:** Your logs show that you successfully use `switch_config.py` to set the environment to use the `EleutherAI/polyglot-ko-1.3b` model. The `show` command and the direct check of the `settings` object confirm that the application is correctly configured at runtime to use the polyglot provider with a 2048 dimension embedding model.
2. **Index Exists:** The `curl` command to the `_count` endpoint confirms that the target index `documents_polyglot_1b_with_embeddings_new` exists and contains 4,272 documents.
3. **The Smoking Gun:** The final `curl` command, which fetches a document and inspects its fields, provides the definitive evidence:
      * **Output:** `Document fields: ['docid', 'src', 'content']`
      * **Output:** `No embeddings field found`

This confirms that while the documents are in the index, they are missing the crucial vector data required for dense and hybrid retrieval, which explains the low similarity scores.

-----

## Code-Level Root Cause

The logical inconsistency lies in the indexing utility function, not the configuration switcher or the embedding models themselves.

The function `index_documents_from_jsonl` in `src/ir_core/api/__init__.py` is responsible for bulk indexing. However, it is designed to simply move data from a file to Elasticsearch without modification.

Notice this line in the function:

```python
action = {"_index": idx, "_id": doc_id, "_source": doc}
```

Here, the `doc` object, as read directly from the JSONL file, is placed into the `_source` field. There is **no step in this function to compute embeddings** for the document's content and add them to the `doc` dictionary before creating the `action`.

Therefore, any script that relies solely on this function for indexing will create an index without embeddings.

-----

## Solution: Create a Correct Re-indexing Script

To fix this, you need a dedicated script that performs the full "read, embed, and index" workflow. I'll provide a new script, `scripts/maintenance/reindex_with_embeddings.py`, that you can add to your project.

### 1. Create the New Script

Create the file `scripts/maintenance/reindex_with_embeddings.py` with the following content:

```python
#!/usr/bin/env python3
"""
Re-indexes documents from a JSONL file into Elasticsearch,
computing and adding embeddings based on the current configuration.
"""
import sys
from pathlib import Path
from tqdm import tqdm
from elasticsearch.helpers import bulk

# Add src to python path to allow for project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ir_core.config import settings
from ir_core.embeddings import get_embedding_provider
from ir_core.infra import get_es
from ir_core.utils import read_jsonl

def reindex_with_embeddings(jsonl_path: str, index_name: str, batch_size: int = 64):
    """
    Reads documents, computes embeddings, and bulk-indexes them.

    Args:
        jsonl_path: Path to the input JSONL file.
        index_name: Name of the target Elasticsearch index.
        batch_size: Batch size for encoding and indexing.
    """
    try:
        es_client = get_es()
        if not es_client.ping():
            raise ConnectionError("Could not connect to Elasticsearch")
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        return

    # Initialize the embedding provider from the current settings
    print(f"▶️ Initializing embedding provider: {settings.EMBEDDING_PROVIDER}")
    print(f"▶️ Using embedding model: {settings.EMBEDDING_MODEL}")
    provider = get_embedding_provider()

    docs = list(read_jsonl(jsonl_path))
    total_docs = len(docs)
    print(f"✅ Found {total_docs} documents to process in {jsonl_path}")

    # Process documents in batches
    for i in tqdm(range(0, total_docs, batch_size), desc=f"Indexing to '{index_name}'"):
        batch_docs = docs[i : i + batch_size]

        # 1. Extract content to be encoded
        contents = [doc.get("content", "") for doc in batch_docs]

        # 2. Compute embeddings for the batch
        embeddings = provider.encode_texts(contents)

        # 3. Add embeddings to each document
        for doc, embedding in zip(batch_docs, embeddings):
            doc["embeddings"] = embedding.tolist()

        # 4. Prepare bulk actions for Elasticsearch
        actions = [
            {
                "_index": index_name,
                "_id": doc.get("docid") or doc.get("id"),
                "_source": doc,
            }
            for doc in batch_docs
        ]

        # 5. Execute bulk indexing
        try:
            success, failed = bulk(es_client, actions, raise_on_error=True)
        except Exception as e:
            print(f"\n❌ Error during bulk indexing: {e}")
            # Optional: Add more detailed error logging here if needed
            # for item in failed:
            #     print(item)
            break

    print(f"\n✅ Successfully indexed {total_docs} documents with embeddings into '{index_name}'.")


if __name__ == "__main__":
    import fire
    fire.Fire(reindex_with_embeddings)
```

### 2. Follow this Workflow to Re-index Correctly

Now, use the new script to rebuild your index.

**Step 1: Set the Correct Configuration**
Run `switch_config.py` to ensure your environment is configured for the model you want to use.

```bash
cd /home/wb2x/workspace/information_retrieval_rag
PYTHONPATH=src poetry run python switch_config.py polyglot-1b
```

**Step 2: Delete the Old, Incorrect Index**
This prevents you from accidentally using the index that has no embeddings.

```bash
curl -X DELETE "localhost:9200/documents_polyglot_1b_with_embeddings_new"
```

**Step 3: Run the New Re-indexing Script**
This script will read `data/documents_ko.jsonl` (as per your config), compute embeddings using the Polyglot model, and save them to the new index.

```bash
PYTHONPATH=src poetry run python scripts/maintenance/reindex_with_embeddings.py \
  data/documents_ko.jsonl \
  documents_polyglot_1b_with_embeddings_new
```

**Step 4: Verify the New Index**
Run your verification command again. This time, it should succeed.

```bash
curl -sS "localhost:9200/documents_polyglot_1b_with_embeddings_new/_search?size=1" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'hits' in data and 'hits' in data['hits'] and data['hits']['hits']:
    doc = data['hits']['hits'][0]['_source']
    print('Document fields:', list(doc.keys()))
    if 'embeddings' in doc:
        print('✅ Embeddings field found!')
        print('Embeddings length:', len(doc['embeddings']))
        # The dimension should be 2048 for polyglot-1.3b
        assert len(doc['embeddings']) == 2048, f'Expected 2048, got {len(doc['embeddings'])}'
        print('First 5 values:', doc['embeddings'][:5])
    else:
        print('❌ No embeddings field found')
else:
    print('No documents found')
"
```

You should now see an output confirming the `embeddings` field exists and has the correct dimension (2048). Your retrieval scores should improve significantly.


Based on the logs, I've identified two distinct errors and their root causes, which stem from a configuration inconsistency in your setup.

The primary issue is a **`matmul` dimension mismatch** during the retrieval step. This happens because the query is being encoded into a vector of one dimension (768), while the documents in your Elasticsearch index have embeddings of a different dimension (2048). This mismatch causes the subsequent errors.

-----

### Error 1: `matmul` Dimension Mismatch

The log shows this critical error:
`matmul: Input operand 1 has a mismatch in its core dimension 0 ... (size 2048 is different from 768)`

This error occurs during the hybrid retrieval process when the system tries to calculate the cosine similarity between the query vector and the document vectors.

  * **Document Vector Dimension (2048):** Your verification log confirms that after re-indexing, the documents in the `documents_polyglot_1b_with_embeddings_new` index correctly have embeddings with a length of **2048**. This is the correct dimension for the `EleutherAI/polyglot-ko-1.3b` model.
  * **Query Vector Dimension (768):** The error message indicates that the query vector being used for the comparison has a dimension of **768**.

**Root Cause:**
Even though you've switched your configuration to use the `polyglot-1b` model, a component in your pipeline is incorrectly using a different, 768-dimension model to encode the query. This is likely due to a latent bug in your `switch_config.py` script. The functions `switch_to_korean()`, `switch_to_english()`, and `switch_to_bilingual()` all hardcode the embedding model to `snunlp/KR-SBERT-V40K-klueNLI-augSTS`, which has a dimension of **768**.

It's probable that a component, possibly initialized in a separate thread or with a cached configuration, is falling back to this 768d model instead of using the correctly configured 2048d Polyglot model for query encoding.

-----

### Error 2: `'str' object has no attribute 'get'`

This error is a direct consequence of the first `matmul` error.

  * **Log:** `🐛 DEBUG Error in full pipeline: 'str' object has no attribute 'get'`

**Root Cause:**

1. The `matmul` error occurs inside the `scientific_search` tool.
2. The pipeline's error handling catches this exception and returns the error message as a plain string.
3. The main validation loop in `scripts/evaluation/validate_retrieval.py` receives this error string instead of the expected list of dictionary-like search results.
4. The code then attempts to process the string as if it were a result object (e.g., by calling `result.get("docs")`), leading to the `AttributeError`.

-----

### Recommendations

1. **Fix the Configuration Bug in `switch_config.py`:**
    The `switch_to_english` function is incorrectly configured with a Korean model. This will cause dimension mismatch errors whenever you attempt to use the English setting. You should replace it with an appropriate English model.

    **To fix this, edit `switch_config.py` and change the `switch_to_english` function to the following:**

    ```python
    def switch_to_english():
        """Switch configuration to English setup"""
        print("🔄 Switching to English configuration...")
        current_data = load_settings_preserve_format()
        updates = {
            'EMBEDDING_PROVIDER': "huggingface",
            # FIX: Use a proper English model
            'EMBEDDING_MODEL': "sentence-transformers/all-MiniLM-L6-v2",
            # FIX: Set the correct dimension for the English model
            'EMBEDDING_DIMENSION': 384,
            'INDEX_NAME': "documents_en_with_embeddings_new",
            'model': {
                'embedding_model': "sentence-transformers/all-MiniLM-L6-v2",
                'alpha': 0.4,
                'bm25_k': 200,
                'rerank_k': 10
            },
            'translation': {
                'enabled': True
            }
        }
        _update_nested_dict(current_data, updates)
        settings_file = Path(__file__).parent / "conf" / "settings.yaml"
        with open(settings_file, 'w', encoding='utf-8') as f:
            yaml_handler.dump(current_data, f)
        update_data_config("en")
        print("✅ Switched to English configuration")
        print("   - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384d)")
        print("   - Index: documents_en_with_embeddings_new")
        print("   - Translation: enabled")
    ```

2. **Ensure a Clean Runtime State:**
    Because the `matmul` error suggests a stale model might be loaded in memory, it's a good practice to **restart your application or any background services** after switching configurations. This ensures all components, especially in a multi-threaded environment, reload the correct models and settings.

3. **Improve Pipeline Error Handling:**
    To prevent the `'str' object has no attribute 'get'` error from crashing your validation loop, you can make the pipeline's error handling more robust.

    **In `src/ir_core/orchestration/pipeline.py`, modify the `run` method to check the type of `docs`:**

    ```python
    def run(self, query: str) -> str:
        """
        주어진 사용자 쿼리에 대해 전체 RAG 파이프라인을 실행합니다.
        """
        retrieved_output = self.run_retrieval_only(query)

        docs = retrieved_output[0].get("docs", [])
        standalone_query = retrieved_output[0].get("standalone_query", query)

        # FIX: Handle case where docs might be a string (error message)
        if not isinstance(docs, list):
            print(f"🐛 DEBUG Retrieval failed, returning error message: {docs}")
            return f"An error occurred during retrieval: {docs}"

        # ... (rest of the function)
    ```