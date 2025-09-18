# AI Coding Assistant Instructions for Information Retrieval RAG Project

## Project Overview
This is a modular RAG (Retrieval-Augmented Generation) system for scientific question answering. The codebase implements a clean architecture with separate concerns for retrieval, generation, orchestration, and tooling.

CRITICAL rules for assistants
- ALWAYS run scripts via Poetry: use `poetry run python ...` (never bare python)
- Elasticsearch index MUST predefine `embeddings` as `dense_vector` with correct `dims` (no dynamic mapping)
- Use Nori analyzer for Korean text fields; index creation helpers already configure this
- Before running evaluations or reindex: run the pre-flight validator (see below)

## Architecture Patterns

### Modular Structure
- **Facade Pattern**: `src/ir_core/api.py` provides the main API surface with minimal imports to avoid side effects
- **Factory Pattern**: `src/ir_core/generation/__init__.py` creates generators based on config (`openai` vs `ollama`)
- **Configuration Hierarchy**: Pydantic BaseSettings + Hydra YAML configs with environment variable overrides

### Key Modules
- `retrieval/`: BM25 sparse + dense vector search with Redis caching
- `generation/`: Abstract base class with OpenAI/Ollama implementations
- `orchestration/`: Pipeline coordination with query rewriting and tool calling
- `tools/`: Pydantic-validated tool schemas for retrieval operations
- `infra/`: Elasticsearch client with conservative timeouts (5s, no retries)

## Critical Workflows

### Development Setup
```bash
# Always use Poetry for dependency management
poetry install

# Copy environment template
cp .env.example .env

# Start local infrastructure
./scripts/execution/run-local.sh start

# Use interactive CLI for operations
poetry run python cli_menu.py
```

### Running Scripts
```bash
# ALWAYS use Poetry for script execution to avoid dependency errors
# Pattern: poetry run python scripts/... (NOT just python)
poetry run python scripts/evaluation/validate_retrieval.py

# For scripts requiring PYTHONPATH
PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py
```

**Critical**: Never use bare `python` commands - always prefix with `poetry run` to ensure proper virtual environment activation and dependency resolution.

### Configuration Switching and Pre-flight Validation

This project supports multiple embedding providers with different dimensions. Always validate the current provider dimension matches the target index mapping before heavy runs.

#### When to Switch Configurations
- **Korean submissions**: Use Korean model (768d) with Korean index
- **English submissions**: Use English model (384d) with English index
- **Bilingual submissions**: Use Korean model (768d) with bilingual index
- **Before running evaluations**: Always verify current configuration matches your target

#### Pre-flight Validator (must-run)
```bash
# Validate provider dimension vs index mapping (fails fast on mismatch)
PYTHONPATH=src poetry run python scripts/indexing/validate_index_dimensions.py --index "$INDEX" \
    --provider "${EMBEDDING_PROVIDER:-auto}"

# Example (current default: Polyglot-Ko 2048d)
PYTHONPATH=src poetry run python scripts/indexing/validate_index_dimensions.py --index docs-ko-polyglot-1b-d2048-20250918
```

#### Current default
- Provider: Polyglot-Ko (`EleutherAI/polyglot-ko-1.3b`), dims=2048
- Index: `docs-ko-polyglot-1b-d2048-20250918`

#### Critical Warnings
- **❌ NEVER mix dimensions**: 384d English model cannot search 768d Korean index
- **❌ NEVER use wrong index**: Always ensure index exists before running evaluations
- **❌ NEVER skip configuration check**: Always verify current config with `switch_config.py` status
- **⚠️ Index creation required**: Create indexes with matching embedding model before use

#### Best Practices to Avoid Wasted Resources
1. **Check current configuration** before starting work:
   ```bash
   PYTHONPATH=src poetry run python switch_config.py status
   ```

2. **Create required indexes** with correct embedding model:
   ```bash
   # Korean index (768d)
   PYTHONPATH=src poetry run python switch_config.py korean
   PYTHONPATH=src poetry run python scripts/maintenance/reindex.py data/documents_ko.jsonl --index documents_ko_with_embeddings_new

   # English index (384d)
   PYTHONPATH=src poetry run python switch_config.py english
   PYTHONPATH=src poetry run python scripts/maintenance/reindex.py data/documents_bilingual.jsonl --index documents_en_with_embeddings_new

   # Bilingual index (768d)
   PYTHONPATH=src poetry run python switch_config.py bilingual
   PYTHONPATH=src poetry run python scripts/maintenance/reindex.py data/documents_bilingual.jsonl --index documents_bilingual_with_embeddings_new
   ```

3. **Verify provider/index BEFORE evaluation**:
    ```bash
    # Quick dims + analyzer check
    PYTHONPATH=src poetry run python scripts/indexing/validate_index_dimensions.py --index docs-ko-polyglot-1b-d2048-20250918 --check-analyzer
    ```

4. **Use debug mode** for quick validation before full runs:
   ```bash
   PYTHONPATH=src poetry run python scripts/evaluation/evaluate.py --config-dir conf pipeline=qwen-full model.alpha=0.4 limit=5
   ```

### Script Discovery
```bash
# List all available scripts with descriptions
python scripts/list_scripts.py

# Interactive CLI menu for common operations
python scripts/cli_menu.py
```

**Key Script Categories:**
- `evaluation/`: Validation, testing, and evaluation scripts
- `data/`: Data processing and analysis tools
- `execution/`: Core pipeline execution scripts
- `infra/`: Infrastructure management (ES, Redis)
- `maintenance/`: Indexing, alias management, demos
 - `indexing/`: Index creation, mapping validation, and embedding reindex utilities

**Test Scripts Available:**
- `scripts/evaluation/smoke_test.py`: System health verification
- `scripts/test_*`: Various test utilities and report generators
- Integration tests in `tests/` directory (require `RUN_INTEGRATION=1`)

**Script Documentation:**
- Script descriptions maintained in `scripts/list_scripts.py` (update when adding new scripts)
- Use `python scripts/list_scripts.py` to see all available scripts with descriptions

## Configuration System

### Testing
```bash
# Unit tests (fast, no external deps)
poetry run pytest

# Integration tests (require ES/Redis)
RUN_INTEGRATION=1 poetry run pytest -m integration

# Test utilities and report generators
python scripts/test_report_generator.py
python scripts/test_visualizer.py
```

**Test Script Categories:**
- `tests/test_*.py`: Unit and integration tests
- `scripts/evaluation/smoke_test.py`: System health verification
- `scripts/test_report_generator.py`: Generate test reports
- `scripts/test_visualizer.py`: Visualize test results

## Configuration System

### Settings Loading Priority
1. `conf/settings.yaml` (defaults)
2. Environment variables (highest priority)
3. `.env` file (loaded by pydantic-settings)

### Hydra Config Groups
- `data/`: Dataset configurations
- `model/`: Retrieval parameters (alpha, k values)
- `pipeline/`: Generation settings (openai/ollama, model names)
- `prompts/`: Template paths
- `experiment/`: Optional experiment overrides

## Code Patterns

### Model Loading
```python
# Thread-safe singleton pattern in embeddings/core.py
_tokenizer = None
_model = None
_lock = threading.Lock()

def load_model(name=None):
    global _tokenizer, _model
    with _lock:
        if _tokenizer is None:
            # Load and cache model
```

### Error Handling
```python
# Graceful Redis fallback in retrieval/core.py
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
    redis_client.ping()
except redis.ConnectionError:
    redis_client = None  # Continue without caching
```

### Data I/O
```python
# JSONL utilities in utils/core.py
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
```

### Tool Definitions
```python
# Pydantic schemas in tools/retrieval_tool.py
class ScientificSearchArgs(BaseModel):
    query: str
    top_k: int = 5
```

## Integration Points

### External Services
- **Elasticsearch**: Document indexing/search on `localhost:9200`
- **Redis**: Caching layer on `localhost:6379`
- **OpenAI API**: Generation and query rewriting
- **Ollama**: Local model serving on `localhost:11434`
- **Weights & Biases**: Experiment tracking

### Cross-Component Communication
- Retrieval results passed as `{"hit": es_doc, "score": float}`
- Tool calls use JSON-RPC style dispatch through `tools/dispatcher.py`
- Pipeline coordinates query rewriting → retrieval → generation

## Development Environment & Performance

### Environment Constraints
- **OS**: Ubuntu 20.04
- **Setup**: Ephemeral Docker development environment hosted remotely
- **Privileges**: Limited sudo access (cannot run Docker containers, modify system kernels, etc.)
- **Hardware**: RTX 24GB VRAM GPU, Threadripper 397x CPU (high-performance workstation)

### Performance Priorities
- **Vectorization**: Prefer NumPy/pandas vectorized operations over loops when possible
- **Parallel Processing**: Use ThreadPoolExecutor, multiprocessing, or GPU acceleration for compute-intensive tasks
- **GPU Utilization**: Leverage CUDA/ROCm for embedding computations and model inference when feasible
- **Memory Efficiency**: Design for large datasets with streaming/batching to respect 24GB VRAM limit

### Implementation Guidelines
```python
# Prefer vectorized operations
import numpy as np
# Good: vectorized cosine similarity
similarities = np.dot(query_emb, doc_embs.T)

# Use parallel processing for batch operations
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_batch, batches))
```

## Common Pitfalls

### Environment Setup
- Always run `poetry install` after pulling changes
- Set `PYTHONPATH=src` for script execution
- Check service status with `./scripts/execution/run-local.sh status`

### Configuration
- Changes to `conf/settings.yaml` require restart
- Environment variables override YAML defaults
- Use `hydra.core.hydra_config.HydraConfig` for runtime config inspection
- Do NOT rely on dynamic mapping for `embeddings` — use the provided index creation script or the API that auto-creates mappings with Nori and dense_vector

### Performance
- Embedding models loaded once and cached with threading locks
- Redis caching prevents redundant vector searches
- ES timeouts set conservatively to fail fast in development
- **GPU Optimization**: Leverage RTX 24GB VRAM for batch embedding computations
- **Parallel Processing**: Utilize Threadripper 397x for CPU-intensive tasks (8-16 workers typical)
- **Memory Management**: Design for large datasets with streaming to respect VRAM limits

## File Organization
- `src/ir_core/`: Main package with clean module boundaries
- `scripts/`: Executable scripts (use CLI menu for discovery)
- `conf/`: Hydra configuration groups
- `tests/`: Unit tests + integration tests (marked appropriately)
- `data/`: JSONL datasets for indexing/evaluation</content>
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/.ai/assistant-instructions/copilot-instructions.md