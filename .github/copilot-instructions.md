# AI Coding Assistant Instructions for Information Retrieval RAG Project

## Project Overview
This is a modular RAG (Retrieval-Augmented Generation) system for scientific question answering. The codebase implements a clean architecture with separate concerns for retrieval, generation, orchestration, and tooling.

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
poetry run python scripts/cli_menu.py
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
<parameter name="filePath">/home/wb2x/workspace/information_retrieval_rag/.github/copilot-instructions.md