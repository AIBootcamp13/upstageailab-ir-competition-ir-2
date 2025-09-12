### Assessment of scripts Directory (COMPLETED)
The scripts folder contains 20+ files (mixing Python scripts, shell scripts, and a __pycache__ directory), with no subfolders. Key observations:
- **Diverse purposes**: Files range from core execution (e.g., running RAG pipelines) to data processing, evaluation, infrastructure setup, and utilities. This flat structure makes it hard to navigate and understand responsibilities.
- **Inconsistent naming and structure**: Some files use underscores (e.g., run_rag.py), others hyphens (e.g., `run-local.sh`). No clear grouping by function.
- **Mixed languages**: Python scripts dominate, but shell scripts are interspersed, which could confuse users.
- **Potential duplication/reusability**: Many scripts import from src and use Hydra configs, indicating some logic could be modularized further. However, scripts are standalone entry points, so they should remain here but be better organized.
- **Maintenance issues**: No clear documentation per script, and the flat list grows unwieldy. References to these scripts (e.g., in configs or other files) may break if paths change.
- **Size and scope**: With 20+ files, reorganization is feasible in phases, starting with moving files into subfolders.

Overall, the directory lacks structure, leading to poor discoverability and potential for errors (e.g., accidentally running the wrong script).

### Suggestions for Broader Codebase Maintainability
Before diving into the scripts plan, here are high-level recommendations to make the entire codebase more maintainable and understandable:
- **Modularize reusable code**: Move shared logic (e.g., common imports, utilities like `_add_src_to_path()`) from scripts into src. For example, create `src/scripts_utils.py` for shared helpers.
- **Standardize naming and structure**: Use snake_case for Python files, kebab-case for shell scripts. Enforce via a linter (e.g., add to pyproject.toml or use pre-commit hooks).
- **Add documentation and type hints**: Ensure all scripts have docstrings, and use type hints in Python files. Update README.md with a "Scripts" section explaining each subfolder and how to run scripts.
- **Configuration consistency**: All scripts use Hydra, which is good—ensure configs in conf are well-documented and version-controlled.
- **Testing and validation**: Add unit tests in tests for script logic (e.g., test data processing functions). Use smoke tests (like `smoke_test.py`) as integration checks.
- **Dependency management**: Leverage Poetry for all Python deps; avoid inline installs in scripts.
- **CI/CD integration**: Add GitHub Actions or similar to run key scripts (e.g., evaluation) on PRs.
- **Version control best practices**: Use branches for refactoring; commit reorganization in small, reviewable changes.
- **Code quality tools**: Integrate tools like Black (formatting), Flake8 (linting), and MyPy (type checking) to enforce consistency.

### Reorganization Plan for scripts
The plan breaks the task into phases, focusing on logical grouping by purpose. This creates subfolders for better navigation while minimizing disruption. Estimated effort: 1-2 hours per phase, testable via smoke runs.

#### Phase 1: Create Subfolders and Categorize Files
- **Goal**: Establish structure without moving files yet. Review for any dependencies.
- **Steps**:
  1. Create subfolders: `execution/`, `evaluation/`, data, `infra/`, `maintenance/`.
  2. Categorize files (based on my review):
     - **execution/** (Core runtime scripts for users/developers to run the system):
       - run_rag.py (Runs RAG pipeline).
       - run_query.py (CLI for queries).
       - `run-local.sh` (Local setup runner).
     - **evaluation/** (Scripts for testing, validating, and benchmarking):
       - evaluate.py (Full evaluation with WandB).
       - `validate_retrieval.py` (Retrieval validation).
       - `validate_domain_classification.py` (Domain classification checks).
       - `smoke_test.py` (Python smoke tests).
       - `smoke-test.sh` (Shell smoke tests).
     - **data/** (Data analysis, processing, and transformation):
       - analyze_data.py (Dataset statistics).
       - `check_duplicates.py` (Duplicate detection).
       - create_validation_set.py (Generate validation data).
       - `transform_submission.py` (Submission formatting).
       - `trim_submission.py` (Submission trimming).
     - **infra/** (Infrastructure and environment setup):
       - start-elasticsearch.sh (Start ES).
       - `start-redis.sh` (Start Redis).
       - `cleanup-distros.sh` (Cleanup scripts).
     - **maintenance/** (Utilities for upkeep, demos, and misc tasks):
       - `reindex.py` (Reindexing).
       - `swap_alias.py` (Alias swapping).
       - `parallel_example.py` (Parallel processing example).
       - `demo_ollama_integration.py` (Ollama demo).
  3. Check for dependencies: Grep for references to these scripts in conf, src, or other files (e.g., `grep -r "scripts/" .`). Update paths if needed.
  4. Test: Run a few scripts to ensure they still work.

#### Phase 2: Move Files and Update References
- **Goal**: Physically move files and fix any broken imports/paths.
- **Steps**:
  1. Move files into their subfolders using `mv` (e.g., `mv run_rag.py execution/`).
  2. Update any hardcoded paths in configs or code (e.g., if config.yaml references run_rag.py, change to run_rag.py).
  3. Update shell scripts if they reference other scripts (e.g., `run-local.sh` might call start-elasticsearch.sh).
  4. Clean up: Remove __pycache__ or regenerate it.
  5. Test: Run scripts from new locations; check for import errors.

#### Phase 3: Enhance Documentation and Quality
- **Goal**: Make the reorganized structure user-friendly.
- **Steps**:
  1. Add a `scripts/README.md` with an overview of subfolders and brief descriptions of each script.
  2. Improve individual script docs: Add usage examples, required args, and dependencies.
  3. Add a simple script (e.g., `scripts/list_scripts.py`) to list all scripts with descriptions.
  4. Run linters/formatters on moved files.
  5. Test: Full smoke test suite.

#### Phase 4: Follow-Up and Integration
- **Goal**: Integrate with broader codebase improvements.
- **Steps**:
  1. Move shared code to src (e.g., extract `_add_src_to_path()` to `src/utils/scripts.py`).
  2. Add tests for key scripts in `tests/scripts/`.
  3. Update CI to reference new paths.
  4. Monitor for issues post-reorg.
