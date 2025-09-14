CI guide
========

This project uses a minimal GitHub Actions workflow to provide fast feedback during active development and refactors.

What runs in CI
----------------
- A lightweight job that installs dependencies via Poetry and runs the unit tests (`poetry run pytest -q`).

Why minimal
----------
Active refactoring changes the repository layout frequently. A small, fast CI job helps catch regressions while keeping turnaround quick. Longer-running integration tests should be run manually or on a scheduled workflow to avoid blocking rapid feature work.

Run tests locally
-----------------
1. Install dependencies with Poetry:

```bash
poetry install
```

2. Run unit tests:

```bash
poetry run pytest -q
```

Optional: run integration tests (requires local services like Elasticsearch/Redis):

```bash
RUN_INTEGRATION=1 poetry run pytest -q -m integration
```

Recommended branch policy
-------------------------
- Require the `CI` workflow to pass on pull requests to protected branches (e.g., `main`/`master`).
- Allow developers to merge feature branches if tests pass locally; require CI only for merges into protected branches to keep iteration fast.

Next steps (recommended)
------------------------
- Add lint/type checks (ruff, mypy) as optional steps, then gradually require them on main.
- Add `pre-commit` hooks to catch issues earlier.
- Add a scheduled workflow for heavy integration tests.

Pre-commit
----------
We recommend using `pre-commit` to run formatters and linters locally before commits. To install and enable hooks:

```bash
poetry add --dev pre-commit
pre-commit install
pre-commit run --all-files
```

The repository includes a basic `.pre-commit-config.yaml` that runs `black`, `ruff`, and `isort`.
