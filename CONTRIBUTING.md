Contributing
============

Thank you for contributing! This project uses a lightweight CI setup to balance fast iteration during active refactoring with safety on protected branches. This document outlines recommended branch protection rules, CI expectations, and guidance for contributors performing refactors.

1. Branch protection checklist (recommended)
-----------------------------------------

Protect the primary branches (for example `main` or `master`) with the following rules:

- Require pull request reviews before merging (1+ approver).
- Require status checks to pass. Start by requiring the minimal `CI` job (unit tests). As the repository stabilizes, add the following checks:
  - `CI` (unit tests)
  - `ruff` (lint) and `mypy` (type checks) — optional initially, required later
  - A scheduled integration job (nightly) for heavy tests (do not require on every PR)
- Restrict who can push directly to protected branches (admins or release managers only).
- Enforce linear history (recommended) to keep the commit graph simple.

2. Example GitHub branch-protection policy
-----------------------------------------

Settings for `main` (recommended):

- Require pull request reviews before merging: 1 required
- Require status checks to pass before merging: `CI` required, `ruff` and `mypy` optional at first
- Require branches to be up-to-date before merging (optional)
- Include administrators: yes (or no based on your team's policy)

3. CI expectations and flow
---------------------------

- Fast checks (run on every PR/push): unit tests (`poetry run pytest -q`). Keep these tests small and fast.
- Lint/type checks: added to CI but can be non-blocking initially. Encourage running them locally via pre-commit.
- Integration tests: run them on-demand or on a schedule. Use `RUN_INTEGRATION=1` when invoking pytest locally or in a scheduled workflow.

4. Local developer workflow (recommended)
---------------------------------------

1. Create a feature branch:

```bash
git checkout -b feature/your-descriptive-name
```

2. Run tests and linters locally before pushing:

```bash
poetry install
poetry run pytest -q
# optional: run ruff and mypy
poetry run ruff check .
poetry run mypy src || true
```

3. Use pre-commit hooks to avoid trivial formatting/lint errors:

```bash
poetry add --dev pre-commit
pre-commit install
pre-commit run --all-files
```

4. Push and open a pull request. The CI `CI` job will run automatically.

5. When refactoring large parts of the repo
-----------------------------------------

- Keep changes small and self-contained when possible — one logical refactor per PR.
- If you must rename top-level modules or move packages, include a short migration note in the PR describing how to run tests locally and any manual steps for reviewers.
- Coordinate with reviewers: mark the PR as a draft if the refactor is incomplete and add the `WIP` label.
- Consider adding a temporary CI exemption for complex refactors by asking repository admins to merge the branch after review (use sparingly).

6. Integration tests and services
--------------------------------

This project has integration tests that require external services (Elasticsearch, Redis). They are gated by the `RUN_INTEGRATION` environment variable. To run them locally:

```bash
RUN_INTEGRATION=1 poetry run pytest -q -m integration
```

If your change affects indexing, mappings, or infra orchestration (reindexing, aliases), please include a short smoke test and, if possible, run it locally against a disposable ES instance. The `scripts/maintenance/index_orchestrator.py` in the repo provides an example of a safe reindex flow.

7. How to update branch protection rules
---------------------------------------

1. Go to your repository Settings → Branches → Branch protection rules.
2. Add or edit a rule targeting the protected branch name (e.g., `main`).
3. Enable the required settings listed in section 1 above. Use the names of the checks as they appear in the GitHub Actions runs (for example, `CI`).

8. Troubleshooting common CI failures
-----------------------------------

- Tests fail locally but pass in CI: check Python version; run `poetry install` and make sure the lockfile is up-to-date.
- Lint or mypy failures: run `pre-commit run --all-files` and `poetry run mypy` locally to get faster feedback.
- Integration tests fail: ensure local services are running (Elasticsearch/Redis) and that `RUN_INTEGRATION=1` is set.

9. Thank you
------------
We appreciate contributions and review help. If you need an exception for a complex refactor, mention it in the PR and ping a maintainer to coordinate.
