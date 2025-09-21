import os
import pytest


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless RUN_INTEGRATION=1 or marker selected.

    This keeps the default test run fast and side-effect free while allowing
    developers to opt into running integration tests explicitly.
    """
    run_integration = os.environ.get("RUN_INTEGRATION") == "1"
    if run_integration or config.getoption("markers"):
        return

    skip_integration = pytest.mark.skip(reason="Integration tests disabled; set RUN_INTEGRATION=1 or run with -m integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
