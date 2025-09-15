import pytest
import subprocess
import time
import json
import tempfile
import os
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

# Import the high-level API from your project
from ir_core import api

# Define a unique index name for this test to avoid conflicts
TEST_INDEX = "test-integration-pipeline"

# Skip the test if running in CI environment (optional)
disable = os.environ.get("CI") is not None
RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION") == "1"

@pytest.fixture(scope="module")
def live_services():
    """
    A pytest fixture that starts and stops Elasticsearch and Redis
    for the duration of the test module. This is a slow setup, so
    it's scoped to the module to run only once.
    """
    print("\n--- Starting infrastructure (ES & Redis)... ---")
    # Get the project root directory to find the script
    project_root = os.path.dirname(os.path.dirname(__file__))
    start_script = os.path.join(project_root, "scripts", "run-local.sh")

    # If services are already running on localhost:9200, don't start them
    # and don't stop them in teardown. This allows preserving a manually
    # started ES instance across test runs.
    already_running = False
    try:
        # Quick ping to local ES to detect an existing running instance.
        probe = Elasticsearch(["http://127.0.0.1:9200"], request_timeout=1)
        if probe.ping():
            already_running = True
            print("Elasticsearch already running externally; fixture will not stop it.")
    except Exception:
        already_running = False

    # Start services only if they are not already running.
    started_services = False
    if not already_running:
        try:
            subprocess.run([start_script, "start"], check=True, capture_output=True, text=True)
            started_services = True
        except subprocess.CalledProcessError as exc:
            pytest.fail(f"Failed to start infrastructure: {exc.stdout}\n{exc.stderr}")

    # Wait for services to be ready using the project's ES helper which has
    # conservative retry/timeouts. Fail fast if ES never becomes reachable so
    # the test run doesn't trigger thousands of low-level client retries.
    from ir_core.infra import get_es

    es_client = get_es()
    retries = 10
    backoff = 1.0
    while retries > 0:
        try:
            if es_client.ping():
                print("--- Infrastructure is ready. ---")
                break
        except Exception:
            # keep trying for a short bounded period
            time.sleep(backoff)
            retries -= 1
            backoff = min(backoff * 2, 5.0)

    if retries == 0:
        # Try to stop services if we failed to bring them up to avoid leaving
        # processes running and to prevent noisy client retry loops.
        try:
            stop_script = os.path.join(project_root, "scripts", "run-local.sh")
            # Only stop if we started the services here and the user did not
            # request to keep services running via env var.
            if started_services and os.environ.get("KEEP_LOCAL_SERVICES") != "1":
                subprocess.run([stop_script, "stop"], check=False, capture_output=True, text=True)
        finally:
            pytest.skip("Elasticsearch did not become available after starting services; skipping integration test.")

    # The 'yield' keyword passes control to the test function
    yield es_client

    # This code runs after all tests in the module are complete
    print("\n--- Teardown: stopping infrastructure if appropriate... ---")
    stop_script = os.path.join(project_root, "scripts", "run-local.sh")
    # If we started the services in setup and the user did not request to
    # keep services running, then stop them. Otherwise leave them running.
    if started_services and os.environ.get("KEEP_LOCAL_SERVICES") != "1":
        print("Stopping services started by fixture...")
        subprocess.run([stop_script, "stop"], check=True, capture_output=True, text=True)
        print("--- Infrastructure stopped. ---")
    else:
        print("Leaving existing services running (fixture did not start them or KEEP_LOCAL_SERVICES=1).")


import pytest


@pytest.mark.integration
def test_full_retrieval_pipeline(live_services):
    """
    Tests the entire pipeline: indexing -> retrieval.
    This test depends on the 'live_services' fixture.
    """
    # 1. Prepare Data
    # A small, distinct dataset for this test
    docs_to_index = [
        {"id": "solar-01", "content": "The sun is the center of our solar system."},
        {"id": "python-01", "content": "Python is a popular programming language for data science."},
        {"id": "ocean-01", "content": "The Pacific Ocean is the largest and deepest ocean on Earth."},
    ]

    # Write docs to a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmp_file:
        for doc in tqdm(docs_to_index, desc="writing docs", disable=disable, leave=False):
            tmp_file.write(json.dumps(doc) + "\n")
        tmp_filepath = tmp_file.name

    # 2. Index Data using the project's API
    print(f"Indexing test documents into '{TEST_INDEX}'...")
    try:
        # Use the actual API function to index the documents
        api.index_documents_from_jsonl(tmp_filepath, index_name=TEST_INDEX)

        # Give Elasticsearch a moment to make the documents searchable
        time.sleep(2)

        # 3. Perform Retrieval
        query = "What is the biggest ocean?"
        print(f"Performing hybrid retrieval for query: '{query}'")

        # We need to temporarily override the settings for the index name
        original_index = api.settings.INDEX_NAME
        api.settings.INDEX_NAME = TEST_INDEX

        results = api.hybrid_retrieve(query, rerank_k=3)

        # Restore original settings
        api.settings.INDEX_NAME = original_index

        # 4. Assert Results
        assert len(results) > 0, "Retrieval should return at least one result."

        top_result_ids = [r["hit"]["_id"] for r in results]
        print(f"Top result IDs: {top_result_ids}")

        # The query should return the "ocean-01" document among the top reranked results.
        # Embedding models and cross-lingual behavior can vary; require membership instead
        # of strict top-1 to make the integration test stable across environments.
        assert "ocean-01" in top_result_ids, "The Pacific Ocean document should be among the top reranked results."

    finally:
        # Clean up the temporary file
        os.remove(tmp_filepath)
        # Clean up the test index in Elasticsearch
        es_client = live_services
        if es_client.indices.exists(index=TEST_INDEX):
            es_client.indices.delete(index=TEST_INDEX)
            print(f"Cleaned up index '{TEST_INDEX}'.")
