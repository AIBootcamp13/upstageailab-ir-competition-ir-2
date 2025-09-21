import io
import json
import tempfile
import types
import sys

# Ensure a minimal `elasticsearch.helpers` module with a `bulk` function
# exists at test runtime. This is a standard way to handle optional dependencies in tests.
helpers_name = "elasticsearch.helpers"
if helpers_name not in sys.modules:
    fake_helpers = types.ModuleType(helpers_name)
    sys.modules[helpers_name] = fake_helpers
else:
    fake_helpers = sys.modules[helpers_name]

def _bulk_shim(client, actions, **kwargs):
    success = 0
    errors = []
    for a in actions:
        try:
            client.index(index=a["_index"], id=a.get("_id"), document=a.get("_source"))
            success += 1
        except Exception as exc:
            errors.append(exc)
    return success, errors

setattr(fake_helpers, "bulk", _bulk_shim)


def test_batching_counts(monkeypatch):
    # 1. SETUP MOCKS AND DATA
    # Create a temporary JSONL of 7 docs
    docs = [{"id": str(i), "content": f"doc{i}"} for i in range(7)]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tf:
        for d in docs:
            tf.write(json.dumps(d) + "\n")
        tf_name = tf.name

    # Define the dummy object that will replace the real ES client
    class DummyES:
        def __init__(self):
            self.indexed = []
        def index(self, index, id, document):
            self.indexed.append((index, id, document))
        def options(self, **kwargs):
            return self
    dummy = DummyES()

    # 2. PATCH DEPENDENCIES *BEFORE* IMPORTING THE CODE UNDER TEST
    # Import the module that needs to be patched
    import ir_core.infra as infra

    # Apply the patch to the source of the function
    monkeypatch.setattr(infra, "get_es", lambda: dummy)

    # Mock the bulk helper as before
    try:
        import elasticsearch.helpers as _helpers
        target_helpers = _helpers
    except ImportError:
        target_helpers = sys.modules[helpers_name]

    def _bulk_shim_using_dummy(client, actions, **kwargs):
        success = 0
        errors = []
        for a in actions:
            try:
                client.index(index=a["_index"], id=a.get("_id"), document=a.get("_source"))
                success += 1
            except Exception as exc:
                errors.append(exc)
        return success, errors

    monkeypatch.setattr(target_helpers, "bulk", _bulk_shim_using_dummy)

    # 3. NOW, it is safe to import the 'api' module. It will receive the patched dependencies.
    from ir_core import api

    # 4. EXECUTE THE FUNCTION TO BE TESTED
    api.index_documents_from_jsonl(tf_name, index_name="test", batch_size=3)

    # 5. ASSERT THE RESULTS
    # Expect exactly 7 documents to be recorded by our DummyES instance
    assert len(dummy.indexed) == 7

