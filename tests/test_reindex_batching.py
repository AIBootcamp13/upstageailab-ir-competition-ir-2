import io
import json
import tempfile
import types
import sys


# Ensure an `elasticsearch.helpers` module with a `bulk` function exists
# for the test runtime. This keeps the test deterministic whether or not
# the external `elasticsearch` package is installed in the environment.
helpers_name = "elasticsearch.helpers"
if helpers_name not in sys.modules:
    fake_helpers = types.ModuleType(helpers_name)
    sys.modules[helpers_name] = fake_helpers
else:
    fake_helpers = sys.modules[helpers_name]
import types
import sys

# Ensure a minimal `elasticsearch.helpers` module with a `bulk` function
# exists at test runtime so the code path that does `from
# elasticsearch.helpers import bulk` will succeed regardless of whether
# the real `elasticsearch` package is installed in the test environment.
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
    # Create a temporary JSONL of 7 docs
    docs = [{"id": str(i), "content": f"doc{i}"} for i in range(7)]
    tf = tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8")
    for d in docs:
        tf.write(json.dumps(d) + "\n")
    tf.flush()

    # Monkeypatch get_es to use a dummy object that records calls
    class DummyES:
        def __init__(self):
            self.indexed = []

        def index(self, index, id, document):
            self.indexed.append((index, id, document))

    dummy = DummyES()

    # If the real elasticsearch.helpers.bulk is available in the environment,
    # replace it with a thin shim that calls our DummyES.index so the test can
    # observe indexed documents without requiring a real ES server. Install
    # the shim before importing the api module so the runtime `from
    # elasticsearch.helpers import bulk` inside the function picks it up.
    # Install a bulk shim that will call our dummy instance. We attempt to
    # replace the real elasticsearch.helpers.bulk if present, otherwise we
    # attach it to the in-memory fake_helpers module created above.
    try:
        import elasticsearch.helpers as _helpers
        target_helpers = _helpers
    except Exception:
        # fall back to the fake module we inserted into sys.modules
        import sys as _sys

        target_helpers = _sys.modules[helpers_name]

    def _bulk_shim_using_dummy(client, actions, **kwargs):
        success = 0
        errors = []
        for a in actions:
            try:
                # call the dummy's index method so we record the indexed docs
                dummy.index(index=a["_index"], id=a.get("_id"), document=a.get("_source"))
                success += 1
            except Exception as exc:
                errors.append(exc)
        return success, errors

    monkeypatch.setattr(target_helpers, "bulk", _bulk_shim_using_dummy)

    import ir_core.infra as infra

    # Install get_es shim before importing api so api.get_es() will call our dummy
    monkeypatch.setattr(infra, "get_es", lambda: dummy)

    # Now import api (which does `from ..infra import get_es` at module import time)
    from ir_core import api

    # Run with batch_size=3 to exercise multiple flushes
    api.index_documents_from_jsonl(tf.name, index_name="test", batch_size=3)

    # Expect 7 indexed documents recorded by DummyES
    assert len(dummy.indexed) == 7
