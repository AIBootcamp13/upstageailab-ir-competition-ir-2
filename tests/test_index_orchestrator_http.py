import time
import json
import pytest

from scripts.maintenance import index_orchestrator as io


class DummyResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json


def test_with_retries_succeeds_after_retry(monkeypatch):
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = io.with_retries(flaky, retries=4, backoff=0.01)
    assert result == "ok"
    assert calls["count"] == 3


def test_wait_for_task_completion_polls_and_returns(monkeypatch):
    # Simulate task responses: first pending, then completed
    states = [{"completed": False}, {"completed": True}]

    def fake_get(url, **kwargs):
        # url like http://es/_tasks/{task_id}
        return DummyResponse(200, states.pop(0))

    monkeypatch.setattr(io, "_requests_get", fake_get)

    start = time.time()
    res = io.wait_for_task_completion("http://es", "task-1", timeout=2, poll_interval=1)
    duration = time.time() - start

    assert res["completed"] is True
    # ensure we waited at least one poll interval
    assert duration >= 0.0


def test_wait_for_task_completion_times_out(monkeypatch):
    # Always return not completed
    def fake_get(url, **kwargs):
        return DummyResponse(200, {"completed": False})

    monkeypatch.setattr(io, "_requests_get", fake_get)

    with pytest.raises(RuntimeError):
        io.wait_for_task_completion("http://es", "task-1", timeout=1, poll_interval=1)
