import pytest

from scripts.maintenance import index_orchestrator as io


def test_rollback_on_failure_invokes_delete(monkeypatch):
    es = 'http://fake'
    alias = 'a'
    source = 'src'
    target = 'tgt'

    # Stub functions used in main flow
    monkeypatch.setattr(io, 'index_exists', lambda es_url, idx: True)
    monkeypatch.setattr(io, 'fetch_source_index_info', lambda es_url, idx: ({}, {}))
    monkeypatch.setattr(io, 'create_target_index', lambda es_url, idx, s, m: None)
    monkeypatch.setattr(io, 'start_reindex_task', lambda es_url, s, t: {})
    monkeypatch.setattr(io, 'get_task', lambda es_url, tid: {'completed': True})
    monkeypatch.setattr(io, 'wait_for_task_completion', lambda es_url, tid, timeout=None: {'completed': True})
    monkeypatch.setattr(io, 'resolve_alias', lambda es_url, a: [source])
    # prevent real network call for alias swap
    monkeypatch.setattr(io, 'swap_alias', lambda *a, **k: None)

    # counts: source != target to trigger failure
    monkeypatch.setattr(io, 'count', lambda es_url, idx: 5 if idx == source else 3)

    deleted = {'called': False, 'url': None}

    def fake_delete(url, **kwargs):
        deleted['called'] = True
        deleted['url'] = url
        class R:
            status_code = 200
            def json(self):
                return {}
        return R()

    monkeypatch.setattr(io, '_requests_delete', fake_delete)

    # prevent sys.exit from exiting test runner; capture SystemExit
    with pytest.raises(SystemExit):
        io.main(['--es', es, '--alias', alias, '--force', '--verify', '--rollback-on-failure'])

    assert deleted['called'] is True
    # The deleted URL should contain the target index name, which could be dynamically generated
    # Check that it's a valid index deletion URL (contains the base ES URL and an index name)
    assert deleted['url'].startswith(es + '/')
    assert len(deleted['url']) > len(es) + 1  # Should have something after the base URL


def test_no_rollback_when_counts_match(monkeypatch):
    es = 'http://fake'
    alias = 'a'
    source = 'src'

    monkeypatch.setattr(io, 'index_exists', lambda es_url, idx: True)
    monkeypatch.setattr(io, 'fetch_source_index_info', lambda es_url, idx: ({}, {}))
    monkeypatch.setattr(io, 'create_target_index', lambda es_url, idx, s, m: None)
    monkeypatch.setattr(io, 'start_reindex_task', lambda es_url, s, t: {})
    monkeypatch.setattr(io, 'get_task', lambda es_url, tid: {'completed': True})
    monkeypatch.setattr(io, 'wait_for_task_completion', lambda es_url, tid, timeout=None: {'completed': True})
    monkeypatch.setattr(io, 'resolve_alias', lambda es_url, a: [source])
    # prevent real network call for alias swap
    monkeypatch.setattr(io, 'swap_alias', lambda *a, **k: None)

    # counts match
    monkeypatch.setattr(io, 'count', lambda es_url, idx: 4)

    called = {'deleted': False}
    monkeypatch.setattr(io, '_requests_delete', lambda url, **kwargs: called.update({'deleted': True}))

    # Run main; since alias swap will be attempted, it will reach swap_alias which may call network functions.
    # We run with --force to skip prompts and catch SystemExit because main exits at the end.
    try:
        io.main(['--es', es, '--alias', alias, '--force', '--verify'])
    except SystemExit:
        pass

    assert called['deleted'] is False
