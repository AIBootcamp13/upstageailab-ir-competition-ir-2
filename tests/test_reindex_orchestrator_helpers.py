import json
import os
import tempfile
import json
import os
import tempfile

from scripts.maintenance import index_orchestrator as io
from scripts.maintenance import recompute


def test_build_alias_actions_add_and_remove():
    alias = 'documents'
    add_index = 'documents_v2'
    remove_indices = ['documents_v1', 'documents_old']
    body = io.build_alias_actions(alias, add_index, remove_indices=remove_indices, keep_old=False)
    assert 'actions' in body
    # Expect remove actions first, then add
    actions = body['actions']
    assert any(a.get('add', {}).get('index') == add_index for a in actions)
    for r in remove_indices:
        assert any(a.get('remove', {}).get('index') == r for a in actions)


def test_probe_embedding_dim_returns_int_and_defaults():
    # When repo embedding function exists, probe should return an int dimension
    dim = recompute.probe_embedding_dim(None)
    assert isinstance(dim, int)
    assert dim > 0
