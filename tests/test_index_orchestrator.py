import json
import pytest

from scripts.maintenance import index_orchestrator as io


def test_build_alias_actions_removes_old_when_keep_old_false():
    actions = io.build_alias_actions('papers', 'papers-v2', remove_indices=['papers-v1'], keep_old=False)
    assert isinstance(actions, dict)
    assert 'actions' in actions
    acts = actions['actions']
    # Expect remove then add
    assert {'remove': {'index': 'papers-v1', 'alias': 'papers'}} in acts
    assert {'add': {'index': 'papers-v2', 'alias': 'papers'}} in acts


def test_build_alias_actions_keeps_old_when_flag_true():
    actions = io.build_alias_actions('papers', 'papers-v2', remove_indices=['papers-v1'], keep_old=True)
    acts = actions['actions']
    # Should not contain remove action
    assert all('remove' not in a for a in acts)
    assert {'add': {'index': 'papers-v2', 'alias': 'papers'}} in acts


def test_build_alias_actions_no_old_indices():
    actions = io.build_alias_actions('papers', 'papers-v2', remove_indices=None, keep_old=False)
    acts = actions['actions']
    assert acts == [{'add': {'index': 'papers-v2', 'alias': 'papers'}}]
