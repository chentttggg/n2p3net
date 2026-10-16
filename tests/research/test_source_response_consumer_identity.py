"""Actual source-byte admission without source fitting or numerical scoring."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest


def test_new_family_consumer_rejects_source_drift_and_preserves_legacy_policy(tmp_path, monkeypatch):
    from research import prefix_replay_execution as owner

    repository = Path(__file__).resolve().parents[2]
    paths = (*owner._CONSUMER_PATHS, *owner._SOURCE_RESPONSE_CONSUMER_PATHS)
    for relative in paths:
        target = tmp_path/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((repository/relative).read_bytes())
    monkeypatch.setattr(owner,'_ROOT',tmp_path)
    source = {name:owner.artifact_receipt(tmp_path/name)['sha256'] for name in paths}
    legacy = {'consumer_source':source,'ordinary_packages':[
        {'original_request':{'expert_programs':[{'family':'joint_flat'}]}}], 'selection_packages':[]}
    legacy_identity = owner._consumer_identity(legacy)
    assert tuple(legacy_identity) == owner._CONSUMER_PATHS

    ordinary = copy.deepcopy(legacy)
    ordinary['ordinary_packages'][0]['original_request']['expert_programs'][0]['family'] = 'source_response'
    selected = {'consumer_source':source,'ordinary_packages':[], 'selection_packages':[
        {'original_request':{'candidates':[{'expert_programs':[{'family':'source_response'}]}]}}]}
    for request in (ordinary,selected):
        assert set(owner._consumer_identity(request)) == set(paths)
    for relative in ('src/research/source_response_states.py','src/transfer/dictionary_artifact.py',
            'src/transfer/temporal_noise.py','src/research/shared_stationary_factor.py',
            'src/research/stationary_noise_artifact.py','src/research/joint_flat_authority.py'):
        drifted = tmp_path/relative
        original = drifted.read_bytes()
        drifted.write_bytes(original+b'\n# actual source drift after the consumer snapshot\n')
        for request in (ordinary,selected):
            with pytest.raises(ValueError,match='externally bound executed consumer'):
                owner._consumer_identity(request)
        assert owner._consumer_identity(legacy) == legacy_identity
        drifted.write_bytes(original)

    for family in ('trial','continuous_ar'):
        unchanged = copy.deepcopy(legacy)
        unchanged['ordinary_packages'][0]['original_request']['expert_programs'][0]['family'] = family
        assert owner._consumer_identity(unchanged) == legacy_identity
