"""Actual finalizer writer ownership, without fitting or source/model decoding."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_erp_comparator as runner
from research.cloud_execution import CloudResult
from research.comparison import INPUT_IDENTITY_SCHEMA
from research.contracts import semantic_sha256
from research.source_execution import (
    SourceInputAdmission,
    current_input_admission,
    install_input_admission,
)


def _fixture(tmp_path, monkeypatch):
    output = tmp_path / 'run'
    participant = output / 'participants/participant_000'
    participant.mkdir(parents=True)
    prediction = participant / '00000.npz'
    np.savez_compressed(prediction, y=np.array([0, 1]), test_rows=np.array([0, 1]),
                        calibration_rows=np.array([], dtype=np.int64), posterior_logits=np.array([-1., 1.]))
    model = participant / '00000.pt'
    model.write_bytes(b'opaque_already_admitted_software_numeric_state')
    sha = 'a' * 64
    evidence = {field: sha for field in ('evidence_sha256', 'values_sha256', 'labels_sha256',
                                        'observed_sha256', 'rows_sha256')}
    identity = {'schema': INPUT_IDENTITY_SCHEMA, 'fit': {**evidence, 'n_rows': 0},
        'score': {**evidence, 'n_rows': 2}, 'raw_bytes_bound': True,
        'operator': {'preprocessing': {}, 'spatial': {}, 'raw_support': [], 'execution': []},
        'decision_semantics': {'event_contract': None, 'candidate_task_contract': None, 'candidate_vocabulary': [],
            'session_schema': None, 'timeline_complete': True, 'requested_schedule_sha256': sha},
        'fit_context': {'timeline_sha256': sha, 'metadata_sha256': sha, 'qc_features_sha256': None}}
    identity['digest'] = semantic_sha256(identity)
    cell = {'subject': 'A01', 'participant_key': None, 'study': 'software', 'arm': 'nn',
        'family': 'full_fine', 'K': 0, 'actual_K': 0, 'requested_Ks': [0], 'source_seed': 11, 'target_seed': 22,
        'status': 'completed', 'auc': 1., 'physical_target_fit': False, 'comparison_identity': identity,
        'probability_status': 'completed', 'probability_semantics': {
            'opaque_original_source_witness': {'rows': ['source_witness_' + 'x' * 240] * 12000}},
        'model_artifact': {'path': model.name, 'sha256': hashlib.sha256(model.read_bytes()).hexdigest()},
        'prediction_artifact': {'path': prediction.name, 'sha256': hashlib.sha256(prediction.read_bytes()).hexdigest()}}
    request = {key: cell[key] for key in ('subject', 'arm', 'actual_K', 'participant_key', 'study', 'source_seed', 'target_seed')}
    request.update(requested_K=0, cell_index=0)
    plan = {'evidence_scope': 'software_correctness', 'budgets': [0],
        'arms': [{'name': 'nn', 'family': 'full_fine'}], 'source_seeds': [11], 'target_seeds': [22],
        'contrasts': {}, 'allowed_input_axes': [],
        'probability_pools': [{'name': 'source_fixed', 'expert_ids': ['nn'], 'weights': [1.]}]}
    programme = object.__new__(runner.BinaryCloudProgramme)
    programme.output, programme.plan, programme.plan_path, programme.tree = output, plan, output / 'plan.json', {}
    programme.tasks = [SimpleNamespace(task_id='already_admitted:A01', output=str(participant), arguments={'subject': 'A01'})]
    programme.admitted = {'already_admitted:A01': {'cells': [cell], 'requested_participants': [request],
        'seed_pairs': [[11, 22]], 'execution_snapshot': {'scope': 'software_fixture'}, 'participant_authority': {}}}
    programme.admission_failures, programme.requests = {}, [request]
    programme.admitted_files = {str(prediction): cell['prediction_artifact']['sha256']}
    programme.input_admission = None
    monkeypatch.setattr(runner, '_executing_tree', lambda: {})
    monkeypatch.setattr(runner, '_prefix_bindings', lambda plan, path: {})
    return programme, {'already_admitted:A01': CloudResult('already_admitted:A01', 'completed')}


def test_actual_finalizer_large_metadata_small_prediction_is_admitted_before_write(tmp_path, monkeypatch):
    programme, outcomes = _fixture(tmp_path, monkeypatch)
    with multiprocessing.get_context('spawn').Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=0, load_workers=1)
        previous = current_input_admission()
        assert previous is None
        report = programme.finalize(outcomes, input_admission=owner)
        assert current_input_admission() is previous
        record = owner.record()
        actual = sum(path.stat().st_size for path in programme.output.rglob('*') if path.is_file())
        assert report['requested_cells'] == 1 and report['physical_target_fits'] == 0
        assert report['probability_pools'][0]['status'] == 'completed'
        lease = next(row for row in record['active_leases'] if row['path'] == str(programme.output))
        baseline = lease['baseline_bytes']
        assert lease['committed'] and lease['disk_bytes'] == actual - baseline
        assert not record['pending_nested_artifacts'] and record['load_active'] == 0
        pool = next(iter(record['filesystem_pools'].values()))
        assert pool['peak_reserved_bytes'] >= lease['disk_bytes'] > 2**20
        assert not list(programme.output.glob('.*.tmp'))
        assert json.loads((programme.output / 'result.json').read_bytes()) == report


def test_actual_finalizer_pool_is_reserved_before_numpy_opens_the_path(tmp_path, monkeypatch):
    programme, outcomes = _fixture(tmp_path, monkeypatch)
    original = np.savez_compressed
    observed = []
    with multiprocessing.get_context('spawn').Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=0, load_workers=1)
        def checked(path, **arrays):
            assert current_input_admission() is owner
            pending = owner.record()['pending_nested_artifacts']
            assert any(str(path) in row['paths'] for row in pending)
            observed.append(tuple(arrays))
            return original(path, **arrays)
        monkeypatch.setattr(np, 'savez_compressed', checked)
        programme.finalize(outcomes, input_admission=owner)
    assert observed == [('y', 'test_rows', 'calibration_rows', 'posterior_logits', 'probability')]


def test_finalizer_restores_existing_owner_and_retains_surviving_bytes_on_json_failure(tmp_path, monkeypatch):
    import research.source_execution as execution
    programme, outcomes = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(execution, 'filesystem_uses_host_memory', lambda path: True)
    with multiprocessing.get_context('spawn').Manager() as manager:
        previous = SourceInputAdmission(manager, capacity_bytes=16 * 2**20, load_workers=1)
        owner = SourceInputAdmission(manager, capacity_bytes=2 * 2**20, load_workers=1)
        install_input_admission(previous)
        try:
            with pytest.raises(ValueError, match='remaining shared filesystem/RAM budget|exceeds shared RAM capacity'):
                programme.finalize(outcomes, input_admission=owner)
            assert current_input_admission() is previous
            assert not (programme.output / 'result.json').exists()
            pool_path = programme.output / 'probability_pool_00000.npz'
            assert pool_path.exists()
            record = owner.record()
            lease = next(row for row in record['active_leases'] if row['path'] == str(programme.output))
            assert lease['committed'] and lease['disk_bytes'] == pool_path.stat().st_size
            assert record['free_staging_bytes'] == owner.capacity - pool_path.stat().st_size
            assert not record['pending_nested_artifacts']
        finally:
            install_input_admission(None)
