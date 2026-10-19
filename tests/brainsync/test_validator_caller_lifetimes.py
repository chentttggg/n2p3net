"""Caller lifetime controls keep the actual validated objects reachable."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.brainsync import BRAIN_SYNC_PREPROCESSING
from data.brainsync_contract import validate_brainsync_bids_session
from experiments import brainsync_model_task as task
from experiments import correct_brainsync_reference as reference
from experiments import fit_brainsync_continuous_qc as qc
from experiments import fit_brainsync_eog as eog
from experiments import prepare_brainsync_eog_matrix as matrix
from experiments import run_bidirectional_qc_study as bidirectional
from experiments import run_filter_phase_study as phase
from tests.support.brainsync import write_bids_session


class CallerProbeError(RuntimeError):
    pass


def session(root, name):
    return write_bids_session(root/name, session_id=name, target=2, started_utc='2026-10-19T00:00:00Z')


def retained_validator(held, *, fail_on=None):
    def validate(root):
        if fail_on is not None and len(held) == fail_on:
            raise CallerProbeError('synthetic later validator failed')
        value = validate_brainsync_bids_session(root)
        held.append(value)
        return value
    return validate


def require_closed(held):
    assert held
    for value in held:
        with pytest.raises(ValueError, match='closed'):
            value.bids.assert_content_unchanged()
        # The object and traceback stay reachable; this write must succeed
        # without relying on finalizers or collection of the retained object.
        value.session_path.write_bytes(value.session_path.read_bytes())


def test_model_task_closes_first_metadata_admission_when_later_admission_fails(tmp_path, monkeypatch):
    roots = [session(tmp_path, name) for name in ('first', 'second')]
    held = []
    monkeypatch.setattr(task, 'validate_brainsync_bids_session', retained_validator(held, fail_on=1))
    with pytest.raises(SystemExit) as failure:
        task.main(['--task', 'calibrate', '--dry-run', '--session-dir', str(roots[0]), '--session-dir', str(roots[1]),
            '--model-artifact', str(tmp_path/'model.pt'), '--output-dir', str(tmp_path/'result')])
    assert isinstance(failure.value.__cause__, CallerProbeError)
    require_closed(held)


@pytest.mark.parametrize('module', [eog, qc], ids=['eog', 'continuous_qc'])
def test_raw_calibration_caller_holds_then_closes_lease_on_downstream_error(tmp_path, monkeypatch, module):
    root = session(tmp_path, 'input')
    held = []
    monkeypatch.setattr(module, 'validate_brainsync_bids_session', retained_validator(held))
    def fail_during_raw_consumption(*args, **kwargs):
        assert held
        held[0].bids.assert_content_unchanged()
        if os.name == 'nt':
            with pytest.raises(PermissionError):
                held[0].session_path.write_bytes(held[0].session_path.read_bytes())
        raise CallerProbeError('synthetic raw consumer failure')
    monkeypatch.setattr(module, 'load_brainsync_session', fail_during_raw_consumption)
    with pytest.raises(CallerProbeError, match='raw consumer'):
        if module is eog:
            module.fit_brainsync_eog_sessions([root])
        else:
            spec = replace(BRAIN_SYNC_PREPROCESSING, filter_phase='forward', zero_phase_edge_policy='legacy',
                causal_iir_initial_state='steady_state_first_sample')
            manifest = {'schema': 'brainsync_continuous_qc_calibration/1', 'preprocessing': spec.record(),
                'spec': {'difference_z': 3., 'difference_floor_v': 1e-6, 'settle_samples': 1},
                'sessions': [{'session_dir': str(root), 'clean_intervals_samples': [[1, 9]],
                    'clean_interval_evidence': 'synthetic explicit interval'}]}
            path = tmp_path/'quality_manifest.json'
            path.write_bytes(json.dumps(manifest).encode())
            module.fit_brainsync_continuous_manifest(path)
    require_closed(held)


@pytest.mark.parametrize('module', [matrix, bidirectional, phase], ids=['eog_matrix', 'bidirectional_study', 'filter_phase_study'])
def test_study_caller_closes_prior_inputs_when_later_validator_fails(tmp_path, monkeypatch, module):
    roots = [session(tmp_path, name) for name in ('first', 'second')]
    held = []
    monkeypatch.setattr(module, 'validate_brainsync_bids_session', retained_validator(held, fail_on=1))
    with pytest.raises(CallerProbeError, match='later validator'):
        if module is matrix:
            module.prepare(tmp_path, tmp_path/'output', ('first',), ('second',))
        elif module is phase:
            module.main(['--sessions', *map(str, roots), '--output', str(tmp_path/'output')])
        else:
            path = tmp_path/'study.json'
            path.write_bytes(json.dumps({'calibration_sessions': [str(roots[0])],
                'evaluation_sessions': [str(roots[1])]}).encode())
            module.main(['--plan', str(path), '--output', str(tmp_path/'output')])
    require_closed(held)


@pytest.mark.parametrize('succeed', [False, True], ids=['amend_failure', 'amend_success'])
def test_quality_metadata_amendment_closes_all_leases_on_both_outcomes(tmp_path, monkeypatch, succeed):
    from data import continuous_qc

    root = session(tmp_path, 'input')
    held = []
    actual_validator = retained_validator(held)
    import data.brainsync_contract as contract
    monkeypatch.setattr(contract, 'validate_brainsync_bids_session', actual_validator)
    source = tmp_path/'source_quality.json'
    source.write_bytes(b'synthetic frozen-quality numeric owner intercepted')
    raw = next(root.rglob('*.edf'))
    fit = {'source_reference': 'A2', 'sources': [{'session_id': 'input',
        'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}]}
    original = SimpleNamespace(fit_provenance=fit, epoch_recovery=None,
        record=lambda: {'fit_provenance': dict(fit), 'numerical': {'fixed': 1}})
    monkeypatch.setattr(continuous_qc, 'load_continuous_qc', lambda path: original)
    monkeypatch.setattr(continuous_qc, 'continuous_qc_from_record', lambda value: SimpleNamespace(record=lambda: value))
    def save(path, value):
        assert held
        held[0].bids.assert_content_unchanged()
        return path
    monkeypatch.setattr(continuous_qc, 'save_continuous_qc', save)
    # Rebind only the caller's mutable metadata view. Raw input and its actual
    # lease stay unchanged; no labels or numerical quality parameters are fit.
    def validated(path):
        value = actual_validator(path)
        correction = {'previous_label': 'A2', 'authority': 'synthetic'}
        return SimpleNamespace(session_id=value.session_id,
            bids=SimpleNamespace(close=value.bids.close, raw_path=value.bids.raw_path,
                source_reference='left earlobe' if succeed else 'conflict'),
            manifest={'reference_correction': correction})
    monkeypatch.setattr(contract, 'validate_brainsync_bids_session', validated)
    if succeed:
        reference.amend_continuous_quality_reference(source, [root], previous='A2', authority='synthetic', output=tmp_path/'new_quality.json')
    else:
        with pytest.raises(ValueError, match='corrected REF'):
            reference.amend_continuous_quality_reference(source, [root], previous='A2', authority='synthetic', output=tmp_path/'new_quality.json')
    require_closed(held)


@pytest.mark.parametrize('incomplete', [False, True], ids=['complete_contract', 'explicit_incomplete'])
def test_reference_afterbytes_interference_rejects_completion_and_preserves_foreign_metadata(tmp_path, monkeypatch, incomplete):
    root = session(tmp_path, 'input')
    path = root/'session.json'
    payload = json.loads(path.read_bytes())
    payload['montage'] = {'ref_label': 'A2'}
    path.write_bytes(json.dumps(payload).encode())
    if incomplete:
        (root/payload['recording']['bids']['channels_tsv']).unlink()
    actual_replace = reference._replace_metadata
    interfered = []
    def replace_then_foreign(path, content):
        actual_replace(path, content)
        if path.name == 'session.json' and not interfered:
            record = json.loads(path.read_bytes())
            record['reference_correction']['authority'] = 'concurrent-distinct-authority'
            actual_replace(path, (json.dumps(record, sort_keys=True)+'\n').encode())
            interfered.append(True)
    monkeypatch.setattr(reference, '_replace_metadata', replace_then_foreign)
    with pytest.raises(ValueError, match='after bytes') as failure:
        reference.correct_sessions([root], previous='A2', authority='requested-authority',
            output=tmp_path/'correction', allow_incomplete=incomplete)
    assert not (tmp_path/'correction/manifest.json').exists()
    assert json.loads(path.read_bytes())['reference_correction']['authority'] == 'concurrent-distinct-authority'
    assert any('manual reconciliation' in note for note in failure.value.__notes__)


@pytest.mark.parametrize('incomplete', [False, True], ids=['complete_contract', 'explicit_incomplete'])
def test_reference_completion_is_published_while_metadata_read_lease_is_live(tmp_path, monkeypatch, incomplete):
    root = session(tmp_path, 'input')
    path = root/'session.json'
    payload = json.loads(path.read_bytes())
    payload['montage'] = {'ref_label': 'A2'}
    payload.pop('target_label')
    payload['experiment'].pop('thought_digit')
    payload['experiment'].pop('target_label_status')
    path.write_bytes(json.dumps(payload).encode())
    if incomplete:
        (root/payload['recording']['bids']['channels_tsv']).unlink()
    actual_publish = reference.atomic_write_json
    blocked = []
    def publish(target, record):
        if Path(target).name == 'manifest.json':
            if os.name == 'nt':
                with pytest.raises(PermissionError):
                    path.write_bytes(path.read_bytes())
            blocked.append(True)
        return actual_publish(target, record)
    monkeypatch.setattr(reference, 'atomic_write_json', publish)
    record = reference.correct_sessions([root], previous='A2', authority='synthetic',
        output=tmp_path/'correction', allow_incomplete=incomplete)
    assert blocked == [True]
    assert all(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['after_sha256'] for row in record['files'])
    path.write_bytes(path.read_bytes())
    assert 'target_label' not in json.loads(path.read_bytes())
