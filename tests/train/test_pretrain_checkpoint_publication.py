"""Actual CLI rejection and shared writer fault controls without training."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments import run_pretrain as ssl
from experiments import run_pretrain_supervised as supervised
from research import source_execution
from research.source_execution import save_source_checkpoint
from transfer.checkpoint import CHECKPOINT_SCHEMA, checkpoint_state_digest, load_checkpoint_payload


@pytest.fixture(autouse=True)
def no_inherited_admission(monkeypatch):
    monkeypatch.setattr(source_execution, '_PROCESS_INPUT_ADMISSION', None)


def _forbid(*args, **kwargs):
    pytest.fail('existing checkpoint reached source capture or data/fit work')


@pytest.mark.parametrize('entry', ['ssl', 'supervised'])
def test_existing_checkpoint_refused_by_actual_cli_before_source_capture_or_input(tmp_path, monkeypatch, entry):
    checkpoint = tmp_path/'existing.pt'
    checkpoint.write_bytes(b'immutable original checkpoint')
    module = ssl if entry == 'ssl' else supervised
    monkeypatch.setattr(module, 'capture_executed_run_source', _forbid)
    if entry == 'supervised':
        monkeypatch.setattr(module, '_source_snapshot_sha256_once', _forbid)
    args = ['--source-cache', str(tmp_path/'missing.npz'), '--source-snapshot-manifest', str(tmp_path/'missing.json'),
        '--checkpoint', str(checkpoint)]
    with pytest.raises(FileExistsError):
        if entry == 'ssl':
            monkeypatch.setattr('sys.argv', ['run_pretrain.py', *args])
            module.main()
        else:
            module.main(args)
    assert checkpoint.read_bytes() == b'immutable original checkpoint'
    assert not checkpoint.with_suffix('.pt.source').exists()


def test_snapshot_collision_is_refused_before_source_capture_and_training(tmp_path, monkeypatch):
    checkpoint = tmp_path/'new.pt'
    candidate = tmp_path/'new.u2.pt'
    candidate.write_bytes(b'immutable original update candidate')
    monkeypatch.setattr(supervised, '_source_snapshot_sha256_once', _forbid)
    monkeypatch.setattr(supervised, 'capture_executed_run_source', _forbid)
    with pytest.raises(FileExistsError):
        supervised.main(['--source-cache', str(tmp_path/'missing.npz'), '--source-snapshot-manifest', str(tmp_path/'missing.json'),
            '--checkpoint', str(checkpoint), '--fixed-updates', '3', '--snapshot-updates', '1', '2'])
    assert not checkpoint.exists() and not checkpoint.with_suffix('.pt.source').exists()
    assert candidate.read_bytes() == b'immutable original update candidate'


def test_batch_checks_every_output_before_any_job_source_capture(tmp_path, monkeypatch):
    first, second = tmp_path/'first.pt', tmp_path/'second.pt'
    second.write_bytes(b'old second job checkpoint')
    monkeypatch.setattr(supervised, '_run_batch', _forbid)
    monkeypatch.setattr(supervised, 'capture_executed_run_source', _forbid)
    with pytest.raises(FileExistsError):
        supervised.main(['--source-cache', str(tmp_path/'missing.npz'), '--source-snapshot-manifest', str(tmp_path/'missing.json'),
            '--job', 'H1', str(first), '--job', 'H2', str(second), '--batch-record', str(tmp_path/'batch.json')])
    assert second.read_bytes() == b'old second job checkpoint'
    assert not first.exists() and not first.with_suffix('.pt.source').exists()


@pytest.mark.parametrize('collision', ['batch_record', 'other_job_snapshot'])
def test_batch_rejects_output_aliases_before_any_capture(tmp_path, monkeypatch, collision):
    checkpoint = tmp_path/'first.pt'
    monkeypatch.setattr(supervised, '_run_batch', _forbid)
    second = tmp_path/'first.u1.pt' if collision == 'other_job_snapshot' else tmp_path/'second.pt'
    sentinel = checkpoint if collision == 'batch_record' else tmp_path/'batch.json'
    with pytest.raises(ValueError, match='paths must be unique'):
        supervised.main(['--source-cache', str(tmp_path/'missing.npz'), '--source-snapshot-manifest', str(tmp_path/'missing.json'),
            '--job', 'H1', str(checkpoint), '--job', 'H2', str(second), '--batch-record', str(sentinel),
            '--fixed-updates', '2', '--snapshot-updates', '1'])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('collision', ['existing_checkpoint', 'new_job_alias'])
def test_batch_dry_run_cannot_publish_its_record_over_a_checkpoint(tmp_path, monkeypatch, collision):
    checkpoint = tmp_path/'model.pt'
    if collision == 'existing_checkpoint':
        checkpoint.write_bytes(b'original numeric checkpoint')
        failure = FileExistsError
        sentinel = checkpoint
        job = tmp_path/'planned.pt'
    else:
        failure = ValueError
        sentinel = checkpoint
        job = checkpoint
    monkeypatch.setattr(supervised, '_source_snapshot_sha256_once', _forbid)
    with pytest.raises(failure):
        supervised.main(['--source-cache', str(tmp_path/'missing.npz'), '--source-snapshot-manifest', str(tmp_path/'missing.json'),
            '--job', 'H1', str(job), '--batch-record', str(sentinel), '--dry-run'])
    if collision == 'existing_checkpoint':
        assert checkpoint.read_bytes() == b'original numeric checkpoint'
    else:
        assert not checkpoint.exists()


def test_completed_batch_record_is_an_exclusive_publication(tmp_path):
    record = tmp_path/'batch.json'
    supervised._write_json_atomic(record, {'completed': True}, exclusive=True)
    original = record.read_bytes()
    with pytest.raises(FileExistsError):
        supervised._write_json_atomic(record, {'completed': False}, exclusive=True)
    assert record.read_bytes() == original


def payload():
    state = {'weight': torch.arange(8, dtype=torch.float32).reshape(2, 4)}
    return {'schema': CHECKPOINT_SCHEMA, 'trunk_state_dict': state,
        'trunk_state_dict_sha256': checkpoint_state_digest(state),
        'execution_source': {'source_role': 'actual_new_run_executing_runtime', 'entrypoint': 'software_writer_control'},
        'resource_execution': {'resource_scope': 'software_correctness', 'observation_only_not_cloud_admission': True}}


def test_exclusive_writer_preserves_existing_bytes_before_serializing(tmp_path, monkeypatch):
    checkpoint = tmp_path/'existing.pt'
    checkpoint.write_bytes(b'original producer')
    monkeypatch.setattr(torch, 'save', _forbid)
    with pytest.raises(FileExistsError):
        save_source_checkpoint(checkpoint, payload(), exclusive=True)
    assert checkpoint.read_bytes() == b'original producer'


def test_exclusive_write_failure_never_promotes_partial_to_final_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path/'new.pt'
    def partial(payload, stream):
        stream.write(b'partial checkpoint bytes')
        raise OSError('injected Torch writer failure')
    monkeypatch.setattr(torch, 'save', partial)
    with pytest.raises(OSError, match='Torch writer'):
        save_source_checkpoint(checkpoint, payload(), exclusive=True)
    assert not checkpoint.exists() and not list(tmp_path.iterdir())


def test_competing_creator_during_serialization_keeps_foreign_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path/'race.pt'
    original = torch.save
    def competitor(value, stream):
        original(value, stream)
        checkpoint.write_bytes(b'competing immutable checkpoint')
    monkeypatch.setattr(torch, 'save', competitor)
    with pytest.raises(FileExistsError):
        save_source_checkpoint(checkpoint, payload(), exclusive=True)
    assert checkpoint.read_bytes() == b'competing immutable checkpoint'
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_exclusive_schema3_round_trip_keeps_digest_g8_and_resource_records(tmp_path):
    expected = payload()
    rng_before = torch.get_rng_state().clone()
    checkpoint = tmp_path/'new.pt'
    save_source_checkpoint(checkpoint, expected, exclusive=True)
    actual = load_checkpoint_payload(checkpoint)
    assert actual['schema'] == CHECKPOINT_SCHEMA
    assert actual['trunk_state_dict_sha256'] == expected['trunk_state_dict_sha256']
    assert torch.equal(actual['trunk_state_dict']['weight'], expected['trunk_state_dict']['weight'])
    assert actual['execution_source'] == expected['execution_source']
    assert actual['resource_execution'] == expected['resource_execution']
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize('fail', [False, True], ids=['completed_charge', 'failed_temporary_cleanup'])
def test_exclusive_writer_uses_original_artifact_budget_and_settles_actual_bytes(tmp_path, monkeypatch, fail):
    manager = SimpleNamespace(SourceInputState=source_execution._SourceInputState)
    admission = source_execution.SourceInputAdmission(manager, capacity_bytes=0, load_workers=1)
    monkeypatch.setattr(source_execution, 'filesystem_uses_host_memory', lambda path: False)
    monkeypatch.setattr(source_execution, '_PROCESS_INPUT_ADMISSION', admission)
    checkpoint = tmp_path/'new.pt'
    original = torch.save
    observed = []
    def save(value, stream):
        if hasattr(stream, 'fileno'):
            leases = admission.record()['active_leases']
            assert len(leases) == 1 and len(leases[0]['paths']) == 2
            observed.append(leases[0])
            if fail:
                stream.write(b'partial')
                raise OSError('injected admitted checkpoint failure')
        return original(value, stream)
    monkeypatch.setattr(torch, 'save', save)
    if fail:
        with pytest.raises(OSError, match='admitted checkpoint'):
            save_source_checkpoint(checkpoint, payload(), exclusive=True)
    else:
        save_source_checkpoint(checkpoint, payload(), exclusive=True)
    assert len(observed) == 1
    expected_bytes = 0 if fail else checkpoint.stat().st_size
    record = admission.record()
    assert sum(row['reserved_bytes'] for row in record['filesystem_pools'].values()) == expected_bytes
    assert sum(row['disk_bytes'] for row in record['active_leases']) == expected_bytes
    assert record['free_staging_bytes'] == 0
    assert not any(path.name.startswith('.') for path in tmp_path.iterdir())
