from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.brainsync import BRAIN_SYNC_PREPROCESSING
from experiments import brainsync_model_task as task
from experiments import correct_brainsync_reference as reference
from experiments import fit_brainsync_continuous_qc as qc
from experiments import prepare_brainsync_cache as prepare
from experiments import report_brainsync_quality as report
from experiments import run_brainsync_cross_decision as cross
from experiments import run_brainsync_pilot as pilot
from experiments.brainsync_cli_artifacts import (
    new_output_reservation,
    publish_epoch_cache_new,
    publish_json_new,
)
from tests import PROJECT_ROOT
from tests.support.brainsync import write_bids_session

CONFIG_ROOT = Path(os.environ.get('N2P3_WORKSTATION_TEST_CONFIG_DIR', str(PROJECT_ROOT/'configs')))


def test_fixed_erp_recipe_reaches_preflight_without_fit(tmp_path, monkeypatch):
    recipe = tmp_path/'recipe.json'
    recipe.write_text(json.dumps({'name': 'fixed', 'initialization': 'pooled_lda', 'weights': 'none',
        'bn_affine': 'frozen', 'bn_buffers': 'source', 'dropout': 'off'}))
    monkeypatch.setattr(task, 'validate_brainsync_bids_session', lambda path: SimpleNamespace(
        started_timestamp_s=0, bids=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(task, 'load_brainsync_sessions', lambda *args, **kwargs: object())
    monkeypatch.setattr(task, 'apply_brainsync_reference', lambda dataset, *args: dataset)
    seen = {}
    def preflight(dataset, **kwargs):
        seen.update(kwargs)
        return {'status': 'dry_run', 'planned_ocular': {}}
    monkeypatch.setattr(task, 'preflight_calibrated_predictor', preflight)
    monkeypatch.setattr(task, 'fit_calibrated_predictor', lambda *args, **kwargs: pytest.fail('dry run fitted'))
    task.main(['--task', 'calibrate', '--selector-mode', 'fixed_erp', '--calibration-recipe', str(recipe),
        '--dry-run', '--session-dir', str(tmp_path/'session'), '--model-artifact', str(tmp_path/'model.pt'),
        '--output-dir', str(tmp_path/'output')])
    assert seen['transfer_policy'].initialization == 'pooled_lda'


def test_model_artifact_cannot_collide_with_summary(tmp_path):
    output = tmp_path/'output'
    with pytest.raises(SystemExit) as error:
        task.main(['--task', 'calibrate', '--session-dir', str(tmp_path/'session'),
            '--model-artifact', str(output/'summary.json'), '--output-dir', str(output)])
    assert error.value.code == 2
    assert not (output/'summary.json').exists()


def test_continuous_qc_requires_explicit_complete_preprocessing_before_data(tmp_path):
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'schema': 'brainsync_continuous_qc_calibration/1', 'sessions': [], 'spec': {}}))
    with pytest.raises(ValueError, match='complete preprocessing'):
        qc.fit_brainsync_continuous_manifest(path)
    path.write_text(json.dumps({'schema': 'brainsync_continuous_qc_calibration/1', 'sessions': [], 'spec': {},
        'preprocessing': {'name': 'incomplete'}}))
    with pytest.raises(ValueError, match='complete'):
        qc.fit_brainsync_continuous_manifest(path)


def _reference_session(tmp_path):
    root = write_bids_session(tmp_path/'session', session_id='reference-audit', target=2,
        started_utc='2026-10-19T00:00:00Z')
    path = root/'session.json'
    payload = json.loads(path.read_text())
    payload['montage'] = {'ref_label': 'A2'}
    path.write_text(json.dumps(payload))
    return root


def test_reference_existing_output_cannot_be_overwritten(tmp_path):
    root = _reference_session(tmp_path)
    output = tmp_path/'correction'
    output.mkdir()
    sentinel = output/'manifest.json'
    sentinel.write_text('original')
    before = (root/'session.json').read_bytes()
    with pytest.raises(FileExistsError):
        reference.correct_sessions([root], previous='A2', authority='synthetic-test', output=output)
    assert sentinel.read_text() == 'original'
    assert (root/'session.json').read_bytes() == before


def test_reference_ctrl_c_rolls_back_all_metadata(tmp_path, monkeypatch):
    root = _reference_session(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    original = reference.atomic_write_json
    def interrupt(path, value):
        if Path(path).name == 'manifest.json':
            raise KeyboardInterrupt('synthetic post-write interrupt')
        return original(path, value)
    monkeypatch.setattr(reference, 'atomic_write_json', interrupt)
    with pytest.raises(KeyboardInterrupt):
        reference.correct_sessions([root], previous='A2', authority='synthetic-test', output=tmp_path/'correction')
    assert all(path.read_bytes() == content for path, content in before.items())


def test_reference_conflicting_channel_reference_is_rejected_before_write(tmp_path):
    root = _reference_session(tmp_path)
    payload = json.loads((root/'session.json').read_text())
    channels = root/payload['recording']['bids']['channels_tsv']
    text = channels.read_text()
    rows = [line.split('\t') for line in text.splitlines()]
    if 'reference' not in rows[0]:
        rows[0].append('reference')
        for row in rows[1:]:
            row.append('A2')
    rows[1][rows[0].index('reference')] = 'other-reference'
    channels.write_text('\n'.join('\t'.join(row) for row in rows)+'\n')
    before = (root/'session.json').read_bytes()
    with pytest.raises(ValueError, match='reference'):
        reference.correct_sessions([root], previous='A2', authority='synthetic-test', output=tmp_path/'correction')
    assert (root/'session.json').read_bytes() == before


def _inference_input(repetitions):
    count = repetitions*9
    timeline = SimpleNamespace(evidence_available_times_s=np.arange(count, dtype=float),
        candidate_ids=np.tile(np.arange(1, 10), repetitions), evidence_indices=np.arange(count),
        repetition_indices=np.repeat(np.arange(repetitions), 9), session_ids=np.repeat('one', count), online_causal=False)
    dataset = SimpleNamespace(event_timeline=timeline, n_epochs=count, preprocessing=BRAIN_SYNC_PREPROCESSING)
    predictor = SimpleNamespace(validate_inference_scope=lambda dataset: None,
        predict_logit=lambda dataset: np.tile(np.arange(9, dtype=float), repetitions), metadata={})
    return predictor, dataset


def test_short_balanced_recording_is_not_a_product_prediction():
    predictor, dataset = _inference_input(1)
    result = task.inference_record(predictor, dataset)
    assert result['protocol_compliant'] is False
    assert result['prediction'] is None and result['status'] == 'abstained'
    assert result['scheduled_trials'] == 9


def test_full_r5_protocol_remains_usable_and_short_explicit_diagnostic_is_labelled():
    predictor, dataset = _inference_input(5)
    result = task.inference_record(predictor, dataset)
    assert result['protocol_compliant'] is True and result['prediction'] == 9
    predictor, dataset = _inference_input(1)
    diagnostic = task.inference_record(predictor, dataset, repetitions=1, diagnostic=True)
    assert diagnostic['evaluation_role'] == 'explicit_repetition_diagnostic'
    assert diagnostic['protocol_compliant'] is False and diagnostic['prediction'] == 9


def test_config_preprocessing_geometry_and_workstation_argument_are_one_contract():
    base = CONFIG_ROOT
    preprocessing = json.loads((base/'brainsync_bidirectional_preprocessing.json').read_text())
    support = json.loads((base/'brainsync_epoch_support.json').read_text())
    manifest = json.loads((base/'brainsync_workstation.model.json').read_text())
    assert preprocessing == json.loads(json.dumps(BRAIN_SYNC_PREPROCESSING.record()))
    assert (support['tmin_ms'], support['tmax_ms'], support['n_times'], support['model_sample_rate_hz']) == (
        preprocessing['tmin_ms'], preprocessing['tmax_ms'], preprocessing['n_times'], preprocessing['sfreq'])
    assert '--preprocessing-config' in manifest['parameter_options'][0]['arguments']
    assert manifest['input_contract']['preprocessing_geometry'] == support


def test_collection_plan_matches_current_acquisition_rest_and_per_session_protocol():
    path = CONFIG_ROOT/'brainsync_collection_plan.json'
    plan = json.loads(path.read_text())
    assert plan['calibration']['experiment']['rest_sec'] == 20
    test = plan['test']
    assert test['one_decision_is_one_new_recording'] is True
    assert test['experiment']['digits_per_block'] == 9*test['primary_repetitions'] == 45
    assert test['requested_test_stimulus_trials_per_participant'] == 27*45


def test_prepare_existing_cache_is_rejected_before_loading(tmp_path, monkeypatch):
    path = tmp_path/'epochs.npz'
    path.write_bytes(b'original cache')
    monkeypatch.setattr(prepare, 'load_brainsync_sessions_resilient', lambda *args, **kwargs: pytest.fail('existing output read input'))
    with pytest.raises(SystemExit) as error:
        prepare.main(['--session-dir', str(tmp_path/'session'), '--output', str(path)])
    assert error.value.code == 2 and path.read_bytes() == b'original cache'


def test_cache_publication_is_exclusive_and_rolls_back_its_npz_if_record_publish_fails(tmp_path, monkeypatch):
    from tests.support.datasets import epoch_dataset

    path = tmp_path/'epochs.npz'
    record_path = path.with_suffix('.record.json')
    original_link = os.link
    def fail_record_publication(staged, target):
        if Path(target) == record_path and Path(staged).suffix == '.json':
            raise OSError('synthetic record publication failure')
        return original_link(staged, target)
    monkeypatch.setattr(os, 'link', fail_record_publication)
    with pytest.raises(OSError, match='record publication'):
        publish_epoch_cache_new(path, epoch_dataset())
    assert not path.exists() and not record_path.exists()
    assert not list(tmp_path.glob('.*.npz')) and not list(tmp_path.glob('.*.json'))


def test_cli_cache_competing_publishers_preserve_one_complete_pair(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from data.epochs import load_epoch_dataset
    from tests.support.datasets import epoch_dataset

    path, barrier = tmp_path/'cache.npz', Barrier(2)
    def publish(_):
        barrier.wait()
        try:
            publish_epoch_cache_new(path, epoch_dataset())
            return 'published'
        except FileExistsError:
            return 'refused'
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(publish, range(2))) == ['published', 'refused']
    load_epoch_dataset(path, validation='attested')


def test_cli_cache_record_failure_preserves_competing_bytes(tmp_path, monkeypatch):
    from tests.support.datasets import epoch_dataset

    path = tmp_path/'epochs.npz'
    record_path = path.with_suffix('.record.json')
    original_link = os.link
    competing = b'independently published bytes must not be removed by this failed writer'
    def competing_record_failure(staged, target):
        if Path(target) == record_path and Path(staged).suffix == '.json':
            path.write_bytes(competing)
            raise OSError('synthetic competing writer before record publication failure')
        return original_link(staged, target)
    monkeypatch.setattr(os, 'link', competing_record_failure)
    with pytest.raises(OSError, match='competing writer'):
        publish_epoch_cache_new(path, epoch_dataset())
    assert path.read_bytes() == competing and not record_path.exists()


def test_competing_output_reservation_and_input_alias_are_rejected(tmp_path):
    path = tmp_path/'out.json'
    with new_output_reservation(path):
        with pytest.raises(FileExistsError):
            with new_output_reservation(path):
                pytest.fail('second writer entered')
    with pytest.raises(ValueError, match='differ'):
        with new_output_reservation(path, inputs=(path,)):
            pytest.fail('input alias entered')
    publish_json_new(path, {'original': 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        publish_json_new(path, {'new': 2})
    assert path.read_bytes() == before


def test_cross_decision_rejects_output_input_collision_before_snapshot_or_eeg(tmp_path, monkeypatch):
    path = tmp_path/'cache.npz'
    path.write_bytes(b'original')
    monkeypatch.setattr(cross, 'source_snapshot_sha256_from_archive_manifest', lambda *args: pytest.fail('input loaded before output guard'))
    with pytest.raises(ValueError, match='differ'):
        cross.main(['--dataset-cache', str(path), '--checkpoint', str(tmp_path/'checkpoint.pt'),
            '--source-snapshot-manifest', str(tmp_path/'source.json'), '--arm-name', 'arm',
            '--identity-exclusion-policy', 'source_or_global', '--target-policy', 'observed_sequence', '--output', str(path)])
    assert path.read_bytes() == b'original'


def test_reference_postvalidation_failure_rolls_back_and_recovery_journal_is_idempotent(tmp_path, monkeypatch):
    root = _reference_session(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    def invalid(root):
        raise ValueError('synthetic postvalidation disagreement')
    monkeypatch.setattr(reference, 'validate_brainsync_bids_recording', invalid)
    output = tmp_path/'correction'
    with pytest.raises(ValueError, match='postvalidation'):
        reference.correct_sessions([root], previous='A2', authority='synthetic-test', output=output)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not (output/'manifest.json').exists()
    assert reference.recover_reference_transaction(output)['restored_files'] == 0


def test_short_fully_rejected_recording_retains_r5_abstention_instead_of_index_error(tmp_path, monkeypatch):
    from data.brainsync import BrainSyncNoUsableEvidenceError
    class RejectedError(BrainSyncNoUsableEvidenceError):
        def __init__(self):
            self.scheduled_trials, self.session_id, self.participant_key, self.ledger = 7, 'one', 'BrainSync-GTN:P001', {}
    def reject(root):
        raise RejectedError()
    predictor = SimpleNamespace(metadata={'subject_id': 'P001', 'input_contract': {'simulation_status': 'declared_mock'}},
        validate_personal_workflow_metadata=lambda: None, load_inference_session=reject)
    monkeypatch.setattr(task, 'load_calibrated_predictor', lambda *args, **kwargs: predictor)
    artifact = tmp_path/'model.pt'
    artifact.write_bytes(b'software placeholder intercepted by numeric loader stub')
    task.main(['--task', 'infer', '--session-dir', str(tmp_path/'recording'), '--model-artifact', str(artifact),
        '--output-dir', str(tmp_path/'output')])
    result = json.loads((tmp_path/'output/summary.json').read_text())
    assert result['status'] == 'abstained' and result['scheduled_trials'] == 7
    assert result['requested_repetitions'] == 5 and len(result['by_repetition']) == 5
    assert result['protocol_compliant'] is False


def test_pilot_rejects_unsafe_arm_name_before_snapshot_or_cache(tmp_path):
    plan = {'schema': 'n2p3_brainsync_pilot/1', 'input_sha256': {'a': '0'*64}, 'batch_rule': {},
        'caches': {'calibration': 'a', 'test': 'b', 'full': 'c'}, 'device': 'cpu',
        'arms': [{'name': '../escape', 'source': 'none', 'recipe': {'name': 'recipe'}}]}
    with pytest.raises(ValueError, match='safe'):
        pilot.validate_pilot_plan(plan)


def test_quality_output_cannot_be_session_root_or_overwrite_existing_assets(tmp_path):
    with pytest.raises(SystemExit) as error:
        report.main(['--session-dir', str(tmp_path), '--output-dir', str(tmp_path)])
    assert error.value.code == 2


def test_model_task_bad_recipe_writes_failed_receipt(tmp_path):
    recipe = tmp_path/'broken.json'
    recipe.write_text('{')
    output = tmp_path/'output'
    with pytest.raises(SystemExit) as error:
        task.main(['--task', 'calibrate', '--selector-mode', 'fixed_erp', '--calibration-recipe', str(recipe),
            '--session-dir', str(tmp_path/'session'), '--model-artifact', str(tmp_path/'model.pt'), '--output-dir', str(output)])
    assert error.value.code == 1
    result = json.loads((output/'summary.json').read_text())
    assert result['status'] == 'failed' and result['error']['type'] == 'JSONDecodeError'


def test_eog_cli_passes_the_selected_preprocessing_without_fitting(tmp_path, monkeypatch):
    from experiments import fit_brainsync_eog as eog
    config = tmp_path/'preprocessing.json'
    declared = replace(BRAIN_SYNC_PREPROCESSING, l_freq=.5)
    config.write_text(json.dumps(declared.record()))
    seen = {}
    def capture(*args, **kwargs):
        seen.update(kwargs)
        return object()
    monkeypatch.setattr(eog, 'fit_brainsync_eog_sessions', capture)
    monkeypatch.setattr(eog, 'save_eog_regression', lambda path, model: path)
    eog.main(['--calibration-session-dir', str(tmp_path/'session'), '--output', str(tmp_path/'eog.json'),
        '--preprocessing-config', str(config)])
    assert seen['preprocessing'] == declared


def test_quality_frozen_configuration_error_is_nonzero_with_explanation(tmp_path, monkeypatch):
    model = tmp_path/'quality.json'
    model.write_text('invalid frozen model')
    seen = {}
    def create(root, output, **kwargs):
        try:
            kwargs['loader'](root)
        except ValueError as error:
            seen['error'] = str(error)
        else:
            pytest.fail('invalid frozen model was accepted')
        return {'status': 'not_evaluable', 'error': seen['error']}
    monkeypatch.setattr(report, 'create_recording_quality_report', create)
    with pytest.raises(SystemExit) as error:
        report.main(['--session-dir', str(tmp_path/'session'), '--output-dir', str(tmp_path/'output'),
            '--epoch-qc-model', str(model)])
    assert error.value.code == 3 and seen['error']


def test_reference_raw_postcondition_failure_rolls_back_only_its_metadata(tmp_path, monkeypatch):
    root = _reference_session(tmp_path)
    metadata = {path: path.read_bytes() for path in root.rglob('*') if path.is_file() and path.suffix != '.edf'}
    signal = next(root.rglob('*.edf'))
    original = reference._replace_metadata
    changed = []
    def external_raw_change(path, content):
        original(path, content)
        if not changed:
            signal.write_bytes(signal.read_bytes()+b'external concurrent signal write')
            changed.append(True)
    monkeypatch.setattr(reference, '_replace_metadata', external_raw_change)
    with pytest.raises(ValueError, match='EDF changed'):
        reference.correct_sessions([root], previous='A2', authority='synthetic-test', output=tmp_path/'correction')
    assert all(path.read_bytes() == content for path, content in metadata.items())
    assert signal.read_bytes().endswith(b'external concurrent signal write')


def test_real_cache_codec_staged_publication_reloads_its_exact_attestation(tmp_path):
    from data.brainsync import load_brainsync_session
    from data.epochs import load_epoch_dataset
    session = write_bids_session(tmp_path/'session', session_id='cache-audit', target=3,
        started_utc='2026-10-19T00:00:00Z')
    dataset = load_brainsync_session(session)
    path = publish_epoch_cache_new(tmp_path/'epochs.npz', dataset)
    loaded = load_epoch_dataset(path, validation='attested')
    assert np.array_equal(loaded.X, dataset.X)
    assert loaded.preprocessing == dataset.preprocessing
    with pytest.raises(FileExistsError):
        publish_epoch_cache_new(path, dataset)


def test_continuous_qc_records_the_actual_preprocessing_without_real_fit(tmp_path, monkeypatch):
    declared = replace(BRAIN_SYNC_PREPROCESSING, filter_phase='forward', zero_phase_edge_policy='legacy',
        causal_iir_initial_state='steady_state_first_sample')
    raw_path = tmp_path/'synthetic-raw.edf'
    raw_path.write_bytes(b'synthetic mocked reader input')
    item = SimpleNamespace(subject_id='P001', session_id='one', ended_timestamp_s=3,
        manifest={'recording': {}}, bids=SimpleNamespace(raw_path=raw_path, source_reference='left earlobe',
            channel_types=('EEG',), channel_statuses=('good',), acquisition_boundaries=(), excluded_intervals=(), close=lambda: None))
    raw = SimpleNamespace(n_times=20, ch_names=['Pz'], info={'sfreq': 100.}, set_channel_types=lambda mapping: None)
    monkeypatch.setattr(qc, 'validate_brainsync_bids_session', lambda root: item)
    monkeypatch.setattr(qc, 'load_brainsync_session', lambda root, **kwargs: object())
    table = SimpleNamespace(record_for=lambda subject: SimpleNamespace(authority_key=lambda policy: 'BrainSync-GTN:P001'))
    monkeypatch.setattr(qc, 'materialize_dataset_identity', lambda dataset: table)
    monkeypatch.setattr(qc, 'read_raw', lambda *args, **kwargs: raw)
    monkeypatch.setattr(qc.mne, 'concatenate_raws', lambda values, **kwargs: values[0])
    captured = {}
    def numeric_owner(raw, **kwargs):
        captured.update(kwargs)
        return 'software-captured-no-fit'
    monkeypatch.setattr(qc, 'fit_continuous_qc', numeric_owner)
    manifest = {'schema': 'brainsync_continuous_qc_calibration/1', 'preprocessing': declared.record(),
        'spec': {'difference_z': 3., 'difference_floor_v': 1e-6, 'settle_samples': 1},
        'sessions': [{'session_dir': 'session', 'clean_intervals_samples': [[2, 10]], 'clean_interval_evidence': 'synthetic declaration'}]}
    path = tmp_path/'calibration.json'
    path.write_text(json.dumps(manifest))
    assert qc.fit_brainsync_continuous_manifest(path) == 'software-captured-no-fit'
    assert captured['fit_provenance']['preprocessing'] == declared.record()
    assert np.array_equal(np.flatnonzero(captured['fit_sample_mask']), np.arange(2, 10))


def test_inspect_summary_uses_the_saved_personal_workflow_gate_before_metadata_indexing():
    def require():
        raise ValueError('personal workflow metadata is missing: input_request')
    with pytest.raises(ValueError, match='personal workflow metadata'):
        task._model_summary(SimpleNamespace(metadata={}, validate_personal_workflow_metadata=require))


def test_artifact_failure_receipts_are_immutable_and_preserve_existing_inputs(tmp_path):
    artifact = tmp_path/'existing.pt'
    artifact.write_bytes(b'original model')
    output = tmp_path/'output'
    with pytest.raises(SystemExit):
        task.main(['--task', 'calibrate', '--session-dir', str(tmp_path/'session'),
            '--model-artifact', str(artifact), '--output-dir', str(output)])
    before = (output/'summary.json').read_bytes()
    with pytest.raises(SystemExit) as error:
        task.main(['--task', 'calibrate', '--session-dir', str(tmp_path/'session'),
            '--model-artifact', str(artifact), '--output-dir', str(output)])
    assert error.value.code == 2 and artifact.read_bytes() == b'original model'
    assert (output/'summary.json').read_bytes() == before


def test_reference_prepared_transaction_recovers_after_interrupted_rollback(tmp_path, monkeypatch):
    root = _reference_session(tmp_path)
    original_bytes = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    original_replace = reference._replace_metadata
    calls = []
    def interrupted(path, content):
        calls.append(path)
        if len(calls) == 1:
            original_replace(path, content)
            raise KeyboardInterrupt('synthetic rename completed before interruption')
        raise OSError('synthetic rollback storage failure')
    monkeypatch.setattr(reference, '_replace_metadata', interrupted)
    output = tmp_path/'correction'
    with pytest.raises(KeyboardInterrupt):
        reference.correct_sessions([root], previous='A2', authority='synthetic-test', output=output)
    assert (output/'transaction.json').is_file() and not (output/'manifest.json').exists()
    monkeypatch.setattr(reference, '_replace_metadata', original_replace)
    assert reference.recover_reference_transaction(output)['restored_files'] == 1
    assert all(path.read_bytes() == value for path, value in original_bytes.items())


def test_reference_postvalidation_accepts_unlabelled_recording_without_creating_truth(tmp_path):
    root = _reference_session(tmp_path)
    path = root/'session.json'
    original = json.loads(path.read_text())
    original.pop('target_label')
    original['experiment'].pop('thought_digit')
    original['experiment'].pop('target_label_status')
    path.write_text(json.dumps(original))
    reference.correct_sessions([root], previous='A2', authority='synthetic-test', output=tmp_path/'correction')
    corrected = json.loads(path.read_text())
    assert 'target_label' not in corrected and 'thought_digit' not in corrected['experiment']


@pytest.mark.parametrize('extra', [('--seed', '1'), ('--batch-size', '32'), ('--selection-protocol', 'binary_auc')])
def test_inspect_rejects_unused_calibration_options_before_loading(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        task.main(['--task', 'inspect', '--model-artifact', str(tmp_path/'missing.pt'),
            '--output-dir', str(tmp_path/'output'), *extra])
    assert error.value.code == 2 and not (tmp_path/'output/summary.json').exists()


@pytest.mark.parametrize('extra', [('--epochs', '1'), ('--lr', '.1'), ('--batch-size', '32')])
def test_zero_shot_rejects_unused_fit_options_before_any_input(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(cross, 'source_snapshot_sha256_from_archive_manifest', lambda *args: pytest.fail('unused argument read input'))
    with pytest.raises(SystemExit) as error:
        cross.main(['--dataset-cache', str(tmp_path/'cache.npz'), '--checkpoint', str(tmp_path/'checkpoint.pt'),
            '--source-snapshot-manifest', str(tmp_path/'snapshot.json'), '--arm-name', 'zero',
            '--identity-exclusion-policy', 'source_or_global', '--target-policy', 'observed_sequence',
            '--output', str(tmp_path/'result.json'), *extra])
    assert error.value.code == 2 and not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('fail_postcondition', [False, True])
def test_reference_postvalidator_releases_live_data_lease_before_return_or_rollback(tmp_path, monkeypatch, fail_postcondition):
    root = _reference_session(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    actual_validator = reference.validate_brainsync_bids_recording
    held, closed = [], []
    def validator(path):
        result = actual_validator(path)
        held.append(result)
        def assert_postcondition():
            result.bids.assert_content_unchanged()
            if fail_postcondition:
                raise ValueError('synthetic live-lease postcondition failure')
        def close():
            closed.append(result)
            result.bids.close()
        return SimpleNamespace(bids=SimpleNamespace(assert_content_unchanged=assert_postcondition, close=close))
    monkeypatch.setattr(reference, 'validate_brainsync_bids_recording', validator)
    if fail_postcondition:
        with pytest.raises(ValueError, match='live-lease postcondition'):
            reference.correct_sessions([root], previous='A2', authority='synthetic-test', output=tmp_path/'correction')
        assert all(path.read_bytes() == content for path, content in before.items())
    else:
        reference.correct_sessions([root], previous='A2', authority='synthetic-test', output=tmp_path/'correction')
        # held retains the validated object: this write succeeds only after
        # the explicit close, independently of garbage collection.
        path = root/'session.json'
        path.write_bytes(path.read_bytes())
    assert len(held) == len(closed) == 1
