import copy
import hashlib
import io
from dataclasses import replace

import numpy as np
import pytest
import torch

from data.identity import DatasetIdentityTable, ParticipantIdentityRecord, origin_subject_key
from research.contracts import semantic_sha256
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from transfer.calibration_candidates import fit_calibration_candidate
from transfer.checkpoint import (
    ScopedSourceCheckpointPreparation,
    checkpoint_input_stats,
    checkpoint_training_identity_ledger,
    load_n2p3_trunk_checkpoint,
)
from transfer.recipes import TargetRecipe


def source_fixture(tmp_path):
    dataset = binary_dataset()
    path = write_checkpoint(tmp_path / 'source.pt', dataset, source_snapshot_sha256='a' * 64)
    return dataset, path.read_bytes()


def test_same_source_rebinds_current_fold_and_separates_weights_bn_stats(tmp_path, monkeypatch):
    import transfer.checkpoint as checkpoint

    dataset, content = source_fixture(tmp_path)
    calls = []
    original = checkpoint._validate_source_metadata

    def counted(payload):
        calls.append(payload['training_contract_digest'])
        return original(payload)

    monkeypatch.setattr(checkpoint, '_validate_source_metadata', counted)
    owner = ScopedSourceCheckpointPreparation()
    first = owner.prepare(content, dataset, 'p1')
    # A new current table is bound, even when the physical person has an alias.
    alias = copy.deepcopy(dataset)
    alias.identity_table = dataset.identity_table.relabel_local_subjects({'p1': 'current-fold'})
    alias.subject_ids = np.full(dataset.n_epochs, 'current-fold')
    second = owner.prepare(content, alias, 'current-fold')
    assert len(calls) == 1
    with pytest.raises(ValueError):
        second.trunk_for_target('p1')
    a, b = first.trunk_for_target('p1'), second.trunk_for_target('current-fold')
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    with torch.no_grad():
        for value in a.state_dict().values():
            value.add_(1)
    c = second.trunk_for_target('current-fold')
    for key, value in b.state_dict().items():
        torch.testing.assert_close(value, c.state_dict()[key], rtol=0, atol=0)
    first.input_statistics[0][:] = 123
    first.input_statistics[1][:] = 456
    third = owner.prepare(content, dataset, 'p1')
    assert np.array_equal(second.input_statistics[0], third.input_statistics[0])
    assert np.array_equal(second.input_statistics[1], third.input_statistics[1])
    with pytest.raises(TypeError):
        first.payload['training_contract']['seed'] = 1
    assert owner.record()['source_admissions'] == 1
    assert owner.record()['target_bindings'] == 3


@pytest.mark.parametrize('change', ['channels', 'reference', 'sfreq', 'tmin', 'preprocessing', 'coordinates'])
def test_cached_source_preserves_current_input_gates(tmp_path, change):
    dataset, content = source_fixture(tmp_path)
    owner = ScopedSourceCheckpointPreparation()
    owner.prepare(content, dataset, 'p1')
    changed = copy.deepcopy(dataset)
    if change == 'channels':
        changed.channel_names = tuple(reversed(dataset.channel_names))
    elif change == 'reference':
        changed.provenance['source_reference'] = 'changed-measured-reference'
    elif change == 'sfreq':
        changed.preprocessing = replace(dataset.preprocessing, sfreq=dataset.preprocessing.sfreq * 2)
    elif change == 'tmin':
        changed.preprocessing = replace(dataset.preprocessing, tmin_ms=dataset.preprocessing.tmin_ms + 10)
    elif change == 'preprocessing':
        changed.preprocessing = replace(dataset.preprocessing, h_freq=29.)
    else:
        changed.coordinate_system = 'dimensionless_EA'
    with pytest.raises(ValueError):
        load_n2p3_trunk_checkpoint(io.BytesIO(content), changed, target_subject='p1')
    with pytest.raises(ValueError):
        owner.prepare(content, changed, 'p1')
    assert owner.record()['target_bindings'] == 1


def test_current_participant_global_exclusion_and_changed_source_bytes(tmp_path):
    dataset, content = source_fixture(tmp_path)
    owner = ScopedSourceCheckpointPreparation()
    owner.prepare(content, dataset, 'p1')
    other = copy.deepcopy(dataset)
    other.identity_table = DatasetIdentityTable(records=(ParticipantIdentityRecord(
        local_subject_id='p1', origin_subject_keys=(origin_subject_key('new-study', 'new-person'),),
        identity_status='source_verified'),))
    with pytest.raises(ValueError, match='participant'):
        owner.prepare(content, other, 'p1')
    payload = torch.load(io.BytesIO(content), weights_only=False)
    payload['training_contract']['seed'] = 77
    stream = io.BytesIO()
    torch.save(payload, stream)
    with pytest.raises(ValueError, match='digest'):
        owner.prepare(stream.getvalue(), dataset, 'p1')
    assert owner.record()['source_admissions'] == 1

    source_table = DatasetIdentityTable(records=(ParticipantIdentityRecord(
        local_subject_id='external-source', origin_subject_keys=(origin_subject_key('other', 'external-source'),),
        global_person_keys=('verified-person-1',), identity_status='global_verified'),))
    payload['training_identity_ledger'] = source_table.payload()
    payload['training_identity_ledger_digest'] = source_table.digest()
    payload['training_contract']['training_participant_keys'] = source_table.authority_keys()
    payload['training_contract']['source_identity_digest'] = source_table.digest()
    payload.pop('source_study_membership')
    payload['training_contract']['objective'].pop('source_study_membership')
    payload['training_contract_digest'] = semantic_sha256(payload['training_contract'])
    stream = io.BytesIO()
    torch.save(payload, stream)
    globally_same = copy.deepcopy(dataset)
    globally_same.identity_table = DatasetIdentityTable(records=(ParticipantIdentityRecord(
        local_subject_id='p1', origin_subject_keys=(origin_subject_key('target', 'apparently-new'),),
        global_person_keys=('verified-person-1',), identity_status='global_verified'),))
    with pytest.raises(ValueError, match='includes target'):
        ScopedSourceCheckpointPreparation().prepare(stream.getvalue(), globally_same, 'p1')


@pytest.mark.parametrize('initialization', ['source', 'scratch'])
def test_recipe_seed_reset_preserves_actual_fitted_state_after_ambient_rng_change(tmp_path, initialization):
    dataset, content = source_fixture(tmp_path)
    original, payload = load_n2p3_trunk_checkpoint(io.BytesIO(content), dataset, target_subject='p1')
    owner = ScopedSourceCheckpointPreparation()
    bound = owner.prepare(content, dataset, 'p1')
    recipe = replace(TargetRecipe('seed-control', batch_size=8, epochs=2, seed=19), initialization=initialization)
    rows = np.arange(dataset.n_epochs) < 20
    torch.manual_seed(111)
    torch.rand(17)
    expected = fit_calibration_candidate(dataset, recipe, rows, source_trunk=original,
        source_input_stats=checkpoint_input_stats(payload, dataset.n_channels), device='cpu')
    torch.manual_seed(999)
    torch.rand(113)
    actual = fit_calibration_candidate(dataset, recipe, rows, source_trunk=bound.trunk_for_target('p1'),
        source_input_stats=bound.input_statistics, device='cpu')
    for key, value in expected.state.items():
        torch.testing.assert_close(value, actual.state[key], rtol=0, atol=0)
    np.testing.assert_array_equal(expected.predict_logit(dataset.X), actual.predict_logit(dataset.X))
    assert expected.fit_history['initial_state_sha256'] == actual.fit_history['initial_state_sha256']
    assert expected.fit_history['probability_fit'] == actual.fit_history['probability_fit']
    assert hashlib.sha256(content).hexdigest() == owner.record()['source_keys'][0][1]


def test_admitted_source_study_relation_uses_original_proof_not_frozen_metadata(tmp_path):
    from data.study import dataset_study_membership

    dataset, content = source_fixture(tmp_path)
    owner = ScopedSourceCheckpointPreparation()
    first = owner.prepare(content, dataset, 'p1')
    assert first.source_membership_present
    proof = dataset_study_membership(dataset)
    first.assert_library_exclusion(target_membership=proof)
    # Presentation metadata remains frozen.  The external identity parser is
    # deliberately strict; re-parsing this copy cannot grant source authority.
    with pytest.raises(ValueError):
        checkpoint_training_identity_ledger(first.payload)
    assert isinstance(first.payload['source_study_membership']['evidence'], tuple)
    with pytest.raises(ValueError, match='both source and target'):
        first.assert_library_exclusion()

    changed = copy.deepcopy(dataset)
    changed.provenance['dataset_class'] = 'external-study'
    bound = owner.prepare(content, changed, 'p1')
    with pytest.raises(ValueError, match='excluded target libraries'):
        bound.assert_library_exclusion(target_membership=dataset_study_membership(changed))
    assert owner.record()['source_admissions'] == 1 and owner.record()['target_bindings'] == 2


def test_absent_source_membership_keeps_literal_policy_and_rejects_single_proof(tmp_path):
    from data.study import dataset_study_membership

    dataset, content = source_fixture(tmp_path)
    payload = torch.load(io.BytesIO(content), weights_only=False)
    payload.pop('source_study_membership')
    payload['training_contract']['objective'].pop('source_study_membership')
    payload['training_contract_digest'] = semantic_sha256(payload['training_contract'])
    stream = io.BytesIO()
    torch.save(payload, stream)
    bound = ScopedSourceCheckpointPreparation().prepare(stream.getvalue(), dataset, 'p1')
    assert not bound.source_membership_present and 'source_study_membership' not in bound.payload
    bound.assert_library_exclusion()
    with pytest.raises(ValueError, match='both source and target'):
        bound.assert_library_exclusion(target_membership=dataset_study_membership(dataset))


def test_literal_library_permission_comes_from_private_admitted_ledger(tmp_path, monkeypatch):
    from data.identity import source_dataset_ids_from_identity

    dataset, content = source_fixture(tmp_path)
    payload = torch.load(io.BytesIO(content), weights_only=False)
    target_stream = source_dataset_ids_from_identity(dataset.identity_table)[0]
    actual_source = DatasetIdentityTable.from_source_rows(['another-person'], [target_stream])
    payload['training_identity_ledger'] = actual_source.payload()
    payload['training_identity_ledger_digest'] = actual_source.digest()
    payload['training_contract']['source_identity_digest'] = actual_source.digest()
    payload['training_contract']['training_participant_keys'] = actual_source.authority_keys()
    payload.pop('source_study_membership')
    payload['training_contract']['objective'].pop('source_study_membership')
    payload['training_contract_digest'] = semantic_sha256(payload['training_contract'])
    stream = io.BytesIO()
    torch.save(payload, stream)
    bound = ScopedSourceCheckpointPreparation().prepare(stream.getvalue(), dataset, 'p1')
    with pytest.raises(ValueError, match='excluded target libraries'):
        bound.assert_library_exclusion()
    foreign = DatasetIdentityTable.from_source_rows(['unrelated'], ['foreign-study'])
    with pytest.raises(AttributeError):
        bound.training_identity = foreign
    # A presentation getter remains outside the permission path even if an
    # extension overrides it.  Source bytes/current target are unchanged.
    monkeypatch.setattr(type(bound), 'training_identity', property(lambda self: foreign))
    assert bound.training_identity is foreign
    with pytest.raises(ValueError, match='excluded target libraries'):
        bound.assert_library_exclusion()


def test_prefix_k0_full_and_oof_share_source_admission_with_current_study_gates(tmp_path, monkeypatch):
    from research.nested_selection_replay import InnerReplayRecorder
    from research.prefix_inner_replay import persist_prefix_inner_replay, replay_prefix_inner_replay
    from tests.support.independent_prefix import independent_prefix_case
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import CalibrationExpertProgram, fit_calibration_prefix_programs
    from transfer.ensemble_views import prepare_bnci_independent_prefixes

    _, _, path, authorization, experts, views = independent_prefix_case(tmp_path)
    profile = experts[1].profile
    prepared = prepare_bnci_independent_prefixes(path, source_sha256=views.ledger['source_sha256'],
        budgets=(0, 400), profiles=(profile,), authorization_preprocessing=authorization)
    checkpoint = write_checkpoint(tmp_path / 'source-prefix.pt', prepared.calibration[400].trial_datasets[profile.digest],
                                  source_snapshot_sha256='c' * 64)
    source = CalibrationExpertProgram('source', 'trial', TargetRecipe('source', initialization='source',
        input_statistics='source', epochs=1, batch_size=512, seed=0, dropout='off'), profile)
    roster = EnsembleProgram.equal_weights('cached-prefix', ('source',))
    owner = ScopedSourceCheckpointPreparation()
    import transfer.checkpoint as checkpoint_owner

    source_calls, current_calls = [], []
    original_source, original_target = checkpoint_owner._validate_source_metadata, checkpoint_owner._validate_target_metadata

    def counted_source(payload):
        source_calls.append(payload['training_contract_digest'])
        return original_source(payload)

    def counted_target(payload, data, admission, **kwargs):
        current_calls.append(data.identity_table.digest())
        return original_target(payload, data, admission, **kwargs)

    monkeypatch.setattr(checkpoint_owner, '_validate_source_metadata', counted_source)
    monkeypatch.setattr(checkpoint_owner, '_validate_target_metadata', counted_target)
    k0 = fit_calibration_prefix_programs(prepared.calibration[0], (source,), roster,
        strategies=('fixed', 'grouped_temperature'), source_checkpoints={'source': checkpoint}, _source_preparation=owner)
    assert k0['fixed'].expert_fit_available['source']
    recorder = InnerReplayRecorder()
    programs = fit_calibration_prefix_programs(prepared.calibration[400], (source,), roster,
        strategies=('fixed', 'grouped_temperature'), source_checkpoints={'source': checkpoint},
        _source_preparation=owner, inner_replay_recorder=recorder)
    assert len(source_calls) == 1 and len(current_calls) == 5  # K0/full C/three held-group fits.
    assert owner.record()['source_admissions'] == 1 and owner.record()['target_bindings'] == 5
    assert len(recorder.producers) == 3 and programs['grouped_temperature'].temperature_record['strategy'] == 'grouped_temperature'
    directory = persist_prefix_inner_replay(prepared.calibration[400], programs, tmp_path / 'prefix-package', recorder=recorder)
    replay = replay_prefix_inner_replay(directory, prefix=prepared.calibration[400], source_checkpoints={'source': checkpoint})
    assert replay['producer_count'] == replay['exact_score_checks'] == 3


def test_one_view_profile_transform_is_shared_but_reloaded_scores_compute_again(tmp_path, monkeypatch):
    from tests.support.independent_prefix import independent_prefix_case
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import (
        CalibrationExpertProgram,
        FittedEnsemble,
        fit_calibration_prefix_programs,
    )
    from transfer.ensemble_views import SelectionEpochProfile
    from transfer.recipes import erp_target_recipe

    _, _, _, _, original_specs, views = independent_prefix_case(tmp_path)
    profile = original_specs[1].profile
    recipe = erp_target_recipe('toeplitz_lda', bin_width=64)
    specs = tuple(CalibrationExpertProgram(name, 'trial', recipe, profile) for name in ('first', 'second'))
    program = EnsembleProgram.equal_weights('same-current-profile', tuple(spec.expert_id for spec in specs))
    fitted = fit_calibration_prefix_programs(views.calibration[200], specs, program, strategies=('fixed',))
    model = fitted['fixed']
    view = next(view for view in views.decisions if view.samples is not None)
    calls = []
    original = SelectionEpochProfile.transform

    def counted(self, current):
        calls.append((self.digest, current.record_id, current.raw_input_digest))
        return original(self, current)

    monkeypatch.setattr(SelectionEpochProfile, 'transform', counted)
    uncached = {expert.program.expert_id: expert.score(view).record() for expert in model._experts}
    assert len(calls) == 2
    calls.clear()
    captured = {name: score.record() for name, score in model.score_experts(view).items()}
    assert len(calls) == 1 and captured == uncached
    directory = FittedEnsemble.save_programs(tmp_path / 'same-profile', fitted, participant_key=model.participant_key)
    loaded = FittedEnsemble.load_programs(directory)['fixed']
    calls.clear()
    replayed = {name: score.record() for name, score in loaded.score_experts(view).items()}
    assert len(calls) == 1 and replayed == captured
    changed = replace(view, samples=np.asarray(view.samples) * 1.01)
    calls.clear()
    altered = {name: score.record() for name, score in loaded.score_experts(changed).items()}
    assert len(calls) == 1 and altered != replayed


def test_actual_prefix_caller_source_scratch_cache_and_parent_admission(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import experiments.run_candidate_ensemble_study as runner
    from data.bnci2014_008_candidate import BNCI2014_008_SOURCE_REFERENCE
    from research.cloud_execution import CloudResult
    from tests.support.independent_prefix import independent_prefix_case
    from transfer.calibration_candidates import waveform_lda_recipe
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import CalibrationExpertProgram
    from transfer.ensemble_views import SelectionEpochProfile, prepare_bnci_independent_prefixes

    _, _, raw_path, authorization, _, native = independent_prefix_case(tmp_path)
    profile = SelectionEpochProfile.from_preprocessing(replace(authorization, baseline_mode='none'),
        source_sample_rate_hz=256., channel_names=('Fz', 'P3', 'Pz', 'P4', 'PO8', 'Oz'),
        reference=BNCI2014_008_SOURCE_REFERENCE, reference_channel='Cz')
    views = prepare_bnci_independent_prefixes(raw_path, source_sha256=native.ledger['source_sha256'],
        budgets=(0, 200), profiles=(profile,), authorization_preprocessing=authorization)
    checkpoint = write_checkpoint(tmp_path / 'source-current.pt', views.calibration[200].trial_datasets[profile.digest],
                                  source_snapshot_sha256='d' * 64)
    payload = torch.load(checkpoint, weights_only=False)
    payload['training_contract']['seed'] = 11
    payload['training_contract']['optimizer']['config'] = {'pos_weight': 8.0}
    payload['training_contract_digest'] = semantic_sha256(payload['training_contract'])
    torch.save(payload, checkpoint)
    recipe = TargetRecipe('NN', initialization='source', input_statistics='source', epochs=1,
        batch_size=512, seed=21, dropout='off')
    specs = (CalibrationExpertProgram('NN', 'trial', recipe, profile), CalibrationExpertProgram('scratch', 'trial',
        replace(recipe, name='scratch', initialization='scratch', input_statistics='target_prefix'), profile),
        CalibrationExpertProgram('waveform', 'trial', waveform_lda_recipe(), profile))
    mixtures = tuple(EnsembleProgram.equal_weights(spec.expert_id, (spec.expert_id,)) for spec in specs[:2])
    plan = {'schema': 'n2p3_candidate_ensemble_study/1', 'evidence_scope': 'software_correctness',
        'evaluation_design': 'independent_prefix', 'seed_pairs': [[11, 21]],
        'records': [{'path': str(raw_path), 'sha256': native.ledger['source_sha256']}],
        'budgets': [0, 200], 'experts': [spec.record() for spec in specs], 'mixtures': [pool.record() for pool in mixtures],
        'baseline_mixture': 'NN', 'authorization_preprocessing': authorization.record(),
        'observation_response_seconds': 308 / 256, 'pre_roll_seconds': .2, 'device': 'cpu',
        'workers': 1, 'cpu_threads_per_worker': 1, 'source_checkpoints': {},
        'source_seed_catalog': [{'outer_target_study': 'BNCI2014-008', 'source_seed': 11,
            'path': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'procedure': {'kind': 'fixed_recipe', 'selection': 'not_applicable'}}],
        'calibration_strategies': ['fixed', 'grouped_temperature']}
    path = tmp_path / 'caller-plan.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    owners = []

    class TrackedPreparation(ScopedSourceCheckpointPreparation):
        def __init__(self):
            super().__init__()
            owners.append(self)

    monkeypatch.setattr(runner, 'ScopedSourceCheckpointPreparation', TrackedPreparation)
    output = tmp_path / 'caller-run'
    report = runner.run_study(path, output, execution='inline')
    task = SimpleNamespace(task_id='software:0', output=str(output), arguments={'record_index': 0},
                           resources=SimpleNamespace(native_threads=1))
    outcome = CloudResult(task.task_id, 'completed', str(output / 'participant_000' / 'result.json'))
    admitted = runner._admit_candidate_participant(task, outcome,
        json.loads((output / 'plan.json').read_bytes()), json.loads((output / 'executed_source.json').read_bytes()))
    assert admitted['status'] == report['status'] == 'completed'
    assert len(owners) == 1 and owners[0].record()['source_admissions'] == 1
    assert owners[0].record()['target_bindings'] == 3  # Source-only K0 plus source/scratch full C.
    assert len(admitted['predictions']) == 2 * 2 * 2 * len(views.decisions)
    assert {row['source_seed'] for row in admitted['predictions']} == {11}
    assert {row['target_seed'] for row in admitted['predictions']} == {21}
