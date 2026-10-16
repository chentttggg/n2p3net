"""Requested failure ledgers and captured-only replay, without source refits."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.execution import ExpectedSubjectError, SubjectFailureCode
from research.joint_flat_replay import _replay_expert_scores, _score_programs
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, _forward_joint_temperature
from transfer.joint_flat_programme import JointFlatProgram, JointFlatSamplingProgram


def _view(ordinal=0, *, reason=None):
    return SimpleNamespace(record_id=f'held-{ordinal}', participant_key='p', decision_id=f'd-{ordinal}',
        raw_input_digest='1'*64, physical_support_digest='2'*64, onsets=np.array([2, 3]), raw_scope=(0, 6),
        unavailable_reason=reason, decision=SimpleNamespace(candidate_vocabulary=('A', 'B'), repetitions=1,
            membership=np.eye(2, dtype=bool), evidence_indices=np.arange(2), schedule_complete=True))


def _evidence(view, *, reason=None):
    identity = CandidateExpertIdentity('joint', '1'*64, '3'*64, '4'*64, '5'*64, 'test', False)
    return CandidateEvidence(('A', 'B'), np.log([.3, .7]) if reason is None else None, 'p', view.decision_id,
        view.physical_support_digest, (identity,), 1, (1, 1), (1, 1), failure_reason=reason,
        failure_kind=None if reason is None else 'not_available')


def _programme():
    sampler = JointFlatSamplingProgram(4., 8, 8, (2, 3), 1, 2, 1., 1.2, 5., 5., .1)
    return CalibrationExpertProgram('joint', 'joint_flat', joint_flat=JointFlatProgram(sampler, 2, 'whole_observed_current_q'))


def test_forward_oof_preserves_all_ordered_attempts_after_fit_and_score_failures(monkeypatch):
    from transfer import ensemble_fit, ensemble_views

    groups = tuple(SimpleNamespace(view=_view(i), target_candidate='A') for i in range(4))
    prefixes = [SimpleNamespace(budget=10*i, complete_groups=groups[:i], record=lambda i=i: {'K': 10*i})
                for i in range(4)]
    compiler = SimpleNamespace(before_group=lambda _parent, held: prefixes[groups.index(held)])
    calls, captured = [], []
    def fit(prefix, *args, **kwargs):
        calls.append(prefix.budget)
        if prefix.budget == 10:
            raise ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage='C', detail='actual fit failed')
        def score(view, **kwargs):
            return {'joint': _evidence(view, reason='actual numerical failure' if prefix.budget == 0 else None)}
        return SimpleNamespace(score_experts=score, fit_receipts=())
    monkeypatch.setattr(ensemble_fit, 'fit_calibration_prefix', fit)
    monkeypatch.setattr(ensemble_views, 'calibration_prefix_identity', lambda prefix: str(prefix.budget))
    model = SimpleNamespace(scorer=SimpleNamespace(program=EnsembleProgram.equal_weights('actual', ('joint',))))
    recorder = SimpleNamespace(record_joint_forward_oof=lambda **row: captured.append(row))
    with pytest.raises(ExpectedSubjectError, match='complete original forward OOF'):
        _forward_joint_temperature(model, groups, (_programme(),), {}, 'cpu', prefix=object(),
            prefix_compiler=compiler, temperature_expert_ids=None, producer_recorder=recorder, producer_context={})
    assert calls == [0, 10, 20, 30]
    assert [row['row']['held_record_id'] for row in captured] == [g.view.record_id for g in groups]
    assert [row['row']['status'] for row in captured] == ['unavailable', 'unavailable', 'scored', 'scored']
    assert captured[1]['fitted'] is None and captured[1]['row']['fit_failure']['stage'] == 'C'
    assert all(row['row']['held_labels_used_for_fit'] is False for row in captured)


def test_failed_joint_replay_requires_original_permission_and_never_samples(monkeypatch):
    from research import joint_flat_chain
    from transfer import joint_flat_model

    def forbidden(*_args, **_kwargs):
        raise AssertionError('read-only replay called a random sampler')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(joint_flat_model, '_run_prefix', forbidden)
    monkeypatch.setattr(joint_flat_chain, 'run_joint_history_gibbs', forbidden)
    view = _view()
    original = _evidence(view, reason='actual numerical resolution failure')
    checked = []
    expert = SimpleNamespace(program=_programme(), predictor=SimpleNamespace(
        validate_history_permission=lambda v, cap: checked.append((v, cap))), score=forbidden,
        unavailable=lambda v, reason, kind: _evidence(v, reason=reason))
    model = SimpleNamespace(_experts=(expert,))
    capability = SimpleNamespace(digest='6'*64)
    compiler = SimpleNamespace(record_read_scope=lambda *args, **kwargs: capability)
    permission = {'joint': {'sha256': capability.digest, 'binding': {'conditioning_sample_support': [0, 4]}}}
    result = _replay_expert_scores(model, view, compiler=compiler, prefix=object(), captures={},
        saved={'joint': original.record()}, permissions=permission)
    assert result['joint'].record() == original.record() and len(checked) == 1
    with pytest.raises(ValueError, match='issued history permission'):
        _replay_expert_scores(model, view, compiler=compiler, prefix=object(), captures={},
            saved={'joint': original.record()}, permissions={})
    with pytest.raises(ValueError, match='no original captured P measure'):
        _replay_expert_scores(model, view, compiler=compiler, prefix=object(), captures={},
            saved={'joint': _evidence(view).record()}, permissions={})
    capture, calls = object(), []
    expert.score = lambda v, **kwargs: (calls.append(kwargs['captured_history']), _evidence(v))[1]
    result = _replay_expert_scores(model, view, compiler=compiler, prefix=object(), captures={'joint': capture},
        saved={'joint': _evidence(view).record()}, permissions=permission)
    assert calls == [capture] and result['joint'].probability_available


def test_all_unavailable_joint_programmes_keep_the_requested_expert_roster():
    failure = ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS, stage='temperature', detail='warmup')
    prefix = SimpleNamespace(record=lambda: {'K': 0})
    rows = _score_programs({'grouped_temperature': failure}, _view(), prefix=prefix, specs=(_programme(),))
    assert set(rows['grouped_temperature']) == {'joint'}
    assert not rows['grouped_temperature']['joint'].probability_available


def test_mixed_joint_replay_recomputes_raw_expert_before_saved_temperature_comparison():
    from dataclasses import replace

    view = _view()
    raw = _evidence(view)
    saved_temperature = replace(raw, log_probability=np.log([.4, .6]))
    expert = SimpleNamespace(program=SimpleNamespace(expert_id='other', family='continuous_ar'),
        score=lambda *_a, **_k: raw)
    result = _replay_expert_scores(SimpleNamespace(_experts=(expert,)), view, compiler=None,
        prefix=None, captures={}, saved={'other': saved_temperature.record()}, permissions={})
    assert result['other'].record() == raw.record()


def test_saved_joint_strategy_order_survives_sorted_json_keys():
    from research.contracts import semantic_sha256
    from research.joint_flat_replay import JointFlatPackageAdmission

    strategies = ['grouped_temperature', 'fixed']
    view = {'record_id': 'future'}
    record = {'strategy_order': strategies, 'programs': {
        'grouped_temperature': {'status': 'unavailable', 'failure': {'error_code': 'insufficient_calibration_groups'}},
        'fixed': {'status': 'completed'}}, 'prefix_input': {'complete_groups': []}, 'OOF': [],
        'fit_context': {}, 'future': [{'view': view, 'scores': {name: {} for name in strategies}}],
        'future_score_rows': [{'strategy': name, 'input': {'view': view, 'view_record_sha256': semantic_sha256(view)},
            'expert_evidence': {}} for name in strategies], 'inner_producers': []}
    admission = object.__new__(JointFlatPackageAdmission)
    admission.record = json.loads(json.dumps(record, sort_keys=True))
    admission._check_saved_rosters()
    admission.record['strategy_order'] = ['fixed', 'grouped_temperature']
    with pytest.raises(ValueError, match='ordered requested future roster'):
        admission._check_saved_rosters()


def test_joint_capture_references_must_belong_to_verified_package_members(tmp_path):
    from research.joint_flat_replay import _artifact, _check_member_references

    (tmp_path/'captured').mkdir()
    source, capture = tmp_path/'source.json', tmp_path/'captured/measure.npz'
    source.write_bytes(b'original source')
    capture.write_bytes(b'original capture')
    source_row, capture_row = _artifact(source, tmp_path), _artifact(capture, tmp_path)
    assert capture_row['path'] == 'captured/measure.npz'
    record = {'members': [source_row, capture_row], 'original_source_files': {'joint': source_row},
        'OOF': [], 'future': [{'captures': {'joint': {'capture': capture_row}}}]}
    _check_member_references(tmp_path, record)
    record['future'][0]['captures']['joint']['capture'] = {**capture_row, 'path': '../unverified.npz'}
    with pytest.raises(ValueError, match='outside the verified package members'):
        _check_member_references(tmp_path, record)


def test_joint_k0_dispatch_has_no_trial_target_or_target_arrays(monkeypatch):
    from transfer import ensemble_fit
    from transfer.ensemble_views import CalibrationPrefix

    prefix = CalibrationPrefix(0, None, {}, (), {'raw_scope': (0, 0), 'participant_key': 'p'},
        {'raw': {'used_labels': 0, 'requested_labels': 0, 'read_sample_support': None}, 'trial': {}})
    monkeypatch.setattr(CalibrationPrefix, 'trial_target', lambda *_args: pytest.fail('joint K0 called trial_target'))
    calls = []
    def fit(spec, data, records, target, source, receipts, inputs, **kwargs):
        calls.append((data, records, kwargs['source_only']))
        return ensemble_fit._FittedExpert(spec, None, None, '5'*64, None, 0., 'declared test-only unavailable')
    monkeypatch.setattr(ensemble_fit, '_fit_expert', fit)
    spec = _programme()
    result = ensemble_fit.fit_calibration_prefix(prefix, (spec,), EnsembleProgram.equal_weights('K0', ('joint',)))
    assert calls == [(None, (), True)] and result.participant_key == 'p'


def test_joint_pool_cannot_enter_nonchronological_group_leave_one_out(monkeypatch):
    from transfer import ensemble_fit
    from transfer.pool_calibration import POOL_STRATEGY

    monkeypatch.setattr(ensemble_fit, '_grouped_pool_calibration', lambda *_a, **_k: pytest.fail('old pool law entered'))
    with pytest.raises(ValueError, match='forward nested programme'):
        ensemble_fit.fit_calibration_prefix_programs(object(), (_programme(),),
            EnsembleProgram.equal_weights('joint', ('joint',)), strategies=(POOL_STRATEGY,))


def test_actual_joint_fit_digest_binds_source_role_and_original_c_authority():
    from research.contracts import semantic_sha256
    from transfer.ensemble_fit import _calibration_expert_fit_digest

    spec, authority, source = _programme(), {'original_C': 'actual-array-digest', 'K': 500}, '7'*64
    expected = semantic_sha256({'program': spec.record(), 'authorized_calibration': authority,
        'source_checkpoint_sha256': source, 'checkpoint_role': 'original_source_response_and_noise_parameters'})
    assert _calibration_expert_fit_digest(spec, authority, source) == expected
    assert _calibration_expert_fit_digest(spec, {**authority, 'K': 499}, source) != expected


def test_declared_source_roles_and_k0_schedule_keep_matched_scratch_distinct():
    from research.calibration_selection import _prefix_common_training_groups
    from research.prefix_inner_replay import _source_roles
    from transfer.recipes import TargetRecipe

    joint = _programme()
    source = CalibrationExpertProgram('NN', 'trial', TargetRecipe('source'), object())
    scratch = CalibrationExpertProgram('scratch', 'trial', TargetRecipe('scratch', initialization='scratch'), object())
    raw = CalibrationExpertProgram('raw', 'continuous_ar')
    assert [spec.source_checkpoint_role for spec in (joint, source, scratch, raw)] == [
        'original_source_response_and_noise_parameters', 'pretrained_weights', 'architecture_only', None]
    assert [spec.source_only_capable for spec in (joint, source, scratch, raw)] == [True, True, False, False]
    frozen = SimpleNamespace(digest='7'*64)
    roles = _source_roles((joint, source, scratch), {spec.expert_id: frozen for spec in (joint, source, scratch)})
    assert roles['scratch']['role'] == 'architecture_only' and roles['joint']['role'] != roles['NN']['role']
    def schedule(specs, strategy='fixed'):
        return _prefix_common_training_groups((SimpleNamespace(strategy=strategy, expert_programs=specs),))
    assert schedule((joint,)) == schedule((joint, source)) == 0
    assert schedule((joint, scratch)) == schedule((joint, raw)) == 1
    assert schedule((joint,), 'grouped_temperature') == 3
