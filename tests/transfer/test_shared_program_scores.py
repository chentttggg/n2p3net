import copy
from dataclasses import replace

import numpy as np
import pytest

from research.execution import ExpectedSubjectError, SubjectFailureCode
from tests.support.ensemble_fixtures import programs, selections
from transfer.ensemble import EnsembleProgram, EnsembleScorer
from transfer.ensemble_fit import FittedEnsemble, _FittedExpert, fit_calibration_ensemble


def fitted_programs():
    groups = selections()
    spec = programs()[1]
    fixed = fit_calibration_ensemble(groups[:3], (spec,), EnsembleProgram.equal_weights('one', (spec.expert_id,)))
    temperatures = {'strategy': 'grouped_temperature', 'temperatures': {spec.expert_id: 2.},
                    'temperature_bounds': [.05, 20.]}
    grouped = FittedEnsemble(fixed.scorer, fixed._experts, fixed.fit_receipts, temperatures)
    return groups, {'fixed': fixed, 'grouped_temperature': grouped}


def records(scored):
    return {name: {key: value.record() for key, value in scores.items()} for name, scores in scored.items()}


def test_shared_scoring_matches_both_programs_and_reloaded_computes_independently(tmp_path, monkeypatch):
    groups, fitted = fitted_programs()
    expected = {name: value.score_experts(groups[3].view) for name, value in fitted.items()}
    path = FittedEnsemble.save_programs(tmp_path / 'package', fitted, participant_key=fitted['fixed'].participant_key)
    replay = FittedEnsemble.load_programs(path)
    calls = []
    original = _FittedExpert.score

    def counted(self, view, **kwargs):
        calls.append((id(self), view.raw_input_digest))
        return original(self, view, **kwargs)

    monkeypatch.setattr(_FittedExpert, 'score', counted)
    actual = FittedEnsemble.score_programs(fitted, groups[3].view)
    restored = FittedEnsemble.score_programs(replay, groups[3].view)
    assert records(actual) == records(restored) == records(expected)
    assert len(calls) == 2 and calls[0][0] != calls[1][0]
    changed = replace(groups[3].view, samples=groups[3].view.samples * 1.01)
    assert records(FittedEnsemble.score_programs(fitted, changed)) == {
        name: {key: item.record() for key, item in value.score_experts(changed).items()}
        for name, value in fitted.items()}


def test_different_actual_expert_state_or_fit_authority_rejected_before_scoring(monkeypatch):
    groups, fitted = fitted_programs()
    grouped = fitted['grouped_temperature']
    other = FittedEnsemble(grouped.scorer, copy.deepcopy(grouped._experts), grouped.fit_receipts,
                           grouped.temperature_record)
    # Identical declarations and digests do not authorize sharing distinct state.
    with pytest.raises(ValueError, match='share actual'):
        FittedEnsemble.score_programs({'fixed': fitted['fixed'], 'grouped_temperature': other}, groups[3].view)
    receipt = dict(grouped.fit_receipts[0])
    receipt['record_id'] = 'another-authorized-fit'
    other = FittedEnsemble(grouped.scorer, grouped._experts, (receipt, *grouped.fit_receipts[1:]),
                           grouped.temperature_record)
    with pytest.raises(ValueError, match='fit authority'):
        FittedEnsemble.score_programs({'fixed': fitted['fixed'], 'grouped_temperature': other}, groups[3].view)


def test_current_temperature_contract_changes_transform_and_bad_name_rejected():
    groups, fitted = fitted_programs()
    old = fitted['grouped_temperature']
    temperature = {**old.temperature_record, 'temperatures': {'toeplitz': 3.}}
    changed = FittedEnsemble(old.scorer, old._experts, old.fit_receipts, temperature)
    actual = FittedEnsemble.score_programs({'fixed': fitted['fixed'], 'grouped_temperature': changed}, groups[3].view)
    assert {k:v.record() for k,v in actual['grouped_temperature'].items()} == {
        k:v.record() for k,v in changed.score_experts(groups[3].view).items()}
    with pytest.raises(ValueError, match='actual program'):
        FittedEnsemble.score_programs({'fixed': changed}, groups[3].view)


def test_temperature_subset_keeps_control_evidence_unchanged():
    groups, fitted = fitted_programs()
    first = fitted['fixed']._experts[0]
    control = replace(first, program=replace(first.program, expert_id='control'))
    roster = EnsembleScorer(EnsembleProgram.equal_weights('with-control', ('toeplitz','control')))
    fixed = FittedEnsemble(roster, (first,control), fitted['fixed'].fit_receipts)
    grouped = FittedEnsemble(roster, fixed._experts, fixed.fit_receipts,
        {'strategy':'grouped_temperature','temperatures':{'toeplitz':2.,'control':1.},
         'temperature_bounds':[.05,20.],'temperature_expert_ids':['toeplitz'],
         'fixed_control_expert_ids':['control'],'OOF_groups':[{'expert_fit_digests':{'toeplitz':first.fit_digest}}]})
    actual = FittedEnsemble.score_programs({'fixed':fixed,'grouped_temperature':grouped}, groups[3].view)
    assert actual['fixed']['control'].record() == actual['grouped_temperature']['control'].record()
    assert records(actual) == {'fixed':{k:v.record() for k,v in fixed.score_experts(groups[3].view).items()},
                              'grouped_temperature':{k:v.record() for k,v in grouped.score_experts(groups[3].view).items()}}


@pytest.mark.parametrize('change', ['participant', 'overlap', 'record_reuse', 'vocabulary'])
def test_current_view_gates_are_preserved(change):
    groups, fitted = fitted_programs()
    view = groups[3].view
    if change == 'participant':
        view = replace(view, participant_key='another-physical-participant')
    elif change == 'overlap':
        view = replace(view, raw_scope=groups[0].view.raw_scope, available_sample=groups[0].view.available_sample)
    elif change == 'record_reuse':
        view = groups[0].view
    else:
        view = replace(view, decision=replace(view.decision, candidate_vocabulary=('b','a')))
    with pytest.raises(ValueError):
        fitted['fixed'].score_experts(view)
    with pytest.raises(ValueError):
        FittedEnsemble.score_programs(fitted, view)


def test_missing_observation_unavailable_strategy_and_failed_roster_are_retained():
    groups, fitted = fitted_programs()
    missing = replace(groups[3].view, samples=None, unavailable_reason='unobserved raw')
    actual = FittedEnsemble.score_programs(fitted, missing)
    assert all(not value.probability_available for scores in actual.values() for value in scores.values())
    assert records(actual) == {name:{k:v.record() for k,v in value.score_experts(missing).items()}
                              for name,value in fitted.items()}
    unavailable = ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
                                       stage='temperature', detail='actual insufficient C groups')
    result = FittedEnsemble.score_programs({'fixed':fitted['fixed'], 'grouped_temperature':unavailable}, groups[3].view)
    assert result['grouped_temperature'] is unavailable
    assert FittedEnsemble.score_programs({'fixed':unavailable}, groups[3].view) == {'fixed':unavailable}
    old = fitted['fixed']._experts[0]
    failed = replace(old, predictor=None, model_digest=None, unavailable_reason='legitimate fitted expert failure',
                     unavailable_kind='fit_failure')
    value = FittedEnsemble(EnsembleScorer(EnsembleProgram.equal_weights('one', ('toeplitz',))),
                           (failed,), fitted['fixed'].fit_receipts)
    assert not FittedEnsemble.score_programs({'fixed':value}, groups[3].view)['fixed']['toeplitz'].probability_available


def test_actual_zero_signal_tie_keeps_original_probability_and_winner_record():
    groups, fitted = fitted_programs()
    view = replace(groups[3].view, samples=np.zeros_like(groups[3].view.samples))
    shared = FittedEnsemble.score_programs(fitted, view)
    for name, model in fitted.items():
        evidence = shared[name]['toeplitz']
        assert evidence.log_probability[0] == evidence.log_probability[1]
        assert model.scorer.score(shared[name]).record() == model.score(view).record()


def test_prefix_capture_and_future_reload_each_score_shared_state_once(tmp_path, monkeypatch):
    import json
    from collections import Counter

    from experiments.run_candidate_ensemble_study import run_study
    from tests.support.independent_prefix import independent_prefix_case
    from transfer.ensemble_views import prepare_bnci_independent_prefixes

    _, _, raw_path, authorization, specs, original_views = independent_prefix_case(tmp_path)
    trial = specs[1]
    views = prepare_bnci_independent_prefixes(raw_path, source_sha256=original_views.ledger['source_sha256'],
        budgets=(400,), profiles=(trial.profile,), authorization_preprocessing=authorization)
    mixture = EnsembleProgram.equal_weights('trial', (trial.expert_id,))
    plan = {'schema': 'n2p3_candidate_ensemble_study/1', 'evidence_scope': 'software_correctness',
        'evaluation_design': 'independent_prefix', 'seed_pairs': [[11, 21]],
        'records': [{'path': str(raw_path), 'sha256': original_views.ledger['source_sha256']}],
        'budgets': [400], 'experts': [trial.record()], 'mixtures': [mixture.record()], 'baseline_mixture': 'trial',
        'authorization_preprocessing': authorization.record(), 'observation_response_seconds': 308 / 256,
        'pre_roll_seconds': .2, 'device': 'cpu', 'workers': 1, 'cpu_threads_per_worker': 1,
        'source_checkpoints': {}, 'source_seed_catalog': [], 'calibration_strategies': ['fixed', 'grouped_temperature']}
    path = tmp_path / 'prefix-plan.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    calls = []
    original = _FittedExpert.score

    def counted(self, current, **kwargs):
        calls.append((self.fit_digest, id(self), current.record_id))
        return original(self, current, **kwargs)

    monkeypatch.setattr(_FittedExpert, 'score', counted)
    report = run_study(path, tmp_path / 'prefix-run', execution='inline')
    assert report['status'] == 'completed'
    predictions = report['participants'][0]['predictions']
    assert [row['calibration_strategy'] for row in predictions] == (
        ['fixed'] * len(views.decisions) + ['grouped_temperature'] * len(views.decisions))
    assert all(row['saved_replay_exact'] for row in predictions)
    package = next((tmp_path / 'prefix-run').rglob('prefix_inner_replay.json')).parent
    receipt = json.loads((package / 'prefix_inner_replay.json').read_bytes())['record']
    assert len(receipt['future_score_rows']) == len(predictions) == 2 * len(views.decisions)
    final_digest = receipt['programs']['fixed']['model_identity']['experts'][0]['fit_digest']
    final_calls = [item for item in calls if item[0] == final_digest]
    assert len(final_calls) == 2 * len(views.decisions)  # Original and loaded groups independently compute.
    for view in views.decisions:
        current = [item for item in final_calls if item[2] == view.record_id]
        assert len(current) == 2 and current[0][1] != current[1][1]
    assert len({fit for fit, _, _ in calls if fit != final_digest}) == 3
    # Inner single-fixed models still compute once in fit and once in replay.
    assert set(Counter(fit for fit, _, _ in calls if fit != final_digest).values()) == {2}
