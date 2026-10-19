"""Single-K equivalence and an independent equal-K missing-seed counterexample."""
import numpy as np
import pytest

from stats.transfer import analyze_transfer_contrasts
from tests.support.comparison import comparison_identity

PAIRS = ((1, 10), (1, 20), (2, 10), (2, 20))


def _cell(study, person, budget, pair, arm, delta, available=True):
    return {
        'study': study, 'participant_key': f'authority:{person}', 'K': budget,
        'source_seed': pair[0], 'target_seed': pair[1], 'arm': arm,
        'auc': (.5+delta if arm == 'source' else .5) if available else None,
        'status': 'completed' if available else 'fit_failed',
        'calibration_rows_sha256': 'a'*64, 'test_rows_sha256': 'b'*64,
        'input_sha256': 'c'*64, 'test_labels_sha256': 'd'*64,
        'comparison_identity': comparison_identity(person=f'{study}:{person}'),
    }


def _analyze(cells, budgets, studies):
    return analyze_transfer_contrasts(cells, {'transfer': {'source': 1., 'scratch': -1.}},
        expected_studies=studies, source_seeds=(1, 2), target_seeds=(10, 20),
        standard_budgets=budgets, iterations=1000)['groups'][0]['contrasts']['transfer']


def test_single_k_policy_sensitivity_equals_exact_k_with_one_failed_seed():
    cells = [_cell(study, person, 200, pair, arm, delta,
        available=not (person == 'p1' and pair == PAIRS[0] and arm == 'source'))
        for study, person, delta in (('A', 'p1', .1), ('A', 'p2', .3), ('B', 'p3', -.2))
        for pair in PAIRS for arm in ('source', 'scratch')]
    result = _analyze(cells, (200,), ('A', 'B'))
    exact, policy = result['exact_K']['200'], result['budget_policy']
    for study in ('A', 'B'):
        assert policy['available_seed_pair_sensitivity']['by_study'][study] == (
            exact['available_seed_pair_sensitivity']['by_study'][study])
    available = policy['available_seed_pair_sensitivity']['by_study']['A']
    assert available['n_with_any_complete_seed_pair'] == 2
    assert available['mean_difference'] == pytest.approx(.2)
    assert available['available_seed_pair_counts'] == [3, 4]
    assert exact['n_requested_participants'] == policy['n_requested_participants'] == 3
    assert exact['n_complete_participants'] == policy['n_complete_participants'] == 2
    assert exact['mean_difference'] == policy['mean_difference']
    p1 = policy['participant_rows'][0]
    assert p1['difference'] is None and p1['status'] == 'incomplete_budget_policy'


def test_policy_sensitivity_averages_available_seed_means_per_k_then_weights_k_equally():
    cells = []
    for person in ('p1', 'p2', 'p3'):
        for budget in (200, 800):
            delta = ({200: .4, 800: -.2}[budget] if person == 'p1'
                     else 0. if person == 'p2' else .2)
            for index, pair in enumerate(PAIRS):
                available = (index == 0 if budget == 200 else index > 0) if person == 'p1' else (
                    person != 'p3' or budget == 200)
                for arm in ('source', 'scratch'):
                    cells.append(_cell('A', person, budget, pair, arm, delta, available))
    result = _analyze(cells, (200, 800), ('A',))
    policy = result['budget_policy']
    people = {row['participant_key']: row for row in policy['participant_rows']}
    p1 = people['authority:p1']
    assert p1['available_seed_pair_mean_difference'] == pytest.approx((.4-.2)/2)
    assert p1['available_seed_pair_mean_difference'] != pytest.approx((.4-3*.2)/4)
    assert p1['available_seed_pairs'] == 4
    assert p1['available_seed_pairs_by_actual_K'] == {'200': 1, '800': 3}
    assert p1['available_seed_pair_actual_Ks'] == [200, 800]
    p3 = people['authority:p3']
    assert p3['available_seed_pair_actual_Ks'] == [200]
    assert p3['available_seed_pairs_by_actual_K'] == {'200': 4, '800': 0}
    assert p3['available_seed_pair_mean_difference'] == pytest.approx(.2)
    sensitivity = policy['available_seed_pair_sensitivity']
    assert sensitivity['by_study']['A']['n_with_any_complete_seed_pair'] == 3
    assert sensitivity['by_study']['A']['mean_difference'] == pytest.approx((.1+0.+.2)/3)
    assert sensitivity['by_study']['A']['available_seed_pair_counts'] == [4, 8, 4]
    assert sensitivity['missingness_bias_removed'] is False
    assert 'equal_actual_K' in sensitivity['estimand']
    assert policy['n_requested_participants'] == 3 and policy['n_complete_participants'] == 1
    assert policy['mean_difference'] == 0.
    assert policy['requested_seed_pairs'] == 24 and policy['failed_or_missing_seed_pairs'] == 8


def test_legal_nonexact_zero_sum_contrast_uses_its_actual_auc_range_for_all_bounds():
    negative = -1.0000000001
    assert np.isclose(1.+negative, 0.) and 1.+negative != 0.
    cells = []
    for person in ('p1', 'p2', 'p3'):
        for pair in PAIRS:
            for arm in ('source', 'scratch'):
                row = _cell('A', person, 200, pair, arm, 0., available=person != 'p3')
                if person != 'p3':
                    row['auc'] = 0. if arm == 'source' else 1.
                cells.append(row)
    report = analyze_transfer_contrasts(cells,
        {'transfer': {'source': 1., 'scratch': negative}}, expected_studies=('A',),
        source_seeds=(1, 2), target_seeds=(10, 20), standard_budgets=(200,),
        iterations=1000)['groups'][0]['contrasts']['transfer']
    panel = report['exact_K']['200']
    study = panel['by_study']['A']
    assert study['contrast_lower_bound'] == negative
    assert study['contrast_upper_bound'] == 1.
    assert study['missing_result_bounds'] == [negative, (2*negative+1.)/3]
    assert panel['missing_result_bounds']['interval'] == study['missing_result_bounds']
    interval = study['paired_interval']
    assert interval['population_value_bounds'] == [negative, 1.]
    expected_upper = negative+(1.-negative)*np.sqrt(np.log(40.)/4.)
    assert interval['bounded_mean_interval'][0] == negative
    assert interval['bounded_mean_interval'][1] == pytest.approx(expected_upper)
    assert report['budget_policy']['by_study']['A']['missing_result_bounds'] == study['missing_result_bounds']
