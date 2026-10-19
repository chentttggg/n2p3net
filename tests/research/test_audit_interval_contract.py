"""Exact two-unit counterexample and a bounded-mean inequality oracle."""
import json

import numpy as np
import pytest
from scipy.stats import binom

from stats.hierarchical import paired_interval
from stats.transfer import _participant_rows, _summary


def test_two_unit_bootstrap_range_is_descriptive_not_nominal_coverage():
    # For IID continuous symmetric observations, the sample range contains
    # their population center iff signs differ: exactly 2*(1/2)*(1/2)=1/2.
    assert 2*.5*.5 == .5
    result = paired_interval(np.array([-.1, .2]), inference_scope='conditional_frozen_models',
        iterations=1000, seed=2, value_bounds=(-1., 1.))
    assert result.bootstrap_interval == (-.1, .2)
    assert result.interval_estimable is False
    assert result.interval_role == 'descriptive_resampling_interval'
    assert result.finite_sample_coverage_guaranteed is False
    assert result.bounded_mean_interval == (-1., 1.)
    assert result.population_value_bounds == (-1., 1.)
    json.dumps(result.record(), allow_nan=False)


def test_hoeffding_interval_uses_population_range_and_independent_unit_count():
    values = np.linspace(-.2, .4, 40)
    result = paired_interval(values, inference_scope='training_procedure',
        iterations=1000, seed=4, confidence_level=.9, value_bounds=(-1., 1.))
    radius = np.sqrt(2*np.log(20)/len(values))
    np.testing.assert_allclose(result.bounded_mean_interval, [values.mean()-radius, values.mean()+radius])
    assert result.bounded_mean_method == 'Hoeffding_independent_bounded_mean'
    assert result.sampling_unit == 'independent_training_replicate'
    without = paired_interval(values, inference_scope='training_procedure', iterations=1000, seed=4,
        confidence_level=.9)
    assert without.bounded_mean_interval is None
    np.testing.assert_array_equal(result.bootstrap_interval, without.bootstrap_interval)
    with pytest.raises(ValueError, match='population'):
        paired_interval(values, inference_scope='training_procedure', iterations=1000, seed=4,
            value_bounds=(-.1, .1))


def test_degenerate_empirical_distribution_does_not_certify_mean():
    # All observed outcomes can be equal while unobserved population outcomes
    # differ. A degenerate bootstrap cannot supply its own coverage guarantee.
    result = paired_interval(np.ones(10)*.2, inference_scope='conditional_frozen_models',
        iterations=1000, seed=3, value_bounds=(-1., 1.))
    assert result.interval_estimable is False
    assert result.bounded_mean_interval[0] < .2 < result.bounded_mean_interval[1]
    constant_population = paired_interval(np.zeros(2), inference_scope='conditional_frozen_models',
        iterations=1000, seed=3, value_bounds=(0., 0.))
    assert constant_population.bounded_mean_interval == (0., 0.)


@pytest.mark.parametrize('n', [2, 5, 10, 50])
def test_bounded_mean_interval_covers_by_exact_binomial_enumeration(n):
    # Enumerate the sampling law, not bootstrap replicas: every possible count
    # in n independent Bernoulli draws has its exact binomial probability.
    for probability in (.1, .5, .9):
        coverage = 0.
        for successes in range(n+1):
            values = np.r_[np.ones(successes), np.zeros(n-successes)]
            interval = paired_interval(values, inference_scope='conditional_frozen_models',
                iterations=1000, seed=1, value_bounds=(0., 1.)).bounded_mean_interval
            if interval[0] <= probability <= interval[1]:
                coverage += binom.pmf(successes, n, probability)
        assert coverage >= .95-1e-14


def test_missing_seed_sensitivity_does_not_change_requested_primary_estimand():
    rows = [
        {'study': 'S', 'participant_key': 'a', 'source_seed': 1, 'target_seed': 1,
         'status': 'complete_pair', 'difference': .2},
        {'study': 'S', 'participant_key': 'b', 'source_seed': 2, 'target_seed': 2,
         'status': 'complete_pair', 'difference': -.1},
    ]
    participants = _participant_rows(rows, ((1, 1), (2, 2)))
    report = _summary(participants, ('S',), iterations=1000, seed=1, contrast_bounds={'S': 1.})
    assert report['n_requested_participants'] == 2 and report['n_complete_participants'] == 0
    assert report['mean_difference'] is None
    assert report['by_study']['S']['replacement_decline_gate_passed'] is False
    sensitivity = report['available_seed_pair_sensitivity']['by_study']['S']
    assert sensitivity['mean_difference'] == pytest.approx(.05)
    assert sensitivity['available_seed_pair_counts'] == [1, 1]
    assert report['available_seed_pair_sensitivity']['missingness_bias_removed'] is False
