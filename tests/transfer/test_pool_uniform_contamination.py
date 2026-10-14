"""Declared pool-level contamination, with independent convex-risk oracles."""

import importlib.util

import numpy as np
import pytest
from scipy.special import logsumexp


def owner():
    assert importlib.util.find_spec("transfer.pool_calibration") is not None, "pool calibration owner is missing"
    from transfer import pool_calibration
    return pool_calibration


def test_true_zero_has_uniform_endpoint_and_zero_keeps_original_log_bytes():
    module = owner()
    log_probability = np.array([[0., -np.inf], [0., -np.inf]])
    fitted = module.uniform_contamination_from_oof(log_probability, np.array([1, 1]))
    assert fitted["epsilon"] == 1.
    assert fitted["before_nll"] == "infinity"
    assert fitted["after_nll"] == pytest.approx(np.log(2.))
    np.testing.assert_array_equal(module.uniform_contamination_log_probability(log_probability, 0.), log_probability)
    np.testing.assert_array_equal(module.uniform_contamination_log_probability(log_probability, 1.), np.full((2, 2), -np.log(2.)))


def test_true_component_flat_keeps_zero_even_when_full_rows_are_not_uniform():
    module = owner()
    probability = np.array([[.25, .7, .03, .02], [.1, .25, .6, .05]])
    fitted = module.uniform_contamination_from_oof(np.log(probability), np.array([0, 1]))
    assert fitted["epsilon"] == 0. and fitted["optimum"] == "flat_smallest_epsilon"


def test_interior_optimizer_matches_independent_scalar_solution_and_gradient():
    module = owner()
    probability = np.array([[1., 0.], [0., 1.], [0., 1.]])
    with np.errstate(divide="ignore"):
        logs = np.log(probability)
    fitted = module.uniform_contamination_from_oof(logs, np.array([1, 1, 1]))
    assert fitted["epsilon"] == pytest.approx(2 / 3, abs=5e-15)
    epsilon = fitted["epsilon"]
    truth = probability[np.arange(3), [1, 1, 1]]
    numerator = truth - .5
    denominator = (1 - epsilon) * truth + epsilon * .5
    assert abs(np.mean(numerator / denominator)) < 1e-13
    adjusted = module.uniform_contamination_log_probability(logs, epsilon)
    assert -adjusted[np.arange(3), [1, 1, 1]].mean() == pytest.approx(fitted["after_nll"])
    assert np.all(-adjusted <= np.log(2 / epsilon) + 1e-14)


def test_finite_underflow_log_probability_is_not_true_zero():
    module = owner()
    logs = np.array([[-1000., 0.], [np.log(.9), np.log(.1)]])
    fitted = module.uniform_contamination_from_oof(logs, np.array([0, 0]))
    assert fitted["before_nll"] == pytest.approx((1000. - np.log(.9)) / 2)
    assert fitted["epsilon"] > 0
    adjusted = module.uniform_contamination_log_probability(logs, np.nextafter(0., 1.))
    assert np.isfinite(adjusted).all()
    assert adjusted[0, 0] > -1000.


def test_positive_temperature_and_pool_contamination_have_distinct_order():
    module = owner()
    first, second = np.array([2., 0.]), np.array([0., 1.])
    logs = np.vstack((first - logsumexp(first), second - logsumexp(second)))
    pool = logsumexp(np.log([.3, .7])[:, None] + logs, axis=0)
    adjusted = module.uniform_contamination_log_probability(pool, .2)
    np.testing.assert_allclose(np.exp(adjusted), .8 * np.exp(pool) + .1, rtol=0, atol=1e-15)
    assert np.argmax(adjusted) == np.argmax(pool)


def test_each_expert_learning_is_a_different_objective_from_the_complete_pool():
    module = owner()
    first = np.array([[.9, .1], [.01, .99]])
    second = np.array([[.99, .01], [.1, .9]])
    target = np.array([0, 0])
    pool = .5 * (first + second)
    whole = module.uniform_contamination_from_oof(np.log(pool), target)
    own = [module.uniform_contamination_from_oof(np.log(probability), target) for probability in (first, second)]
    assert whole["epsilon"] == own[0]["epsilon"] == 1.
    separately_calibrated = .5 * sum(np.exp(module.uniform_contamination_log_probability(np.log(probability),
        learned["epsilon"])) for probability, learned in zip((first, second), own, strict=True))
    np.testing.assert_allclose(separately_calibrated[:, 0], [.528125, .4770408163265306], rtol=0, atol=1e-14)
    assert not np.allclose(separately_calibrated, .5)


def test_actual_pool_scorer_keeps_failure_tie_and_the_exact_programme():
    module = owner()
    from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
    from transfer.ensemble import EnsembleProgram, EnsembleScorer

    program = EnsembleProgram("pool", ("a", "b"), (.3, .7))
    state = {"schema": module.POOL_CALIBRATION_SCHEMA, "program_sha256": program.digest(),
             "candidate_count": 2, "epsilon": 1.}

    def evidence(name, probability, failure=None):
        identity = CandidateExpertIdentity(name, "a" * 64, "b" * 64, None, "c" * 64, "software", False)
        return CandidateEvidence(("q0", "q1"), probability, "participant", "decision", "d" * 64,
            (identity,), 1, (1, 1), (1, 1), failure_reason=failure,
            failure_kind="not_available" if failure else None)

    scores = {name: evidence(name, np.log([.9, .1])) for name in program.expert_ids}
    scorer = EnsembleScorer(program, state)
    assert scorer.score(scores).tie
    scores["b"] = evidence("b", None, "original unavailable expert")
    assert not scorer.score(scores).probability_available
    with pytest.raises(ValueError, match="exact expert roster and weights"):
        EnsembleScorer(EnsembleProgram("pool", ("a", "b"), (.5, .5)), state)
    near = np.array([0., -1e-10])
    near -= logsumexp(near)
    scores = {name: evidence(name, near) for name in program.expert_ids}
    assert not EnsembleScorer(program).score(scores).tie
    assert EnsembleScorer(program, {**state, "epsilon": .99999}).score(scores).tie


@pytest.mark.parametrize("logs,targets", [(np.array([[0., -np.inf]]), np.array([0, 1])),
                                           (np.array([[0., 0.]]), np.array([0])),
                                           (np.array([[np.nan, 0.]]), np.array([0]))])
def test_learning_requires_complete_normalized_requested_probability_rows(logs, targets):
    with pytest.raises(ValueError):
        owner().uniform_contamination_from_oof(logs, targets)
