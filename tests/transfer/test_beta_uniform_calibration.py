"""Exact latent enumeration and persistence attacks for the new probability law."""

import itertools
import json
import math
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from transfer.beta_uniform_calibration import (
    BetaCalibrationResolutionError,
    beta_uniform_from_oof,
    beta_uniform_log_probability,
    validate_beta_uniform_calibration,
    validate_beta_uniform_prior,
)


def forecasts(probabilities, sizes):
    return [{"row_id": f"held{i}", "candidate_count": size,
             "true_log_probability": None if p == 0 else math.log(float(p)),
             "forecast_identity": {"actual_producer": f"producer{i}", "past_rows": list(range(i))}}
            for i, (p, size) in enumerate(zip(probabilities, sizes, strict=True))]


def posterior(probabilities, sizes, *, prior):
    rows = forecasts(probabilities, sizes)
    return beta_uniform_from_oof(rows, prior=prior,
        requested_row_ids=[r["row_id"] for r in rows], lineage={"actual_prefix": "source-issued-software"})


def beta_integer(a, b):
    return Fraction(math.factorial(a - 1) * math.factorial(b - 1), math.factorial(a + b - 1))


def enumerated_moments(probabilities, sizes, a, b):
    evidence, first, second = Fraction(), Fraction(), Fraction()
    n = len(probabilities)
    for latent in itertools.product((0, 1), repeat=n):
        k = sum(latent)
        mass = math.prod((Fraction(1, size) if contaminated else probability)
                         for contaminated, probability, size in zip(latent, probabilities, sizes, strict=True))
        mass *= beta_integer(a + k, b + n - k) / beta_integer(a, b)
        evidence += mass
        first += mass * Fraction(a + k, a + b + n)
        second += mass * Fraction((a + k) * (a + k + 1), (a + b + n) * (a + b + n + 1))
    return evidence, first / evidence, second / evidence


@pytest.mark.parametrize("n", range(8))
def test_complete_posterior_matches_independent_latent_enumeration(n):
    probabilities = [Fraction(4, 5), Fraction(), Fraction(1, 3), Fraction(1, 7),
                     Fraction(2, 5), Fraction(1, 32), Fraction(9, 10)][:n]
    sizes = [2, 32, 3, 4, 2, 32, 2][:n]
    state = posterior(probabilities, sizes, prior={"a": 2, "b": 5})
    evidence, first, second = enumerated_moments(probabilities, sizes, 2, 5)
    actual_first = sum((0 if c["log_weight"] is None else math.exp(c["log_weight"]))
                       * c["a"] / (c["a"] + c["b"]) for c in state["components"])
    actual_second = sum((0 if c["log_weight"] is None else math.exp(c["log_weight"]))
                        * c["a"] * (c["a"] + 1) / ((c["a"] + c["b"]) * (c["a"] + c["b"] + 1))
                        for c in state["components"])
    assert math.exp(state["log_evidence"]) == pytest.approx(float(evidence), rel=5e-14)
    assert actual_first == pytest.approx(float(first), abs=2e-15)
    assert actual_second == pytest.approx(float(second), abs=2e-15)
    assert validate_beta_uniform_calibration(json.loads(json.dumps(state, allow_nan=False))) == state


def test_prior_once_sum_likelihood_and_uniform_noninformation():
    state = posterior([0, 0], [32, 32], prior={"a": 2, "b": 3})
    assert state["mean_epsilon_summary"] == pytest.approx(4 / 7)
    assert state["mean_epsilon_summary"] != pytest.approx(1 / 2)  # duplicated original prior
    state = posterior([0, 0], [32, 32], prior={"a": 1, "b": 1})
    assert state["mean_epsilon_summary"] == pytest.approx(3 / 4)
    assert state["mean_epsilon_summary"] != pytest.approx(2 / 3)  # mean instead of sum loglik
    uniform = posterior([Fraction(1, 2), Fraction(1, 7), Fraction(1, 32)], [2, 7, 32], prior={"a": 2, "b": 5})
    for point in (.1, .4, .8):
        density = sum((0 if c["log_weight"] is None else math.exp(c["log_weight"]))
                      * point ** (c["a"] - 1) * (1 - point) ** (c["b"] - 1)
                      / float(beta_integer(int(c["a"]), int(c["b"]))) for c in uniform["components"])
        assert density == pytest.approx(point * (1 - point) ** 4 / float(beta_integer(2, 5)), rel=2e-14)


@pytest.mark.parametrize("n", (0, 1, 2, 7))
@pytest.mark.parametrize("true_logp", (-1000., None))
def test_tiny_proper_clean_shape_survives_nonempty_update_and_standard_json(n, true_logp):
    rows = forecasts([0] * n, [32] * n)
    for row in rows:
        row["true_log_probability"] = true_logp
    state = beta_uniform_from_oof(rows, prior={"a": 1, "b": 1e-20},
        requested_row_ids=[r["row_id"] for r in rows], lineage={"prefix": "actual"})
    assert state["components"][-1]["b"] == 1e-20
    assert math.isfinite(state["log_mean_clean"])
    assert state["mean_epsilon_summary"] == 1.
    restored = validate_beta_uniform_calibration(json.loads(json.dumps(state, allow_nan=False)))
    np.testing.assert_array_equal(beta_uniform_log_probability([0, -np.inf], state),
                                  beta_uniform_log_probability([0, -np.inf], restored))
    if n:
        assert (state["components"][0]["log_weight"] is None) == (true_logp is None)


def test_true_zero_branch_encoding_and_full_state_tamper_rejection():
    state = posterior([0], [2], prior={"a": 1, "b": 1})
    assert state["components"][0]["log_weight"] is None
    for key in ("components", "log_evidence", "log_mean_clean", "mean_epsilon_summary"):
        changed = deepcopy(state)
        if key == "components":
            changed[key][0]["log_weight"] = -1000.
        else:
            changed[key] += .1
        with pytest.raises(ValueError, match="full Beta posterior"):
            validate_beta_uniform_calibration(changed)


def test_finite_unrepresentable_log_branch_is_refused_instead_of_becoming_true_zero():
    rows = forecasts([0, 0], [2, 2])
    for row in rows:
        row["true_log_probability"] = -1e308
    with pytest.raises(BetaCalibrationResolutionError, match="positive finite Beta mixture branch"):
        beta_uniform_from_oof(rows, prior={"a": 1, "b": 1},
            requested_row_ids=[row["row_id"] for row in rows], lineage={"prefix": "actual"})
    assert posterior([0, 0], [2, 2], prior={"a": 1, "b": 1})["components"][0]["log_weight"] is None


def test_tiny_uniform_mass_and_large_proper_shapes_never_require_float_mean_prediction():
    state = posterior([], [], prior={"a": 1e-300, "b": 1e300})
    assert state["mean_epsilon_summary"] == 0.
    assert math.isfinite(state["log_mean_epsilon"])
    logp = beta_uniform_log_probability([0., -np.inf], state)
    assert logp[1] == pytest.approx(math.log(1e-300) - math.log(1e300) - math.log(2))
    large = posterior([], [], prior={"a": 1e308, "b": 1e308})
    assert large["mean_epsilon_summary"] == pytest.approx(.5, abs=1e-15)
    np.testing.assert_allclose(beta_uniform_log_probability([0., -np.inf], large), np.log([.75, .25]), atol=1e-15)


@pytest.mark.parametrize("prior", (None, {}, {"a": 1}, {"a": True, "b": 1},
                                   {"a": 0, "b": 1}, {"a": 1, "b": math.inf}))
def test_prior_is_explicit_proper_and_never_defaulted(prior):
    with pytest.raises(ValueError, match="prior"):
        validate_beta_uniform_prior(prior)


def test_requested_roster_cannot_drop_duplicate_or_reorder_a_forecast():
    rows = forecasts([.2, .8], [2, 2])
    for names, actual in ((["held0", "held1"], rows[:1]), (["held0", "held0"], rows),
                          (["held0", "held1"], rows[::-1])):
        with pytest.raises(ValueError, match="roster"):
            beta_uniform_from_oof(actual, prior={"a": 1, "b": 1}, requested_row_ids=names, lineage={"prefix": "actual"})


def test_prequential_evidence_and_shared_parameter_joint_are_distinct():
    probabilities, sizes = [Fraction(), Fraction()], [2, 2]
    previous = posterior([], [], prior={"a": 1, "b": 1})
    sequential = 0.
    for n in range(2):
        sequential += float(beta_uniform_log_probability([0., -np.inf], previous)[1])
        previous = posterior(probabilities[:n + 1], sizes[:n + 1], prior={"a": 1, "b": 1})
    assert sequential == pytest.approx(previous["log_evidence"], abs=1e-15)
    assert math.exp(previous["log_evidence"]) == pytest.approx(1 / 12)
    assert math.exp(previous["log_evidence"]) != pytest.approx(1 / 16)
