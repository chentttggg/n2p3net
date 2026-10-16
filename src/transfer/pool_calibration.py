"""One declared uniform-contamination law over an actual complete pool.

The scalar learner consumes complete OOF decision probabilities with equal
requested-selection mass. Source/C authority and numerical producers belong to
the fitted ensemble; an array or a declared role cannot grant that authority.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

POOL_CALIBRATION_SCHEMA = "n2p3_pool_uniform_contamination/1"
POOL_OOF_SCHEMA = "n2p3_complete_pool_probability_oof/1"
POOL_STRATEGY = "grouped_uniform_contamination"
TEMPERATURE_POOL_STRATEGY = "grouped_temperature_uniform_contamination"
POOL_STRATEGIES = frozenset({POOL_STRATEGY, TEMPERATURE_POOL_STRATEGY})
FORWARD_BETA_STRATEGY = "forward_beta_uniform"
POOL_RISK = "equal_requested_complete_selection/1"


def _log_probability(value):
    values = np.asarray(value, dtype=np.float64)
    if (values.ndim not in {1, 2} or values.shape[-1] < 2 or np.isnan(values).any()
            or np.isposinf(values).any() or not np.allclose(logsumexp(values, axis=-1), 0., rtol=0, atol=1e-10)):
        raise ValueError("pool probabilities must be normalized on the complete candidate vocabulary")
    return values


def uniform_contamination_log_probability(log_probability, epsilon):
    """Apply the same stable pool-after-arithmetic-mixture operation everywhere."""
    values = _log_probability(log_probability)
    return _contaminate(values, epsilon)


def _contaminate(values, epsilon):
    """Admitted normalized operands; the learner checks its array only once."""
    if isinstance(epsilon, bool) or not np.isscalar(epsilon) or not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("pool contamination requires a frozen epsilon in [0,1]")
    if epsilon == 0:
        return values.copy()
    uniform = -math.log(values.shape[-1])
    if epsilon == 1:
        return np.full(values.shape, uniform)
    return np.logaddexp(values + math.log1p(-epsilon), math.log(epsilon) + uniform)


def _risk(log_truth):
    if np.isneginf(log_truth).any():
        return "infinity"
    return -math.fsum(float(value) / len(log_truth) for value in log_truth)


def uniform_contamination_from_oof(log_probability, targets):
    """Minimize the convex complete-row NLL; exact flat objectives choose zero.

    Positive finite log probabilities remain finite even when exp(log p) would
    underflow. The signed derivative is normalized before exponentiation; this
    normalization changes neither its sign nor its single interior zero.
    """
    values = _log_probability(log_probability)
    targets = np.asarray(targets)
    if (values.ndim != 2 or not len(values) or targets.shape != (len(values),)
            or targets.dtype.kind not in "iu" or np.any(targets < 0) or np.any(targets >= values.shape[1])):
        raise ValueError("the learner requires every requested OOF row and its authorized target")
    truth = values[np.arange(len(values)), targets]
    uniform = -math.log(values.shape[1])
    signs = np.sign(truth - uniform)
    active = signs != 0
    logarithms = np.full(len(truth), -np.inf)
    with np.errstate(divide="ignore"):
        logarithms[active] = uniform + np.log(np.abs(np.expm1(truth[active] - uniform)))

    def derivative(epsilon):
        if epsilon == 0 and np.isneginf(truth).any():
            return -1.
        denominator = _contaminate(values, epsilon)[np.arange(len(values)), targets]
        terms = logarithms - denominator
        positive = float(logsumexp(terms[signs > 0]))
        negative = float(logsumexp(terms[signs < 0]))
        anchor = max(positive, negative)
        if anchor == -np.inf:
            return 0.
        return math.exp(positive - anchor) - math.exp(negative - anchor)

    if not active.any():
        epsilon, optimum = 0., "flat_smallest_epsilon"
    elif derivative(0.) >= 0:
        epsilon, optimum = 0., "zero_boundary"
    elif derivative(1.) <= 0:
        epsilon, optimum = 1., "uniform_boundary"
    else:
        epsilon = float(brentq(derivative, 0., 1., xtol=np.nextafter(0., 1.),
                              rtol=4 * np.finfo(float).eps, maxiter=2048))
        optimum = "unique_interior"
    adjusted = _contaminate(values, epsilon)
    return {"schema": POOL_CALIBRATION_SCHEMA, "epsilon": epsilon, "optimum": optimum,
        "risk_recipe": POOL_RISK, "requested_rows": len(values), "row_masses": [1.] * len(values),
        "candidate_count": values.shape[1], "before_nll": _risk(truth),
        "after_nll": _risk(adjusted[np.arange(len(values)), targets]),
        "positive_epsilon_log_probability_floor": None if epsilon == 0 else math.log(epsilon) + uniform,
        "learner": "convex_nll_signed_log_derivative_brentq/1", "flat_tie": "smallest_epsilon_zero",
        "solver": {"xtol": float(np.nextafter(0., 1.)), "rtol": float(4 * np.finfo(float).eps), "maxiter": 2048}}
