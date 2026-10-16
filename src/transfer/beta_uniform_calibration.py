"""Proper uniform-contamination posterior over predictable selection forecasts.

This scalar owner grants no EEG, prefix, producer or target-label authority.
Its callers admit the complete original forecast roster and authorized labels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp

from research.contracts import semantic_sha256
from transfer.noise_model import freeze_metadata

BETA_UNIFORM_SCHEMA = "n2p3_forward_beta_uniform_calibration/1"
BETA_UNIFORM_OOF_SCHEMA = "n2p3_joint_flat_forward_beta_probability_OOF/1"


class BetaCalibrationResolutionError(ValueError):
    """A positive mathematical branch cannot be represented losslessly."""


def validate_beta_uniform_prior(prior):
    """Normalize an explicit proper prior; there is deliberately no default."""
    if not isinstance(prior, Mapping) or set(prior) != {"a", "b"}:
        raise ValueError("forward Beta calibration requires explicit prior {a,b}")
    result = {}
    for name in ("a", "b"):
        value = prior[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("Beta prior parameters must be positive finite real scalars")
        result[name] = float(value)
    return result


def _log_mass(value):
    return -math.inf if value is None else float(value)


def _saved_log_mass(value):
    return None if value == -math.inf else float(value)


def beta_uniform_from_oof(forecasts, *, prior, requested_row_ids, lineage):
    """Reconstruct the whole posterior, using each original decision once.

    A forecast contains row_id, candidate_count, true_log_probability (null
    means a true zero), and forecast_identity. The enclosing source owner must
    derive these from admitted full probability rows and original C labels.
    """
    prior = validate_beta_uniform_prior(prior)
    names, forecasts = tuple(requested_row_ids), tuple(forecasts)
    if (any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or len(forecasts) != len(names)):
        raise ValueError("Beta calibration requires the complete unique ordered forecast roster")
    if not isinstance(lineage, Mapping) or not lineage:
        raise ValueError("Beta posterior requires its original prefix and base producer lineage")
    a, b = prior["a"], prior["b"]
    weights, evidence, rows = np.array([0.]), 0., []
    for n, (name, forecast) in enumerate(zip(names, forecasts, strict=True)):
        if (not isinstance(forecast, Mapping)
                or set(forecast) != {"row_id", "candidate_count", "true_log_probability", "forecast_identity"}
                or forecast["row_id"] != name):
            raise ValueError("Beta forecast order or likelihood fields differ from requested roster")
        count = forecast["candidate_count"]
        raw = forecast["true_log_probability"]
        identity = forecast["forecast_identity"]
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 2:
            raise ValueError("Beta forecast needs its actual complete candidate count")
        if (raw is not None and (isinstance(raw, bool) or not isinstance(raw, Real)
                                 or not math.isfinite(raw) or raw > 0)
                or not isinstance(identity, Mapping) or not identity):
            raise ValueError("Beta likelihood must retain finite log probability or a true zero and producer identity")
        truth = _log_mass(raw)
        log_total = float(np.logaddexp(math.log(a + n), math.log(b)))
        updated = np.full(n + 2, -np.inf)
        for k, weight in enumerate(weights):
            clean = float(weight) + truth + math.log(b + (n - k)) - log_total
            uniform = float(weight) - math.log(count) + math.log(a + k) - log_total
            if (math.isfinite(weight) and math.isfinite(truth) and not math.isfinite(clean)
                    or math.isfinite(weight) and not math.isfinite(uniform)):
                raise BetaCalibrationResolutionError("positive finite Beta mixture branch exceeds float64 log range")
            updated[k] = np.logaddexp(updated[k], clean)
            updated[k + 1] = np.logaddexp(updated[k + 1], uniform)
        step = float(logsumexp(updated))
        if not math.isfinite(step):
            raise ValueError("proper Beta calibration must have positive finite evidence")
        evidence += step
        if not math.isfinite(evidence):
            raise BetaCalibrationResolutionError("Beta cumulative evidence exceeds float64 log range")
        weights = updated - step
        rows.append({"row_id": name, "candidate_count": int(count),
                     "true_log_probability": _saved_log_mass(truth),
                     "forecast_identity": freeze_metadata(identity)})
    n = len(rows)
    log_total = float(np.logaddexp(math.log(a + n), math.log(b)))
    uniform = float(logsumexp([weight + math.log(a + k) - log_total
                               for k, weight in enumerate(weights)]))
    clean = float(logsumexp([weight + math.log(b + (n - k)) - log_total
                             for k, weight in enumerate(weights)]))
    # Normalize the two separately summed positive masses together. This also
    # removes log(shape)-log(total) roundoff for very large proper shapes.
    total_mass = float(np.logaddexp(clean, uniform))
    clean, uniform = clean - total_mass, uniform - total_mass
    return {"schema": BETA_UNIFORM_SCHEMA, "prior": prior,
            "requested_row_ids": list(names), "forecast_likelihoods": rows,
            "lineage": freeze_metadata(lineage), "n_decisions": n,
            "components": [{"a": a + k, "b": b + (n - k),
                            "log_weight": _saved_log_mass(weight)}
                           for k, weight in enumerate(weights)],
            "log_evidence": evidence, "log_mean_epsilon": uniform, "log_mean_clean": clean,
            "mean_epsilon_summary": math.exp(uniform),
            "zero_log_mass_encoding": "null_is_negative_infinity",
            "likelihood_recipe": "ordered_predictable_selection_sum_log_probability/1",
            "learner": "normalized_positive_log_beta_mixture/1",
            "prediction_scope": "single_future_marginal_shared_epsilon_parameter"}


def validate_beta_uniform_calibration(state):
    """Rebuild all scalar state; caller still verifies original rows and truth."""
    if not isinstance(state, Mapping) or state.get("schema") != BETA_UNIFORM_SCHEMA:
        raise ValueError("unsupported forward Beta calibration state")
    try:
        expected = beta_uniform_from_oof(state["forecast_likelihoods"], prior=state["prior"],
            requested_row_ids=state["requested_row_ids"], lineage=state["lineage"])
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete forward Beta calibration state") from error
    if semantic_sha256(expected) != semantic_sha256(state):
        raise ValueError("saved full Beta posterior differs from its original forecast likelihoods")
    return expected


def beta_uniform_log_probability(log_probability, state):
    """Exact one-decision posterior predictive, retaining both positive masses."""
    state = validate_beta_uniform_calibration(state)
    values = np.asarray(log_probability, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 2 or np.isnan(values).any() or np.isposinf(values).any()
            or not np.isclose(logsumexp(values), 0., rtol=0, atol=1e-10)):
        raise ValueError("Beta prediction requires normalized complete candidate log probabilities")
    return np.logaddexp(values + state["log_mean_clean"],
                        state["log_mean_epsilon"] - math.log(len(values)))
