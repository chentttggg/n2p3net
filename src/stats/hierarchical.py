"""Paired uncertainty summaries that keep target and training units distinct."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

InferenceScope = Literal["conditional_frozen_models", "training_procedure"]


@dataclass(frozen=True)
class PairedInterval:
    inference_scope: InferenceScope
    sampling_unit: Literal["participant", "independent_training_replicate"]
    n_units: int
    mean_difference: float
    confidence_level: float
    bootstrap_interval: tuple[float, float]
    bootstrap_iterations: int
    bootstrap_seed: int
    interval_method: str = "percentile_bootstrap"
    interval_estimable: bool = True
    interval_role: str = "descriptive_resampling_interval"
    finite_sample_coverage_guaranteed: bool = False
    bounded_mean_interval: tuple[float, float] | None = None
    bounded_mean_method: str | None = None
    bounded_mean_assumption: str | None = None
    population_value_bounds: tuple[float, float] | None = None

    def record(self) -> dict[str, object]:
        record = asdict(self)
        record["bootstrap_interval"] = list(self.bootstrap_interval)
        for key in ("bounded_mean_interval", "population_value_bounds"):
            if record[key] is not None:
                record[key] = list(record[key])
        return record


def paired_interval(
    differences: np.ndarray,
    *,
    inference_scope: InferenceScope,
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
    value_bounds: tuple[float, float] | None = None,
) -> PairedInterval:
    """Bootstrap one declared independent unit without silently changing scope.

    For ``conditional_frozen_models``, callers pass one paired difference per
    target participant after applying the frozen model/ensemble rule. For
    ``training_procedure``, callers pass one whole-cohort paired difference per
    independently trained replicate defined by the preregistered design.

    The percentile interval describes empirical resampling and has no finite-n
    coverage guarantee. More units, BCa or a sample-size cutoff alone do not
    supply one. With caller-declared population bounds, a separate Hoeffding
    interval covers the mean under independent representative sampling units;
    these bounds must describe the estimand, not the observed sample range.
    """

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired differences must contain at least two finite independent units.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1000:
        raise ValueError("bootstrap iterations must be an integer >= 1000.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0,1).")
    if inference_scope == "conditional_frozen_models":
        unit = "participant"
    elif inference_scope == "training_procedure":
        unit = "independent_training_replicate"
    else:
        raise ValueError(f"unsupported inference_scope {inference_scope!r}.")
    bounded_interval = None
    if value_bounds is not None:
        bounds = np.asarray(value_bounds, dtype=float)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]
                or np.any(values < bounds[0]) or np.any(values > bounds[1])):
            raise ValueError("population value_bounds must contain every paired difference.")
        radius = (bounds[1]-bounds[0])*np.sqrt(np.log(2/(1-confidence_level))/(2*len(values)))
        bounded_interval = (float(max(bounds[0], values.mean()-radius)),
                            float(min(bounds[1], values.mean()+radius)))

    rng = np.random.default_rng(seed)
    sampled = np.empty(iterations, dtype=np.float64)
    offset = 0
    while offset < iterations:
        take = min(4000, iterations - offset)
        indices = rng.integers(0, len(values), size=(take, len(values)))
        sampled[offset : offset + take] = values[indices].mean(axis=1)
        offset += take
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(sampled, [alpha / 2.0, 1.0 - alpha / 2.0])
    # When resampling every unit identically is at least alpha/2 likely, the tails
    # hit the sample endpoints; even resolved tails do not certify coverage.
    estimable = bool((1.0 / len(values)) ** len(values) < alpha / 2.0 and np.ptp(values) > 0)
    return PairedInterval(
        inference_scope=inference_scope,
        sampling_unit=unit,
        n_units=len(values),
        mean_difference=float(values.mean()),
        confidence_level=float(confidence_level),
        bootstrap_interval=(float(lower), float(upper)),
        bootstrap_iterations=iterations,
        bootstrap_seed=seed,
        interval_method="percentile_bootstrap" if estimable else "not_estimable_sample_range",
        interval_estimable=estimable,
        bounded_mean_interval=bounded_interval,
        bounded_mean_method=None if bounded_interval is None else "Hoeffding_independent_bounded_mean",
        bounded_mean_assumption=(None if bounded_interval is None else
            "independent_representative_units_with_caller_declared_population_bounds"),
        population_value_bounds=None if value_bounds is None else tuple(map(float, bounds)),
    )


__all__ = ["InferenceScope", "PairedInterval", "paired_interval"]
