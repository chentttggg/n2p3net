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

    def record(self) -> dict[str, object]:
        record = asdict(self)
        record["bootstrap_interval"] = list(self.bootstrap_interval)
        return record


def paired_interval(
    differences: np.ndarray,
    *,
    inference_scope: InferenceScope,
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> PairedInterval:
    """Bootstrap one declared independent unit without silently changing scope.

    For ``conditional_frozen_models``, callers pass one paired difference per
    target participant after applying the frozen model/ensemble rule. For
    ``training_procedure``, callers pass one whole-cohort paired difference per
    independently trained replicate defined by the preregistered design.
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
    return PairedInterval(
        inference_scope=inference_scope,
        sampling_unit=unit,
        n_units=len(values),
        mean_difference=float(values.mean()),
        confidence_level=float(confidence_level),
        bootstrap_interval=(float(lower), float(upper)),
        bootstrap_iterations=iterations,
        bootstrap_seed=seed,
    )


__all__ = ["InferenceScope", "PairedInterval", "paired_interval"]
