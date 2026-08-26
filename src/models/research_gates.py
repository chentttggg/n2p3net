"""Pre-registered gate for morphology recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MorphologyThresholds:
    latency_mae_ms_max: float = 20.0
    width_relative_error_max: float = 0.20
    interval_coverage_min: float = 0.85
    interval_coverage_max: float = 0.95


@dataclass(frozen=True)
class MorphologyRecoveryReport:
    passed: bool
    latency_mae_ms: float
    width_relative_error: float
    interval_coverage: float
    n_samples: int
    checks: dict[str, bool]
    thresholds: MorphologyThresholds

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_morphology_recovery(
    true_latency_ms: np.ndarray,
    predicted_latency_ms: np.ndarray,
    true_width_ms: np.ndarray,
    predicted_width_ms: np.ndarray,
    interval_lower_ms: np.ndarray,
    interval_upper_ms: np.ndarray,
    *,
    thresholds: MorphologyThresholds | None = None,
) -> MorphologyRecoveryReport:
    thresholds = MorphologyThresholds() if thresholds is None else thresholds
    arrays = [
        np.asarray(value, dtype=float).reshape(-1)
        for value in (
            true_latency_ms,
            predicted_latency_ms,
            true_width_ms,
            predicted_width_ms,
            interval_lower_ms,
            interval_upper_ms,
        )
    ]
    if len({value.shape for value in arrays}) != 1 or arrays[0].size < 2:
        raise ValueError("Morphology evidence arrays must align and contain at least two rows.")
    finite = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    if finite.sum() < 2:
        raise ValueError("Morphology evidence needs at least two finite rows.")
    true_latency, predicted_latency, true_width, predicted_width, lower, upper = (
        value[finite] for value in arrays
    )
    if np.any(true_width <= 0.0) or np.any(predicted_width <= 0.0):
        raise ValueError("Morphology widths must be positive.")
    if np.any(lower > upper):
        raise ValueError("Every uncertainty interval must satisfy lower<=upper.")

    latency_mae = float(np.mean(np.abs(predicted_latency - true_latency)))
    width_error = float(np.mean(np.abs(predicted_width - true_width) / true_width))
    coverage = float(np.mean((true_latency >= lower) & (true_latency <= upper)))
    checks = {
        "latency_mae": latency_mae < thresholds.latency_mae_ms_max,
        "width_relative_error": width_error < thresholds.width_relative_error_max,
        "interval_coverage": (
            thresholds.interval_coverage_min <= coverage <= thresholds.interval_coverage_max
        ),
    }
    return MorphologyRecoveryReport(
        passed=all(checks.values()),
        latency_mae_ms=latency_mae,
        width_relative_error=width_error,
        interval_coverage=coverage,
        n_samples=int(finite.sum()),
        checks=checks,
        thresholds=thresholds,
    )
