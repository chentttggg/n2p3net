"""Diagnostic gate for Parameterized Component Window routing claims.

The PCW ``tau`` output is a classification routing parameter, not a
physiological latency measurement. This module may only promote the claim
between ``fold-calibrated routing window`` (default) and
``fold-calibrated discriminative routing window`` after the diagnostic checks
pass. Physiological latency claims are reserved for the independent
``measurement.LatencyMeasurement`` object.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyRecovery:
    rmse_ms: float
    pearson_r: float
    n_samples: int


@dataclass(frozen=True)
class GradientHealth:
    tau_gradient_norm: float
    reference_gradient_norm: float
    gradient_ratio: float
    tau0_drift_ms: float


@dataclass(frozen=True)
class SplitHalfStability:
    pearson_r: float
    median_absolute_difference_ms: float
    n_pairs: int


@dataclass(frozen=True)
class AblationEvidence:
    pcw_score: float
    fixed_window_score: float
    mean_pool_score: float
    gain_over_best_control: float


@dataclass(frozen=True)
class PCWGateThresholds:
    max_recovery_rmse_ms: float = 40.0
    min_recovery_correlation: float = 0.80
    min_gradient_ratio: float = 1e-3
    min_tau0_drift_ms: float = 2.0
    min_split_half_correlation: float = 0.70
    max_split_half_difference_ms: float = 30.0
    min_ablation_gain: float = 0.005


DEFAULT_PCW_GATE_THRESHOLDS = PCWGateThresholds()


@dataclass(frozen=True)
class PCWClaimGate:
    passed: bool
    allowed_claim: str
    checks: dict[str, bool]
    failures: tuple[str, ...]
    recovery: LatencyRecovery
    gradient: GradientHealth
    split_half: SplitHalfStability
    ablation: AblationEvidence
    thresholds: PCWGateThresholds

    def to_dict(self) -> dict:
        return asdict(self)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def latency_recovery(
    true_latency_ms: np.ndarray, predicted_latency_ms: np.ndarray
) -> LatencyRecovery:
    true = np.asarray(true_latency_ms, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_latency_ms, dtype=float).reshape(-1)
    if true.shape != predicted.shape:
        raise ValueError("true and predicted latency arrays must align.")
    finite = np.isfinite(true) & np.isfinite(predicted)
    if finite.sum() < 2:
        raise ValueError("latency recovery needs at least two finite pairs.")
    error = predicted[finite] - true[finite]
    return LatencyRecovery(
        rmse_ms=float(np.sqrt(np.mean(error * error))),
        pearson_r=_correlation(true[finite], predicted[finite]),
        n_samples=int(finite.sum()),
    )


def gradient_health(
    tau_gradient_norms: np.ndarray,
    reference_gradient_norms: np.ndarray,
    tau0_before_ms: np.ndarray,
    tau0_after_ms: np.ndarray,
) -> GradientHealth:
    tau_grad = np.asarray(tau_gradient_norms, dtype=float)
    reference_grad = np.asarray(reference_gradient_norms, dtype=float)
    if tau_grad.size == 0 or reference_grad.size == 0:
        raise ValueError("gradient diagnostics need non-empty norm traces.")
    tau_norm = float(np.median(tau_grad[np.isfinite(tau_grad)]))
    ref_norm = float(np.median(reference_grad[np.isfinite(reference_grad)]))
    before = np.asarray(tau0_before_ms, dtype=float)
    after = np.asarray(tau0_after_ms, dtype=float)
    if before.shape != after.shape:
        raise ValueError("tau0 before/after arrays must align.")
    return GradientHealth(
        tau_gradient_norm=tau_norm,
        reference_gradient_norm=ref_norm,
        gradient_ratio=tau_norm / max(ref_norm, 1e-12),
        tau0_drift_ms=float(np.linalg.norm(after - before)),
    )


def split_half_stability(first_ms: np.ndarray, second_ms: np.ndarray) -> SplitHalfStability:
    first = np.asarray(first_ms, dtype=float).reshape(-1)
    second = np.asarray(second_ms, dtype=float).reshape(-1)
    if first.shape != second.shape:
        raise ValueError("split-half latency arrays must align.")
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        raise ValueError("split-half stability needs at least two finite pairs.")
    return SplitHalfStability(
        pearson_r=_correlation(first[finite], second[finite]),
        median_absolute_difference_ms=float(np.median(np.abs(first[finite] - second[finite]))),
        n_pairs=int(finite.sum()),
    )


def ablation_evidence(
    pcw_score: float,
    fixed_window_score: float,
    mean_pool_score: float,
) -> AblationEvidence:
    values = np.asarray([pcw_score, fixed_window_score, mean_pool_score], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ablation scores must be finite.")
    control = max(float(fixed_window_score), float(mean_pool_score))
    return AblationEvidence(
        pcw_score=float(pcw_score),
        fixed_window_score=float(fixed_window_score),
        mean_pool_score=float(mean_pool_score),
        gain_over_best_control=float(pcw_score - control),
    )


def evaluate_pcw_claim_gate(
    recovery: LatencyRecovery,
    gradient: GradientHealth,
    split_half: SplitHalfStability,
    ablation: AblationEvidence,
    thresholds: PCWGateThresholds = DEFAULT_PCW_GATE_THRESHOLDS,
) -> PCWClaimGate:
    checks = {
        "synthetic_latency_recovery": (
            math.isfinite(recovery.pearson_r)
            and recovery.pearson_r >= thresholds.min_recovery_correlation
            and recovery.rmse_ms <= thresholds.max_recovery_rmse_ms
        ),
        "gradient_and_drift": (
            math.isfinite(gradient.gradient_ratio)
            and gradient.gradient_ratio >= thresholds.min_gradient_ratio
            and gradient.tau0_drift_ms >= thresholds.min_tau0_drift_ms
        ),
        "real_split_half_stability": (
            math.isfinite(split_half.pearson_r)
            and split_half.pearson_r >= thresholds.min_split_half_correlation
            and split_half.median_absolute_difference_ms <= thresholds.max_split_half_difference_ms
        ),
        "fixed_window_mean_pool_ablation": (
            ablation.gain_over_best_control >= thresholds.min_ablation_gain
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    passed = not failures
    claim = (
        "fold-calibrated discriminative routing window"
        if passed
        else "fold-calibrated routing window"
    )
    return PCWClaimGate(
        passed=passed,
        allowed_claim=claim,
        checks=checks,
        failures=failures,
        recovery=recovery,
        gradient=gradient,
        split_half=split_half,
        ablation=ablation,
        thresholds=thresholds,
    )
