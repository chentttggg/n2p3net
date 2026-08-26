"""Calibration, diagnostics and aggregation for ERP predictive moments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import torch


@dataclass(frozen=True)
class ERPVarianceCalibration:
    """Validation-fitted multiplicative calibration for predictive variance."""

    scale: float
    source: str
    n_observations: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("ERP variance calibration scale must be finite and positive.")
        if not self.source:
            raise ValueError("ERP variance calibration requires an auditable source.")
        if self.n_observations < 1:
            raise ValueError("ERP variance calibration requires observations.")

    def apply(self, variance: torch.Tensor) -> torch.Tensor:
        return variance * float(self.scale)


@dataclass(frozen=True)
class ERPUncertaintyMetrics:
    gaussian_nll: float
    rmse: float
    sharpness: float
    standardized_residual_mean: float
    standardized_residual_rms: float
    interval_coverage: dict[float, float]
    coverage_calibration_error: float
    n_observations: int


@dataclass(frozen=True)
class ERPTrialAggregate:
    """Precision-weighted ERP moments under independent Gaussian estimates."""

    mean: torch.Tensor
    variance: torch.Tensor
    normalized_weights: torch.Tensor
    effective_sample_size: torch.Tensor


def _validated_observations(
    target: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    variance_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.shape != mean.shape or target.shape != variance.shape:
        raise ValueError("target, mean and variance must have identical shapes.")
    if target.numel() == 0 or variance_floor <= 0.0:
        raise ValueError("ERP uncertainty inputs must be non-empty with a positive floor.")
    selected = torch.ones_like(target, dtype=torch.bool)
    if mask is not None:
        try:
            selected = torch.broadcast_to(
                mask.to(device=target.device, dtype=torch.bool), target.shape
            )
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to ERP observations.") from exc
    if not bool(selected.any()):
        raise ValueError("No ERP uncertainty observations were selected.")
    finite = torch.isfinite(target) & torch.isfinite(mean) & torch.isfinite(variance)
    if not bool(finite[selected].all()):
        raise ValueError("Selected ERP uncertainty observations contain NaN/inf.")
    if bool((variance[selected] < 0.0).any()):
        raise ValueError("Selected ERP predictive variances must be non-negative.")
    error = target.detach().float()[selected] - mean.detach().float()[selected]
    selected_variance = variance.detach().float()[selected].clamp_min(float(variance_floor))
    return error, selected_variance


def fit_erp_variance_calibration(
    target: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    source: str = "subject_disjoint_validation",
    variance_floor: float = 1e-6,
    scale_bounds: tuple[float, float] = (0.05, 20.0),
) -> ERPVarianceCalibration:
    """Fit the Gaussian NLL-optimal scalar using non-test observations only."""

    lower, upper = (float(value) for value in scale_bounds)
    if not 0.0 < lower < upper:
        raise ValueError("scale_bounds must be positive and ordered.")
    error, selected_variance = _validated_observations(
        target,
        mean,
        variance,
        mask,
        variance_floor=variance_floor,
    )
    scale = float((error.square() / selected_variance).mean().clamp(lower, upper))
    return ERPVarianceCalibration(
        scale=scale,
        source=str(source),
        n_observations=int(error.numel()),
    )


def evaluate_erp_uncertainty(
    target: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    interval_levels: Sequence[float] = (0.50, 0.80, 0.90, 0.95),
    variance_floor: float = 1e-6,
) -> ERPUncertaintyMetrics:
    """Evaluate Gaussian calibration without refitting on the evaluated data."""

    levels = tuple(float(level) for level in interval_levels)
    if not levels or any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("interval_levels must contain probabilities in (0,1).")
    if len(set(levels)) != len(levels):
        raise ValueError("interval_levels must be unique.")
    error, selected_variance = _validated_observations(
        target,
        mean,
        variance,
        mask,
        variance_floor=variance_floor,
    )
    standard_deviation = selected_variance.sqrt()
    standardized = error / standard_deviation
    coverage: dict[float, float] = {}
    for level in levels:
        quantile = NormalDist().inv_cdf(0.5 * (1.0 + level))
        coverage[level] = float((standardized.abs() <= quantile).float().mean())
    calibration_error = sum(abs(coverage[level] - level) for level in levels) / len(levels)
    gaussian_nll = 0.5 * (
        error.square() / selected_variance + selected_variance.log() + math.log(2.0 * math.pi)
    )
    return ERPUncertaintyMetrics(
        gaussian_nll=float(gaussian_nll.mean()),
        rmse=float(error.square().mean().sqrt()),
        sharpness=float(standard_deviation.mean()),
        standardized_residual_mean=float(standardized.mean()),
        standardized_residual_rms=float(standardized.square().mean().sqrt()),
        interval_coverage=coverage,
        coverage_calibration_error=float(calibration_error),
        n_observations=int(error.numel()),
    )


def inverse_variance_aggregate(
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    dim: int = 0,
    trial_mask: torch.Tensor | None = None,
    variance_floor: float = 1e-6,
) -> ERPTrialAggregate:
    """Aggregate independent ERP estimates using pointwise precision weights."""

    if mean.shape != variance.shape or mean.numel() == 0:
        raise ValueError("mean and variance must be non-empty with identical shapes.")
    if variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive.")
    dim = dim % mean.dim()
    active = torch.ones_like(mean, dtype=torch.bool)
    if trial_mask is not None:
        mask = trial_mask.to(device=mean.device, dtype=torch.bool)
        if mask.dim() != 1 or mask.numel() != mean.shape[dim]:
            raise ValueError("trial_mask must match the aggregation dimension.")
        shape = [1] * mean.dim()
        shape[dim] = mask.numel()
        active = torch.broadcast_to(mask.reshape(shape), mean.shape)
    finite = torch.isfinite(mean) & torch.isfinite(variance)
    if not bool(finite[active].all()):
        raise ValueError("Selected ERP estimates contain NaN/inf.")
    if bool((variance[active] < 0.0).any()):
        raise ValueError("Selected ERP predictive variances must be non-negative.")
    safe_variance = torch.where(active, variance.float(), torch.ones_like(variance.float()))
    precision = safe_variance.clamp_min(float(variance_floor)).reciprocal() * active
    precision_sum = precision.sum(dim=dim, keepdim=True)
    if bool((precision_sum <= 0.0).any()):
        raise ValueError("At least one ERP estimate is required at every output point.")
    normalized_weights = precision / precision_sum
    safe_mean = torch.where(active, mean.float(), torch.zeros_like(mean.float()))
    aggregate_mean = (normalized_weights * safe_mean).sum(dim=dim)
    aggregate_variance = precision_sum.reciprocal().squeeze(dim)
    effective_n = normalized_weights.square().sum(dim=dim).reciprocal()
    return ERPTrialAggregate(
        mean=aggregate_mean,
        variance=aggregate_variance,
        normalized_weights=normalized_weights,
        effective_sample_size=effective_n,
    )
