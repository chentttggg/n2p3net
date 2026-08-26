"""ERP predictive uncertainty calibration and aggregation tests."""

from __future__ import annotations

import pytest
import torch

from models.erp_uncertainty import (
    evaluate_erp_uncertainty,
    fit_erp_variance_calibration,
    inverse_variance_aggregate,
)


def test_variance_scale_recovers_known_global_miscalibration() -> None:
    target = torch.tensor([-2.0, -1.0, 1.0, 2.0]).view(1, 1, -1)
    mean = torch.zeros_like(target)
    variance = target.square() / 4.0
    calibration = fit_erp_variance_calibration(
        target,
        mean,
        variance,
        source="subject_disjoint_validation",
    )
    assert abs(calibration.scale - 4.0) < 1e-6
    metrics = evaluate_erp_uncertainty(target, mean, calibration.apply(variance))
    assert abs(metrics.standardized_residual_rms - 1.0) < 1e-6
    assert metrics.n_observations == 4


def test_uncertainty_metrics_respect_broadcast_mask() -> None:
    target = torch.zeros(2, 3, 4)
    mean = target.clone()
    mean[:, 2] = 100.0
    variance = torch.ones_like(target)
    metrics = evaluate_erp_uncertainty(
        target,
        mean,
        variance,
        mask=torch.tensor([True, True, False])[:, None],
    )
    assert metrics.rmse == 0.0
    assert metrics.n_observations == 16
    assert all(value == 1.0 for value in metrics.interval_coverage.values())


def test_inverse_variance_aggregation_prefers_precise_trials() -> None:
    mean = torch.tensor([0.0, 10.0, 20.0]).view(3, 1, 1)
    variance = torch.tensor([1.0, 4.0, 100.0]).view(3, 1, 1)
    result = inverse_variance_aggregate(mean, variance)
    precision = variance.reciprocal()
    expected = (precision * mean).sum(dim=0) / precision.sum(dim=0)
    assert torch.allclose(result.mean, expected)
    assert torch.allclose(result.variance, precision.sum(dim=0).reciprocal())
    assert result.normalized_weights[0] > result.normalized_weights[1]
    assert result.normalized_weights[1] > result.normalized_weights[2]
    assert torch.all((result.effective_sample_size >= 1.0) & (result.effective_sample_size <= 3.0))


def test_uncertainty_apis_fail_closed_on_unmasked_invalid_variance() -> None:
    target = torch.zeros(2, 1, 1)
    mean = torch.zeros_like(target)
    variance = torch.tensor([1.0, -1.0]).view_as(target)
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_erp_uncertainty(target, mean, variance)
    with pytest.raises(ValueError, match="non-negative"):
        inverse_variance_aggregate(mean, variance)

    aggregate = inverse_variance_aggregate(
        mean,
        variance,
        trial_mask=torch.tensor([True, False]),
    )
    assert aggregate.mean.item() == 0.0
    assert aggregate.variance.item() == 1.0
