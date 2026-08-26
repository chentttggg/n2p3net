from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from data.channel import STANDARD_CHANNELS, channel_coords
from models.canonical import (
    CanonicalProjection,
    CoordinateResidualAttention,
    RegisteredCoordinateGPProjector,
)
from models.n2p3net import N2P3Net


def _gtn_projector() -> RegisteredCoordinateGPProjector:
    observed, _ = channel_coords(("Fz", "Cz", "Pz"))
    query, _ = channel_coords(STANDARD_CHANNELS)
    return RegisteredCoordinateGPProjector(observed, query, noise_variance=0.05)


def test_gtn_to_canonical_preserves_high_occipital_uncertainty() -> None:
    projector = _gtn_projector()
    output = projector(torch.randn(4, 3, 32))
    index = {name: i for i, name in enumerate(STANDARD_CHANNELS)}
    observed_variance = output.variance[:, [index["Fz"], index["Cz"], index["Pz"]]]
    occipital_variance = output.variance[:, [index["PO7"], index["PO8"], index["Oz"]]]

    assert output.mean.shape == (4, 8, 32)
    assert output.covariance.shape == (4, 8, 8)
    assert float(observed_variance.max()) < 0.10
    assert float(occipital_variance.min()) > 0.75
    assert torch.linalg.eigvalsh(output.covariance).min() >= -1e-6


def test_gp_mean_uses_kriging_and_cache_for_fixed_montage() -> None:
    RegisteredCoordinateGPProjector.clear_cache()
    projector = _gtn_projector()
    signal = torch.tensor([[[1.0], [2.0], [3.0]]])
    first = projector(signal)
    cache_after_first = projector.cache_info()
    second = projector(signal)
    cache_after_second = projector.cache_info()

    assert torch.allclose(first.mean, second.mean)
    assert cache_after_first.misses == 1
    assert cache_after_second.hits >= 1
    # The noisy posterior should remain close to measurements at collocated queries.
    assert first.mean[0, 0, 0].item() == pytest.approx(1.0, abs=0.12)
    assert first.mean[0, 1, 0].item() == pytest.approx(2.0, abs=0.12)
    assert first.mean[0, 3, 0].item() == pytest.approx(3.0, abs=0.14)


def test_per_trial_channel_masks_increase_uncertainty_without_fake_observations() -> None:
    projector = _gtn_projector()
    values = torch.ones(2, 3, 4)
    mask = torch.tensor([[True, True, True], [False, False, False]])
    output = projector(values, mask)

    assert torch.allclose(output.mean[1], torch.zeros_like(output.mean[1]))
    assert torch.allclose(output.variance[1], torch.ones(8), atol=1e-5)
    assert output.variance[1].mean() > output.variance[0].mean()


def test_different_trial_masks_produce_different_posterior_covariance() -> None:
    projector = _gtn_projector()
    values = torch.ones(2, 3, 4)
    mask = torch.tensor([[True, True, True], [True, False, True]])

    output = projector(values, mask)

    assert not torch.allclose(output.covariance[0], output.covariance[1])
    assert output.variance[1].mean() > output.variance[0].mean()


def test_coordinate_attention_is_exact_gp_at_initialization_and_bounded() -> None:
    projector = _gtn_projector()
    attention = CoordinateResidualAttention(
        projector.observed_positions,
        projector.query_positions,
        max_residual_fraction=0.1,
    )
    observed = torch.randn(2, 3, 5, 7)
    gp = projector(observed).mean
    assert torch.equal(attention(observed, gp), gp)

    with torch.no_grad():
        attention.gate_raw.fill_(100.0)
    corrected = attention(observed, gp)
    # A convex coordinate-attention estimate corrected by at most 10% cannot
    # become an unconstrained replacement for the GP posterior mean.
    assert torch.isfinite(corrected).all()
    assert (
        float((corrected - gp).detach().abs().max())
        <= 0.1 * float(observed.abs().max() + gp.detach().abs().max()) + 1e-6
    )


def test_gp_rejects_origin_and_negative_noise() -> None:
    query = np.array([[0.0, 0.0, 0.1]])
    with pytest.raises(ValueError, match="origin"):
        RegisteredCoordinateGPProjector(np.zeros((1, 3)), query)
    with pytest.raises(ValueError, match="noise_variance"):
        RegisteredCoordinateGPProjector(query, query, noise_variance=-1.0)


def test_gp_uses_metre_scale_chordal_kernel_and_rejects_fake_manifold_claim() -> None:
    projector = _gtn_projector()
    spec = projector.kernel_spec()
    assert spec["geometry"] == "chordal_3d"
    assert spec["coordinate_units"] == "m"
    assert spec["length_scale_m"] == pytest.approx(0.055)
    observed, _ = channel_coords(("Fz", "Cz", "Pz"))
    query, _ = channel_coords(STANDARD_CHANNELS)
    with pytest.raises(ValueError, match="Laplace-Beltrami"):
        RegisteredCoordinateGPProjector(
            observed,
            query,
            kernel_geometry="manifold_matern",
        )


class _FixedPosterior(nn.Module):
    def __init__(self, covariance: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("covariance", covariance)
        self.kernel_variance = 1.0

    def forward(self, values: torch.Tensor, channel_mask=None) -> CanonicalProjection:
        batch_size = values.shape[0]
        mean = values[:, :1].expand(batch_size, 8, *values.shape[2:]).clone()
        covariance = self.covariance[None].expand(batch_size, -1, -1)
        return CanonicalProjection(mean=mean, covariance=covariance)


def test_encoder_consumes_full_covariance_not_only_marginal_variance() -> None:
    torch.manual_seed(11)
    base = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        n_time=64,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        sfreq=64.0,
        baseline_n=12,
        d_model=16,
        temporal_kernels=(13,),
        filters_per_scale=2,
        encoder_depth=1,
        component_decoder=False,
        canonical_channel_names=STANDARD_CHANNELS,
        canonical_residual_attention=False,
    )
    independent = copy.deepcopy(base).eval()
    correlated = copy.deepcopy(base).eval()
    independent.tokenizer.canonical_projector = _FixedPosterior(torch.eye(8))
    covariance = torch.full((8, 8), 0.5)
    covariance.fill_diagonal_(1.0)
    correlated.tokenizer.canonical_projector = _FixedPosterior(covariance)
    X = torch.randn(3, 3, 64)
    with torch.no_grad():
        left = independent(X)
        right = correlated(X)

    assert not torch.equal(left.canonical_covariance, right.canonical_covariance)
    assert not torch.equal(left.features, right.features)
    assert not torch.equal(left.heads.logit_target, right.heads.logit_target)
