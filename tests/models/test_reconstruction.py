from __future__ import annotations

import pytest
import torch

from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from transfer.heads import SubjectProbeHead, WaveDecoderHead
from transfer.losses import (
    DEFAULT_BANDS_HZ,
    ReconstructionLossConfig,
    band_balanced_spectral_loss,
    band_magnitudes,
    estimate_band_weights,
    reconstruction_loss,
)
from transfer.masking import MaskingConfig, apply_time_mask, make_temporal_mask


def test_temporal_mask_is_deterministic_and_nonempty() -> None:
    config = MaskingConfig(mask_fraction=0.5, min_block_samples=12, max_block_samples=32)
    gen = torch.Generator().manual_seed(7)
    first = make_temporal_mask(128, config=config, generator=gen)
    gen = torch.Generator().manual_seed(7)
    second = make_temporal_mask(128, config=config, generator=gen)
    assert torch.equal(first, second)
    fraction = float(first.to(torch.float32).mean())
    assert 0.35 <= fraction <= 0.65
    x = torch.randn(4, 3, 128)
    masked, keep = apply_time_mask(x, first)
    assert masked.shape == x.shape
    assert torch.all(masked[:, :, first] == 0.0)
    assert keep.dtype == x.dtype


def test_band_weights_sum_to_one_and_zero_reconstruction_loss() -> None:
    x = torch.randn(8, 3, 128)
    weights = estimate_band_weights(x, sfreq=128.0)
    assert weights.shape == (4,)
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    config = ReconstructionLossConfig()
    losses = reconstruction_loss(x, x, sfreq=128.0, config=config, weights=weights)
    assert float(losses["waveform"]) < 1e-6
    assert float(losses["spectral"]) < 1e-5


def test_reconstruction_loss_ignores_visible_region_errors() -> None:
    x = torch.zeros(2, 3, 128)
    x_hat = x.clone()
    x_hat[..., :64] = 10.0
    masked_region = torch.zeros(1, 1, 128, dtype=torch.bool)
    masked_region[..., 64:] = True

    losses = reconstruction_loss(
        x,
        x_hat,
        sfreq=128.0,
        config=ReconstructionLossConfig(waveform_weight=1.0, spectral_weight=0.0),
        sample_mask=masked_region,
    )
    assert float(losses["total"]) == pytest.approx(0.0, abs=1e-8)


def test_band_balanced_loss_is_lower_for_exact_waveform() -> None:
    x = torch.randn(4, 3, 128)
    bad = torch.randn(4, 3, 128)
    good = band_balanced_spectral_loss(x, x, sfreq=128.0)
    worse = band_balanced_spectral_loss(x, bad, sfreq=128.0)
    assert float(good) < float(worse)


def test_band_balanced_loss_is_invariant_to_repeated_batch_rows() -> None:
    x = torch.randn(3, 3, 128)
    x_hat = x + 0.1 * torch.randn_like(x)
    single = band_balanced_spectral_loss(x, x_hat, sfreq=128.0)
    repeated = band_balanced_spectral_loss(
        x.repeat(4, 1, 1),
        x_hat.repeat(4, 1, 1),
        sfreq=128.0,
    )
    assert torch.allclose(single, repeated, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(3, 128), (2, 3, 128)])
def test_cached_band_projection_matches_direct_slices_and_has_gradient(shape) -> None:
    x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    actual = torch.stack(band_magnitudes(x, sfreq=128.0, bands_hz=DEFAULT_BANDS_HZ))
    window = torch.hann_window(128, periodic=False, dtype=x.dtype)
    spectrum = torch.fft.rfft((x - x.mean(dim=-1, keepdim=True)) * window, dim=-1).abs()
    freqs = torch.fft.rfftfreq(128, d=1.0 / 128.0, dtype=x.dtype)
    expected = torch.stack(
        [
            spectrum[..., (freqs >= start) & (freqs < end)].mean(dim=-1)
            for start, end in DEFAULT_BANDS_HZ
        ]
    )
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-12)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_wave_decoder_matches_contract_and_is_small() -> None:
    decoder = WaveDecoderHead(trunk_channels=4, output_channels=3, st_pool_size=4)
    features = torch.randn(2, 4, 32)
    mask = torch.ones(2, 128)
    out = decoder(features, mask)
    assert out.shape == (2, 3, 128)
    assert decoder.parameter_count() < 200


def test_subject_probe_shape() -> None:
    probe = SubjectProbeHead(16, 5)
    out = probe(torch.randn(3, 16))
    assert out.shape == (3, 5)


def test_wave_decoder_restores_nondivisible_causal_length() -> None:
    decoder = WaveDecoderHead(trunk_channels=4, output_channels=3, st_pool_size=4)
    mask = torch.ones(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times)
    features = torch.randn(2, 4, len(mask) // 4)

    reconstructed = decoder(features, mask)

    assert reconstructed.shape == (2, 3, len(mask))
