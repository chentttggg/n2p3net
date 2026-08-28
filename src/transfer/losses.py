"""Loss components for masked EEG reconstruction.

The waveform loss follows SpellerSSL. The spectral term is a band-balanced
variant of the FFT-consistency loss: each canonical P300 band contributes
equally to supervision instead of the high-power low-frequency component
dominating the gradient (the same failure mode diagnosed by FAME).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.nn import functional as F

DEFAULT_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
)


@dataclass(frozen=True)
class ReconstructionLossConfig:
    waveform_weight: float = 1.0
    spectral_weight: float = 1.0
    bands_hz: tuple[tuple[float, float], ...] = DEFAULT_BANDS_HZ

    def validate(self, sfreq: float) -> None:
        if self.waveform_weight < 0.0 or self.spectral_weight < 0.0:
            raise ValueError("reconstruction loss weights must be non-negative.")
        if self.waveform_weight == 0.0 and self.spectral_weight == 0.0:
            raise ValueError("at least one reconstruction term must be active.")
        if not self.bands_hz:
            raise ValueError("bands_hz must not be empty.")
        nyquist = sfreq / 2.0
        for start, end in self.bands_hz:
            if start < 0.0 or end <= start or end > nyquist:
                raise ValueError(f"invalid band {start}-{end} Hz at sfreq={sfreq}.")


@lru_cache(maxsize=64)
def _spectral_basis(
    n_times: int,
    sfreq: float,
    bands_hz: tuple[tuple[float, float], ...],
    device_name: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache the fixed Hann window and band-reduction matrix per device."""

    basis_dtype = torch.float64 if dtype is torch.float64 else torch.float32
    freqs = torch.fft.rfftfreq(n_times, d=1.0 / sfreq, dtype=basis_dtype)
    selectors = torch.stack(
        [(freqs >= start) & (freqs < end) for start, end in bands_hz]
    )
    counts = selectors.sum(dim=1)
    if bool((counts == 0).any()):
        empty = int(torch.nonzero(counts == 0, as_tuple=False)[0])
        start, end = bands_hz[empty]
        raise ValueError(f"band {start}-{end} Hz contains no FFT bins.")
    reduction = selectors.to(basis_dtype) / counts[:, None]
    window = torch.hann_window(n_times, periodic=False, dtype=basis_dtype)
    device = torch.device(device_name)
    return window.to(device=device, dtype=dtype), reduction.to(device=device, dtype=dtype)


def _band_magnitude_tensor(
    x: torch.Tensor,
    *,
    sfreq: float,
    bands_hz: Sequence[tuple[float, float]],
) -> torch.Tensor:
    """Return band magnitudes with the band axis first."""

    t = x.shape[-1]
    bands = tuple((float(start), float(end)) for start, end in bands_hz)
    window, reduction = _spectral_basis(t, float(sfreq), bands, str(x.device), x.dtype)
    centered = x - x.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(centered * window, dim=-1).abs()
    return torch.einsum("...f,kf->k...", spectrum, reduction)


def band_magnitudes(
    x: torch.Tensor,
    *,
    sfreq: float,
    bands_hz: Sequence[tuple[float, float]],
) -> list[torch.Tensor]:
    """Return mean absolute FFT magnitude inside each band.

    ``x`` is ``(B,C,T)`` or ``(C,T)``. The signal is detrended and Hann-windowed
    before the FFT so the reconstruction target is the ERP morphology, not the
    epoch mean or edge step.
    """

    if x.ndim not in {2, 3}:
        raise ValueError("band_magnitudes expects (C,T) or (B,C,T).")
    if x.shape[-1] < 8:
        raise ValueError("the input is too short for a stable FFT magnitude.")
    return list(_band_magnitude_tensor(x, sfreq=sfreq, bands_hz=bands_hz).unbind(0))


def estimate_band_weights(
    x: torch.Tensor,
    *,
    sfreq: float,
    bands_hz: Sequence[tuple[float, float]] = DEFAULT_BANDS_HZ,
) -> torch.Tensor:
    """Estimate inverse-average band weights from a finite source sample.

    The result is used only as a loss-scale normalizer, not as model input.
    """

    magnitudes = _band_magnitude_tensor(x, sfreq=sfreq, bands_hz=bands_hz)
    means = magnitudes.flatten(start_dim=1).mean(dim=1)
    means = means.clamp_min(1e-12)
    weights = (1.0 / means) / (1.0 / means).sum()
    return weights


def band_balanced_spectral_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    *,
    sfreq: float,
    bands_hz: Sequence[tuple[float, float]] = DEFAULT_BANDS_HZ,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.shape != x_hat.shape:
        raise ValueError("x and x_hat must have identical shape.")
    paired = torch.stack((x, x_hat), dim=0)
    magnitudes = _band_magnitude_tensor(paired, sfreq=sfreq, bands_hz=bands_hz)
    magnitudes_x, magnitudes_hat = magnitudes.unbind(dim=1)
    if weights is None:
        weights = torch.ones(len(bands_hz), device=x.device, dtype=x.dtype) / len(bands_hz)
    if weights.shape != (len(bands_hz),):
        raise ValueError("weights must contain one value per band.")
    weights = weights.to(device=x.device, dtype=x.dtype)
    per_band = F.l1_loss(magnitudes_x, magnitudes_hat, reduction="none")
    view = (-1,) + (1,) * (per_band.ndim - 1)
    return (weights.view(*view) * per_band).sum(dim=0).mean()


def waveform_l1_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    if x.shape != x_hat.shape:
        raise ValueError("x and x_hat must have identical shape.")
    return F.l1_loss(x_hat, x)


def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    *,
    sfreq: float,
    config: ReconstructionLossConfig,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return a labelled loss dict suitable for recording and weighting."""

    config.validate(sfreq)
    wave = waveform_l1_loss(x, x_hat)
    spec = band_balanced_spectral_loss(
        x,
        x_hat,
        sfreq=sfreq,
        bands_hz=config.bands_hz,
        weights=weights,
    )
    weighted_wave = config.waveform_weight * wave
    weighted_spec = config.spectral_weight * spec
    return {
        "waveform": weighted_wave,
        "spectral": weighted_spec,
        "total": weighted_wave + weighted_spec,
    }
