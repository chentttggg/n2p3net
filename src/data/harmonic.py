"""Explicit offline, segment-local robust line regression for operator studies.

This operator estimates nuisance parameters from each supplied segment. It is
not frozen-state inference, a streaming filter, or a replacement for acquisition
QC. It subtracts only fitted sinusoids; drift terms condition the regression but
are not subtracted. See de Cheveigne and Arzounian (2018), Fig. 5.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class HarmonicConfig:
    frequency_hz: float = 50.0
    search_half_width_hz: float = 0.5
    window_seconds: float = 1.0
    overlap_fraction: float = 0.5
    huber_multiplier: float = 2.5
    min_line_to_residual_rms: float = 0.2
    robust_iterations: int = 5
    envelope_order: int = 3
    protected_low_frequency_hz: float = 30.0

    def validate(self, sfreq):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("harmonic configuration must be finite")
        if not (0 < self.frequency_hz - self.search_half_width_hz
                < self.frequency_hz + self.search_half_width_hz < sfreq / 2):
            raise ValueError("line search must be positive and strictly below Nyquist")
        if self.window_seconds <= 0 or not 0 <= self.overlap_fraction < 1:
            raise ValueError("invalid harmonic window or overlap")
        if self.huber_multiplier <= 0 or self.min_line_to_residual_rms < 0:
            raise ValueError("invalid robust regression thresholds")
        if not isinstance(self.robust_iterations, int) or self.robust_iterations < 1:
            raise ValueError("robust_iterations must be a positive integer")
        if not isinstance(self.envelope_order, int) or not 0 <= self.envelope_order <= 3:
            raise ValueError("envelope_order must be an integer from zero through three")
        if not 0 <= self.protected_low_frequency_hz < self.frequency_hz - self.search_half_width_hz:
            raise ValueError("protected nuisance band must stay below the line search")


@dataclass(frozen=True)
class HarmonicResult:
    cleaned: np.ndarray
    windows: tuple[dict, ...]
    input_sha256: str
    output_sha256: str
    config: HarmonicConfig
    protocol: str = "offline_segment_local_estimation"

    def record(self):
        return {
            "schema": "n2p3_harmonic_segment_processing/1",
            "protocol": self.protocol,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "config": asdict(self.config),
            "windows": list(self.windows),
        }


def _digest(array):
    value = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def _fit_window(y, sfreq, config):
    times = np.arange(len(y), dtype=float) / sfreq
    centered = times - times.mean()
    trend = centered / max(np.ptp(centered), 1 / sfreq)
    modes = np.arange(int(np.floor(2 * len(y) * config.protected_low_frequency_hz / sfreq)) + 1)
    low_basis = np.cos(np.pi * (np.arange(len(y))[:, None] + 0.5) * modes / len(y))
    nuisance = np.column_stack([low_basis, trend])
    scale_floor = max(np.max(np.abs(y)) * 1e-12, np.finfo(float).tiny)

    def solve(freq, iterations):
        phase = 2 * np.pi * freq * centered
        sine, cosine = np.sin(phase), np.cos(phase)
        # A low-order local envelope handles smooth amplitude/phase modulation.
        # Frequency is a basis center, not an independently identifiable physical
        # estimate once time-varying quadrature amplitudes are allowed.
        harmonic = np.column_stack([wave * trend**order
                                    for order in range(config.envelope_order + 1)
                                    for wave in (sine, cosine)])
        design = np.column_stack([nuisance, harmonic])
        weights = np.ones(len(y))
        for _ in range(iterations):
            root = np.sqrt(weights)
            coef, _, rank, _ = lstsq(design * root[:, None], y * root,
                                    lapack_driver="gelsy", check_finite=False)
            if rank != design.shape[1]:
                raise ValueError("line and protected signal subspaces are not identifiable")
            residual = y - design @ coef
            scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))),
                        scale_floor)
            weights = np.minimum(1, config.huber_multiplier * scale
                                 / np.maximum(np.abs(residual), scale_floor))
        return harmonic @ coef[nuisance.shape[1]:], residual, weights, scale

    low = config.frequency_hz - config.search_half_width_hz
    high = config.frequency_hz + config.search_half_width_hz
    # Resolve the main lobe before the local optimizer; no full-record spectrum.
    step = min(0.1, 0.25 * sfreq / len(y))
    grid = np.linspace(low, high, max(3, int(np.ceil((high - low) / step)) + 1))
    fits = [solve(freq, 2) for freq in grid]
    costs = [np.median(np.abs(fit[1])) for fit in fits]
    best = int(np.argmin(costs))
    delta = config.huber_multiplier * fits[best][3]

    def objective(freq):
        residual = np.abs(solve(freq, config.robust_iterations)[1])
        return float(np.mean(np.where(residual <= delta,
                                     residual**2 / 2, delta * (residual - delta / 2))))

    left, right = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    optimized = minimize_scalar(objective, bounds=(left, right), method="bounded",
                                options={"xatol": 1e-10})
    frequency = float(optimized.x)
    line, residual, weights, scale = solve(frequency, config.robust_iterations)
    line_rms = float(np.sqrt(np.mean(line**2)))
    residual_rms = float(np.sqrt(np.average(residual**2, weights=weights)))
    applied = line_rms >= config.min_line_to_residual_rms * max(residual_rms, scale_floor)
    return (line if applied else np.zeros_like(line)), {
        "frequency_hz": frequency,
        "line_rms_v": line_rms,
        "robust_residual_rms_v": residual_rms,
        "robust_scale_v": float(scale),
        "downweighted_fraction": float(np.mean(weights < 0.99)),
        "applied": bool(applied),
        "frequency_at_search_edge": bool(min(frequency - low, high - frequency) < step / 4),
    }


def clean_harmonic_segments(X, sfreq, segments, config=None):
    """Return cleaned physical-voltage channels without reading across segments.

    X has shape (observed_channels, samples). Segments must partition all samples
    in order; neither absent channels nor gaps are silently reconstructed. Every
    segment needs at least one second. The declared polynomial envelope expands
    the narrow-band model; abrupt modulation remains an explicit limitation.
    The caller owns physical source/fit authority and must declare this operator
    separately from frozen-model and causal preprocessing routes.
    """
    raw = np.asarray(X, dtype=float)
    config = HarmonicConfig() if config is None else config
    if raw.ndim != 2 or min(raw.shape) == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("expected finite nonempty observed channels x samples")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be finite and positive")
    config.validate(sfreq)
    intervals = list(segments)
    previous = 0
    for interval in intervals:
        if len(interval) != 2 or any(not isinstance(v, (int, np.integer)) for v in interval):
            raise ValueError("segments require exact integer sample bounds")
        start, stop = interval
        if start != previous or stop > raw.shape[1] or stop - start < max(16, sfreq):
            raise ValueError("segments must partition input and contain at least one second")
        previous = stop
    if previous != raw.shape[1]:
        raise ValueError("segments must cover every supplied sample")
    removed = np.zeros_like(raw)
    ledger = []
    requested_window = max(16, round(config.window_seconds * sfreq))
    for segment_id, (start, stop) in enumerate(intervals):
        length = stop - start
        width = min(requested_window, length)
        hop = max(1, round(width * (1 - config.overlap_fraction)))
        starts = list(range(start, stop - width + 1, hop))
        if starts[-1] != stop - width:
            starts.append(stop - width)
        blend = np.hanning(width + 2)[1:-1]
        denominator = np.zeros(length)
        for offset in starts:
            local = slice(offset - start, offset - start + width)
            denominator[local] += blend
            for channel, samples in enumerate(raw[:, offset:offset + width]):
                line, record = _fit_window(samples, sfreq, config)
                removed[channel, offset:offset + width] += line * blend
                ledger.append({"segment": segment_id, "channel": channel,
                               "start": offset, "stop": offset + width, **record})
        removed[:, start:stop] /= denominator
    cleaned = raw - removed
    if not np.all(np.isfinite(cleaned)):
        raise ValueError("harmonic cleaning produced nonfinite output")
    return HarmonicResult(cleaned, tuple(ledger), _digest(raw), _digest(cleaned), config)
