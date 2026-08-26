"""P300-specific, label-free feature lexicon for the PEC audit.

The registry deliberately describes signal observables rather than model
outputs.  It contains 63 scalar features in six families, matching the
audit's need for a compact, reproducible concept vocabulary while adapting
the generic clinical lexicon to stimulus-locked P300 epochs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import factorial
from typing import Literal

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from .types import AuditInputError, FeatureTable, P300AuditData


@dataclass(frozen=True)
class P300Windows:
    """Physiological windows in milliseconds.

    The windows are feature definitions, not labels.  A separate audit of a
    model's learned latency should compare these fixed definitions with the
    model's own intermediate representation rather than feeding model
    latency back into feature extraction.
    """

    baseline: tuple[float, float] = (-200.0, 0.0)
    n2: tuple[float, float] = (150.0, 300.0)
    p3a: tuple[float, float] = (250.0, 430.0)
    p3b: tuple[float, float] = (300.0, 650.0)
    analysis: tuple[float, float] = (0.0, 800.0)

    def __post_init__(self) -> None:
        for name, window in (
            ("baseline", self.baseline),
            ("n2", self.n2),
            ("p3a", self.p3a),
            ("p3b", self.p3b),
            ("analysis", self.analysis),
        ):
            if len(window) != 2 or not np.all(np.isfinite(window)) or window[0] >= window[1]:
                raise AuditInputError(f"{name} window must be two finite values with lo < hi.")

    def component_items(self) -> tuple[tuple[str, tuple[float, float]], ...]:
        return (("n2", self.n2), ("p3a", self.p3a), ("p3b", self.p3b))


@dataclass(frozen=True)
class P300FeatureConfig:
    windows: P300Windows = P300Windows()
    sampling_rate_hz: float | None = None
    baseline_policy: Literal["strict", "first_window"] = "strict"
    min_window_samples: int = 8
    complexity_max_points: int = 128
    filter_order: int = 3

    def __post_init__(self) -> None:
        if self.sampling_rate_hz is not None and self.sampling_rate_hz <= 2:
            raise AuditInputError("sampling_rate_hz must be greater than 2Hz.")
        if self.min_window_samples < 4:
            raise AuditInputError("min_window_samples must be at least 4.")
        if self.complexity_max_points < 8:
            raise AuditInputError("complexity_max_points must be at least 8.")
        if self.filter_order < 1:
            raise AuditInputError("filter_order must be positive.")


@dataclass(frozen=True)
class _FeatureSpec:
    name: str
    family: str
    description: str
    compute: Callable[[dict[str, np.ndarray]], np.ndarray]


def _window_indices(
    time_ms: np.ndarray,
    window: tuple[float, float],
    *,
    name: str,
    min_samples: int,
) -> np.ndarray:
    lo, hi = window
    if not lo < hi:
        raise AuditInputError(f"{name} window must have lo < hi.")
    idx = np.flatnonzero((time_ms >= lo) & (time_ms < hi))
    if idx.size < min_samples:
        raise AuditInputError(
            f"{name} window {window}ms contains {idx.size} samples; "
            f"at least {min_samples} are required."
        )
    return idx


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(np.abs(denominator), np.finfo(float).eps)


def _safe_log_power(power: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(power, np.finfo(float).tiny))


def _mean_channels(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return values[:, indices].mean(axis=1)


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=-1, keepdims=True)
    b0 = b - b.mean(axis=-1, keepdims=True)
    numerator = np.sum(a0 * b0, axis=-1)
    denominator = np.sqrt(np.sum(a0 * a0, axis=-1) * np.sum(b0 * b0, axis=-1))
    return numerator / np.maximum(denominator, np.finfo(float).eps)


def _sample_entropy(signal: np.ndarray, order: int = 2) -> float:
    """Small, deterministic sample entropy implementation for one vector."""

    x = np.asarray(signal, dtype=float).reshape(-1)
    if x.size <= order + 2:
        return 0.0
    scale = max(float(np.std(x)) * 0.2, np.finfo(float).eps)

    def count_matches(m: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(x, m)
        if templates.shape[0] < 2:
            return 0
        count = 0
        for i in range(templates.shape[0] - 1):
            distance = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.count_nonzero(distance <= scale))
        return count

    count_m = count_matches(order)
    count_m1 = count_matches(order + 1)
    if count_m == 0 or count_m1 == 0:
        return float(-np.log(1.0 / max(x.size, 2)))
    return float(-np.log(count_m1 / count_m))


def _permutation_entropy(signal: np.ndarray, order: int = 3) -> float:
    x = np.asarray(signal, dtype=float).reshape(-1)
    if x.size < order + 1:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(x, order)
    patterns = np.argsort(windows, axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)))
    return float(entropy / np.log(factorial(order)))


def _band_envelope(
    signal: np.ndarray, fs: float, low: float, high: float, order: int
) -> np.ndarray:
    """Return Hilbert envelope with a short-epoch-safe zero-phase filter."""

    n = signal.shape[-1]
    nyquist = fs / 2.0
    low = max(float(low), 0.1)
    high = min(float(high), nyquist * 0.95)
    if high <= low or n < 16:
        return np.abs(hilbert(signal, axis=-1))
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    padlen = min(3 * (2 * sos.shape[0] + 1), n - 1)
    filtered = sosfiltfilt(sos, signal, axis=-1, padlen=padlen)
    return np.abs(hilbert(filtered, axis=-1))


def _pac_modulation_index(phase: np.ndarray, amplitude: np.ndarray, bins: int = 18) -> float:
    phase = np.asarray(phase, dtype=float).reshape(-1)
    amplitude = np.asarray(amplitude, dtype=float).reshape(-1)
    if phase.size != amplitude.size or phase.size < bins * 2:
        return 0.0
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    means = np.zeros(bins, dtype=float)
    for i in range(bins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if np.any(mask):
            means[i] = float(np.mean(amplitude[mask]))
    if means.sum() <= np.finfo(float).eps:
        return 0.0
    probabilities = means / means.sum()
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)))
    return float((np.log(bins) - entropy) / np.log(bins))


class P300FeatureLexicon:
    """Compute the fixed 63-feature P300 audit vocabulary."""

    def __init__(self, config: P300FeatureConfig | None = None):
        self.config = config or P300FeatureConfig()
        if self.config.baseline_policy not in ("strict", "first_window"):
            raise AuditInputError(f"Unsupported baseline_policy {self.config.baseline_policy!r}.")
        if self.config.min_window_samples < 4:
            raise AuditInputError("min_window_samples must be at least 4.")
        self.specs = self._build_registry()
        if len(self.specs) != 63:
            raise RuntimeError(
                f"Internal P300 feature registry must contain 63 features, got {len(self.specs)}."
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(spec.family for spec in self.specs)

    @property
    def descriptions(self) -> tuple[str, ...]:
        return tuple(spec.description for spec in self.specs)

    def extract(self, data: P300AuditData) -> FeatureTable:
        """Extract all registry features without reading labels or predictions."""

        fs = self._resolve_sampling_rate(data.time_ms)
        indices = self._resolve_indices(data.time_ms)
        primitives = self._build_primitives(data.X, data.time_ms, indices, fs, data.channel_names)
        columns = []
        for spec in self.specs:
            value = np.asarray(spec.compute(primitives), dtype=float).reshape(-1)
            if value.shape[0] != data.n_trials:
                raise RuntimeError(f"Feature {spec.name!r} returned {value.shape[0]} rows.")
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"Feature {spec.name!r} produced NaN or infinite values.")
            columns.append(value)
        return FeatureTable(
            values=np.column_stack(columns),
            names=self.names,
            families=self.families,
            descriptions=self.descriptions,
        )

    def _resolve_sampling_rate(self, time_ms: np.ndarray) -> float:
        if self.config.sampling_rate_hz is not None:
            fs = float(self.config.sampling_rate_hz)
        else:
            spacing = float(np.median(np.diff(time_ms)))
            if spacing <= 0:
                raise AuditInputError("Cannot infer a positive sampling rate from time_ms.")
            fs = 1000.0 / spacing
        if not np.isfinite(fs) or fs <= 2.0:
            raise AuditInputError(f"sampling rate must be >2Hz, got {fs}.")
        return fs

    def _resolve_indices(self, time_ms: np.ndarray) -> dict[str, np.ndarray]:
        windows = self.config.windows
        resolved: dict[str, np.ndarray] = {}
        try:
            resolved["baseline"] = _window_indices(
                time_ms,
                windows.baseline,
                name="baseline",
                min_samples=self.config.min_window_samples,
            )
        except AuditInputError:
            if self.config.baseline_policy == "strict":
                raise
            first_lo = float(time_ms[0])
            first_hi = first_lo + max(50.0, windows.baseline[1] - windows.baseline[0])
            resolved["baseline"] = _window_indices(
                time_ms,
                (first_lo, first_hi),
                name="fallback baseline",
                min_samples=self.config.min_window_samples,
            )
        for name, window in windows.component_items() + (("analysis", windows.analysis),):
            resolved[name] = _window_indices(
                time_ms, window, name=name, min_samples=self.config.min_window_samples
            )
        return resolved

    def _channel_groups(self, names: tuple[str, ...]) -> dict[str, np.ndarray]:
        def exact(candidates: tuple[str, ...], fallback: np.ndarray) -> np.ndarray:
            found = np.asarray([i for i, name in enumerate(names) if name in candidates], dtype=int)
            return found if found.size else fallback

        n = len(names)
        all_channels = np.arange(n, dtype=int)
        posterior = exact(("P3", "Pz", "P4", "PO7", "PO8", "Oz"), all_channels[max(0, n // 2) :])
        frontal = exact(("Fz", "F3", "F4", "Cz"), all_channels[: max(1, n // 3)])
        p3 = exact(("P3", "Pz", "P4"), posterior)
        central = exact(("Cz", "C3", "C4", "Pz"), all_channels[max(0, n // 3) : max(1, 2 * n // 3)])
        return {
            "all": all_channels,
            "frontal": frontal,
            "posterior": posterior,
            "p3": p3,
            "central": central,
        }

    def _build_primitives(
        self,
        epochs: np.ndarray,
        time_ms: np.ndarray,
        indices: dict[str, np.ndarray],
        fs: float,
        channel_names: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        n_trials, n_channels, _ = epochs.shape
        baseline = epochs[:, :, indices["baseline"]]
        baseline_mean = baseline.mean(axis=-1)
        corrected = epochs - baseline_mean[:, :, None]
        groups = self._channel_groups(channel_names)
        components = ("n2", "p3a", "p3b")
        component_mean: dict[str, np.ndarray] = {}
        component_std: dict[str, np.ndarray] = {}
        component_area: dict[str, np.ndarray] = {}
        component_peak: dict[str, np.ndarray] = {}
        component_latency: dict[str, np.ndarray] = {}
        component_ptp: dict[str, np.ndarray] = {}
        component_trace: dict[str, np.ndarray] = {}
        for component in components:
            idx = indices[component]
            trace = corrected[:, :, idx]
            component_trace[component] = trace
            component_mean[component] = trace.mean(axis=-1)
            component_std[component] = trace.std(axis=-1)
            component_area[component] = trapezoid(trace, time_ms[idx], axis=-1)
            if component == "n2":
                peak_index = np.argmin(trace, axis=-1)
            else:
                peak_index = np.argmax(trace, axis=-1)
            peak_index_expanded = peak_index[..., None]
            component_peak[component] = np.take_along_axis(trace, peak_index_expanded, axis=-1)[
                ..., 0
            ]
            component_latency[component] = time_ms[idx][peak_index]
            component_ptp[component] = trace.max(axis=-1) - trace.min(axis=-1)

        analysis = corrected[:, :, indices["analysis"]]
        frequencies, powers = welch(
            analysis,
            fs=fs,
            axis=-1,
            nperseg=min(128, analysis.shape[-1]),
            noverlap=0,
            detrend="constant",
        )
        band_limits = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
        }
        band_power: dict[str, np.ndarray] = {}
        for band, (low, high) in band_limits.items():
            mask = (frequencies >= low) & (frequencies < high)
            band_power[band] = trapezoid(powers[..., mask], frequencies[mask], axis=-1).mean(axis=1)

        p3b_trace = component_trace["p3b"]
        p3b_mean_trace = p3b_trace.mean(axis=1)
        tf_envelope: dict[str, np.ndarray] = {}
        for band, (low, high) in (
            ("delta", (1.0, 4.0)),
            ("theta", (4.0, 8.0)),
            ("alpha", (8.0, 13.0)),
            ("beta", (13.0, 30.0)),
        ):
            tf_envelope[band] = _band_envelope(
                p3b_mean_trace, fs, low, high, self.config.filter_order
            )

        post = analysis.mean(axis=1)
        if post.shape[-1] > self.config.complexity_max_points:
            sampled = np.linspace(0, post.shape[-1] - 1, self.config.complexity_max_points).astype(
                int
            )
            complexity_trace = post[:, sampled]
        else:
            complexity_trace = post
        hjorth_activity = complexity_trace.var(axis=-1)
        diff1 = np.diff(complexity_trace, axis=-1)
        diff2 = np.diff(complexity_trace, n=2, axis=-1)
        mobility = np.sqrt(_safe_ratio(diff1.var(axis=-1), hjorth_activity))
        complexity = np.sqrt(
            _safe_ratio(diff2.var(axis=-1), np.maximum(diff1.var(axis=-1), np.finfo(float).eps))
        )
        line_length = np.mean(np.abs(diff1), axis=-1)
        centered = complexity_trace - complexity_trace.mean(axis=-1, keepdims=True)
        zero_crossings = np.mean(centered[:, 1:] * centered[:, :-1] < 0, axis=-1)
        sample_entropy = np.asarray([_sample_entropy(row) for row in complexity_trace])
        permutation_entropy = np.asarray([_permutation_entropy(row) for row in complexity_trace])
        kurtosis = (
            np.mean(centered**4, axis=-1)
            / np.maximum(np.mean(centered**2, axis=-1) ** 2, np.finfo(float).eps)
            - 3.0
        )

        pac_pairs = (
            ("theta", "delta"),
            ("theta", "alpha"),
            ("theta", "beta"),
            ("alpha", "beta"),
            ("delta", "theta"),
        )
        pac_values = []
        analysis_mean = analysis.mean(axis=1)
        for phase_band, amp_band in pac_pairs:
            phase_signal = _band_envelope(
                analysis_mean, fs, *band_limits[phase_band], self.config.filter_order
            )
            # _band_envelope returns magnitude; use a separately filtered signal for phase.
            phase_sos = butter(
                self.config.filter_order,
                [band_limits[phase_band][0] / (fs / 2), band_limits[phase_band][1] / (fs / 2)],
                btype="bandpass",
                output="sos",
            )
            padlen = min(3 * (2 * phase_sos.shape[0] + 1), analysis_mean.shape[-1] - 1)
            phase_signal = sosfiltfilt(phase_sos, analysis_mean, axis=-1, padlen=padlen)
            amp_signal = _band_envelope(
                analysis_mean, fs, *band_limits[amp_band], self.config.filter_order
            )
            pac_values.append(
                np.asarray(
                    [
                        _pac_modulation_index(np.angle(hilbert(p)), a)
                        for p, a in zip(phase_signal, amp_signal, strict=True)
                    ]
                )
            )

        return {
            "groups": groups,
            "baseline_mean": baseline_mean.mean(axis=1),
            "baseline_std": baseline.std(axis=-1).mean(axis=1),
            "baseline_trace": baseline.mean(axis=1),
            "post_trace": post,
            "component_mean": component_mean,
            "component_std": component_std,
            "component_area": component_area,
            "component_peak": component_peak,
            "component_latency": component_latency,
            "component_ptp": component_ptp,
            "component_trace": component_trace,
            "band_power": band_power,
            "tf_envelope": tf_envelope,
            "hjorth_activity": hjorth_activity,
            "hjorth_mobility": mobility,
            "hjorth_complexity": complexity,
            "line_length": line_length,
            "zero_crossings": zero_crossings,
            "sample_entropy": sample_entropy,
            "permutation_entropy": permutation_entropy,
            "kurtosis": kurtosis,
            "pac_values": pac_values,
            "p3b_mean_trace": p3b_mean_trace,
            "time_ms": time_ms[indices["p3b"]],
            "fs": np.asarray(fs),
        }

    def _build_registry(self) -> tuple[_FeatureSpec, ...]:
        specs: list[_FeatureSpec] = []

        def add(
            name: str,
            family: str,
            description: str,
            fn: Callable[[dict[str, np.ndarray]], np.ndarray],
        ) -> None:
            specs.append(_FeatureSpec(name, family, description, fn))

        component_regions = {"n2": "frontal", "p3a": "central", "p3b": "p3"}
        for component in ("n2", "p3a", "p3b"):
            region = component_regions[component]
            add(
                f"T.{component}.mean",
                "time",
                f"Baseline-corrected mean in the {component} window.",
                lambda p, c=component, r=region: _mean_channels(
                    p["component_mean"][c], p["groups"][r]
                ),
            )
            add(
                f"T.{component}.peak",
                "time",
                f"Signed canonical peak in the {component} window.",
                lambda p, c=component, r=region: _mean_channels(
                    p["component_peak"][c], p["groups"][r]
                ),
            )
            add(
                f"T.{component}.latency_ms",
                "time",
                f"Canonical peak latency in the {component} window.",
                lambda p, c=component, r=region: _mean_channels(
                    p["component_latency"][c], p["groups"][r]
                ),
            )
            add(
                f"T.{component}.area",
                "time",
                f"Signed area under the baseline-corrected {component} waveform.",
                lambda p, c=component, r=region: _mean_channels(
                    p["component_area"][c], p["groups"][r]
                ),
            )
            add(
                f"T.{component}.ptp",
                "time",
                f"Peak-to-peak amplitude in the {component} window.",
                lambda p, c=component, r=region: _mean_channels(
                    p["component_ptp"][c], p["groups"][r]
                ),
            )
        add("T.baseline.mean", "time", "Mean pre-stimulus baseline.", lambda p: p["baseline_mean"])
        add(
            "T.baseline.std",
            "time",
            "Mean pre-stimulus baseline standard deviation.",
            lambda p: p["baseline_std"],
        )
        add(
            "T.p3b.post_minus_baseline",
            "time",
            "P3b-window mean minus pre-stimulus mean.",
            lambda p: (
                _mean_channels(p["component_mean"]["p3b"], p["groups"]["p3"]) - p["baseline_mean"]
            ),
        )
        add(
            "T.p3b.half_peak_width_ms",
            "time",
            "Width above half canonical P3b peak relative to baseline.",
            lambda p: self._half_peak_width(p, "p3b", "p3"),
        )

        bands = ("delta", "theta", "alpha", "beta")
        for band in bands:
            add(
                f"F.{band}.log_power",
                "frequency",
                f"Log mean {band} power over the post-stimulus analysis window.",
                lambda p, b=band: _safe_log_power(p["band_power"][b]),
            )
        for numerator, denominator in (
            ("delta", "theta"),
            ("theta", "alpha"),
            ("alpha", "beta"),
            ("delta", "alpha"),
        ):
            add(
                f"F.{numerator}_over_{denominator}",
                "frequency",
                f"Log-power ratio {numerator}/{denominator}.",
                lambda p, a=numerator, b=denominator: _safe_log_power(
                    _safe_ratio(p["band_power"][a], p["band_power"][b])
                ),
            )
        for band in bands:
            add(
                f"F.{band}.p3b_over_baseline",
                "frequency",
                f"P3b-to-baseline {band} power ratio.",
                lambda p, b=band: self._band_window_ratio(p, b),
            )
        add(
            "F.spectral_centroid",
            "frequency",
            "Power-weighted spectral centroid over the analysis window.",
            lambda p: self._spectral_centroid(p),
        )
        add(
            "F.high_over_low",
            "frequency",
            "High-frequency (alpha+beta) over low-frequency (delta+theta) power.",
            lambda p: _safe_log_power(
                _safe_ratio(
                    p["band_power"]["alpha"] + p["band_power"]["beta"],
                    p["band_power"]["delta"] + p["band_power"]["theta"],
                )
            ),
        )
        add(
            "F.beta_over_delta",
            "frequency",
            "Log-power ratio beta/delta.",
            lambda p: _safe_log_power(
                _safe_ratio(p["band_power"]["beta"], p["band_power"]["delta"])
            ),
        )

        for band in ("delta", "theta", "alpha"):
            add(
                f"TF.p3b.{band}.envelope_change",
                "time_frequency",
                f"P3b {band} envelope relative to its baseline value.",
                lambda p, b=band: self._envelope_change(p, b),
            )
            add(
                f"TF.p3b.{band}.envelope_peak",
                "time_frequency",
                f"Peak P3b {band} Hilbert envelope.",
                lambda p, b=band: self._envelope_peak(p, b),
            )
            add(
                f"TF.p3b.{band}.envelope_cv",
                "time_frequency",
                f"Coefficient of variation of P3b {band} envelope.",
                lambda p, b=band: self._envelope_cv(p, b),
            )

        complexity_specs = (
            ("hjorth_activity", "Hjorth activity"),
            ("hjorth_mobility", "Hjorth mobility"),
            ("hjorth_complexity", "Hjorth complexity"),
            ("line_length", "Mean line length"),
            ("zero_crossings", "Zero-crossing rate"),
            ("sample_entropy", "Sample entropy"),
            ("permutation_entropy", "Permutation entropy"),
            ("kurtosis", "Excess kurtosis"),
        )
        for key, description in complexity_specs:
            add(f"C.{key}", "complexity", description, lambda p, k=key: p[k])

        pac_names = ("theta_delta", "theta_alpha", "theta_beta", "alpha_beta", "delta_theta")
        for i, name in enumerate(pac_names):
            add(
                f"X.PAC.{name}",
                "cross_frequency",
                f"Tort-style phase-amplitude coupling ({name}).",
                lambda p, j=i: p["pac_values"][j],
            )

        add(
            "R.p3b.front_posterior_difference",
            "cross_channel",
            "P3b posterior mean minus frontal mean.",
            lambda p: (
                _mean_channels(p["component_mean"]["p3b"], p["groups"]["posterior"])
                - _mean_channels(p["component_mean"]["p3b"], p["groups"]["frontal"])
            ),
        )
        add(
            "R.p3b.topography_std",
            "cross_channel",
            "Across-channel standard deviation of P3b mean amplitudes.",
            lambda p: p["component_mean"]["p3b"].std(axis=1),
        )
        add(
            "R.p3b.topography_max",
            "cross_channel",
            "Maximum channel P3b mean amplitude.",
            lambda p: p["component_mean"]["p3b"].max(axis=1),
        )
        add(
            "R.p3b.pz_cz_corr",
            "cross_channel",
            "P3b waveform correlation between Pz/P3 and Cz/central fallback.",
            lambda p: _corr(
                p["component_trace"]["p3b"][:, p["groups"]["p3"][0]],
                p["component_trace"]["p3b"][:, p["groups"]["central"][0]],
            ),
        )
        add(
            "R.p3b.front_posterior_ratio",
            "cross_channel",
            "Absolute posterior-to-frontal P3b amplitude ratio.",
            lambda p: _safe_ratio(
                np.abs(_mean_channels(p["component_mean"]["p3b"], p["groups"]["posterior"])),
                np.abs(_mean_channels(p["component_mean"]["p3b"], p["groups"]["frontal"])),
            ),
        )
        add(
            "R.p3b.central_posterior_difference",
            "cross_channel",
            "P3b posterior mean minus central mean.",
            lambda p: (
                _mean_channels(p["component_mean"]["p3b"], p["groups"]["posterior"])
                - _mean_channels(p["component_mean"]["p3b"], p["groups"]["central"])
            ),
        )
        add(
            "R.p3b.spatial_energy",
            "cross_channel",
            "Root-mean-square P3b topography energy.",
            lambda p: np.sqrt(np.mean(p["component_mean"]["p3b"] ** 2, axis=1)),
        )
        return tuple(specs)

    @staticmethod
    def _half_peak_width(p: dict[str, np.ndarray], component: str, region: str) -> np.ndarray:
        trace = p["component_trace"][component][:, p["groups"][region], :].mean(axis=1)
        baseline = p["baseline_mean"]
        peak = p["component_peak"][component][:, p["groups"][region]].mean(axis=1)
        if component == "n2":
            mask = trace <= baseline[:, None] + 0.5 * (peak - baseline)[:, None]
        else:
            mask = trace >= baseline[:, None] + 0.5 * (peak - baseline)[:, None]
        return mask.sum(axis=1) * float(np.median(np.diff(p["time_ms"])))

    @staticmethod
    def _spectral_centroid(p: dict[str, np.ndarray]) -> np.ndarray:
        low = p["band_power"]["delta"]
        theta = p["band_power"]["theta"]
        alpha = p["band_power"]["alpha"]
        beta = p["band_power"]["beta"]
        return _safe_ratio(
            2.5 * low + 6.0 * theta + 10.5 * alpha + 21.5 * beta, low + theta + alpha + beta
        )

    @staticmethod
    def _band_window_ratio(p: dict[str, np.ndarray], band: str) -> np.ndarray:
        # The full-epoch Welch powers are already the stable quantity.  This
        # ratio uses the P3b envelope energy against its first 100ms, keeping
        # the definition well behaved for tmin=0 datasets without pretending
        # that a missing pre-stimulus baseline exists.
        envelope = p["tf_envelope"][band]
        n = envelope.shape[-1]
        split = max(2, min(n // 3, n - 1))
        return _safe_ratio(envelope[:, split:].mean(axis=1), envelope[:, :split].mean(axis=1))

    @staticmethod
    def _envelope_change(p: dict[str, np.ndarray], band: str) -> np.ndarray:
        envelope = p["tf_envelope"][band]
        split = max(2, min(envelope.shape[-1] // 3, envelope.shape[-1] - 1))
        return _safe_ratio(
            envelope[:, split:].mean(axis=1) - envelope[:, :split].mean(axis=1),
            envelope[:, :split].mean(axis=1),
        )

    @staticmethod
    def _envelope_peak(p: dict[str, np.ndarray], band: str) -> np.ndarray:
        return p["tf_envelope"][band].max(axis=1)

    @staticmethod
    def _envelope_cv(p: dict[str, np.ndarray], band: str) -> np.ndarray:
        envelope = p["tf_envelope"][band]
        return _safe_ratio(envelope.std(axis=1), envelope.mean(axis=1))
