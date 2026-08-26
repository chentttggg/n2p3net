"""Fold-local, subject-balanced calibration of structured ERP windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from models.component_window import PCW_CANONICAL_SIGMA_BOUNDS
from models.time_axis import EpochTimeAxis

SEARCH_WINDOWS_MS = {
    "N2": (100.0, 350.0),
    "P3a": (180.0, 420.0),
    "P3b": (250.0, 700.0),
}
_FWHM_TO_SIGMA = 2.355
_DTAU_HI_MS = (30.0, 0.0, 150.0)
_MIN_COMPONENT_GAP_MS = 1.0
_ORDERING_EPS_MS = 1e-3


def _window_indices(t_ms: np.ndarray, lo: float, hi: float) -> np.ndarray:
    indices = np.flatnonzero((t_ms >= lo) & (t_ms <= hi))
    if indices.size == 0:
        raise ValueError(f"ERP search window [{lo},{hi}] ms is outside the epoch.")
    return indices


def _fwhm(curve: np.ndarray, t_ms: np.ndarray, peak_index: int) -> float:
    amplitude = float(curve[peak_index])
    if abs(amplitude) < 1e-12:
        return 0.0
    half = amplitude / 2.0
    sign = np.sign(amplitude)
    left = peak_index
    while left > 0 and (curve[left] - half) * sign > 0:
        left -= 1
    right = peak_index
    while right < len(curve) - 1 and (curve[right] - half) * sign > 0:
        right += 1
    return float(t_ms[right] - t_ms[left])


def _ordered_tau_bounds(
    bounds: Sequence[tuple[float, float]],
    tau0_ms: Sequence[float],
    *,
    tmin_ms: float,
    tmax_ms: float,
) -> tuple[tuple[float, float], ...]:
    """Make fold-calibrated tau bounds feasible for ComponentWindow ordering.

    ERP peak estimates can put the N2 and fallback P3a windows close together.
    The PCW delta-tau bounds still require the upper absolute latency bounds to
    admit N2 < P3a < P3b. Widen only later-component upper bounds when possible;
    this preserves the calibrated centers and lower bounds.
    """

    if len(bounds) != 3 or len(tau0_ms) != 3:
        raise ValueError("ERP calibration requires exactly three component bounds and centers.")
    lower = np.asarray([pair[0] for pair in bounds], dtype=float)
    upper = np.asarray([pair[1] for pair in bounds], dtype=float)
    centers = np.asarray(tau0_ms, dtype=float)
    if not np.isfinite(np.concatenate((lower, upper, centers))).all():
        raise ValueError("ERP calibration produced non-finite component bounds.")

    lower = np.maximum(lower, float(tmin_ms))
    upper = np.minimum(upper, float(tmax_ms))
    for index in range(3):
        if lower[index] > centers[index] or centers[index] > upper[index]:
            raise ValueError("ERP calibration component center lies outside its calibrated bounds.")

    for index in (1, 2):
        required_upper = (
            upper[index - 1]
            + _DTAU_HI_MS[index - 1]
            + _MIN_COMPONENT_GAP_MS
            + _ORDERING_EPS_MS
            - _DTAU_HI_MS[index]
        )
        if required_upper > float(tmax_ms):
            raise ValueError(
                "ERP calibration cannot produce ordered PCW latency bounds inside the epoch."
            )
        upper[index] = max(upper[index], required_upper)

    return tuple((float(lower[index]), float(upper[index])) for index in range(3))


def calibrate_erp_fold(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    *,
    time_axis: EpochTimeAxis,
    channel_names: Sequence[str],
    sigma_bounds: Sequence[tuple[float, float]] | None = None,
    trial_channel_mask: np.ndarray | None = None,
) -> dict:
    """Estimate ERP window priors from one outer-fold training partition only.

    Each subject contributes one target-minus-nontarget curve, and the fold
    curve is their median. This avoids weighting subjects by their retained
    trial counts. No artificial baseline is subtracted for epochs starting at
    stimulus onset.
    """

    X = np.asarray(X, dtype=np.float64)
    y_raw = np.asarray(y)
    subject_ids = np.asarray(subject_ids)
    if X.ndim != 3 or X.shape[2] != time_axis.n_time:
        raise ValueError(f"X must be (N,C,{time_axis.n_time}), got {X.shape}.")
    if y_raw.shape != (len(X),) or subject_ids.shape != (len(X),):
        raise ValueError("X/y/subject_ids lengths must match.")
    if not np.issubdtype(y_raw.dtype, np.integer) or set(np.unique(y_raw).tolist()) != {0, 1}:
        raise ValueError("ERP calibration requires integer binary labels {0,1}.")
    y = y_raw.astype(bool)
    if len(channel_names) != X.shape[1]:
        raise ValueError("channel_names length must match X channels.")
    if not np.isfinite(X).all():
        raise ValueError("ERP calibration requires finite epochs.")
    if trial_channel_mask is None:
        observed = np.ones(X.shape[:2], dtype=bool)
    else:
        observed = np.asarray(trial_channel_mask)
        if observed.dtype != np.dtype(bool) or observed.shape != X.shape[:2]:
            raise ValueError("trial_channel_mask must be a boolean (N,C) array.")
        if not bool(observed.any(axis=1).all()):
            raise ValueError("Every ERP calibration trial must retain an observed channel.")
        if bool((X[~observed] != 0.0).any()):
            raise ValueError("ERP calibration inputs must be zero where channels are masked.")

    sigma_contract = tuple(
        (float(lo), float(hi))
        for lo, hi in (sigma_bounds if sigma_bounds is not None else PCW_CANONICAL_SIGMA_BOUNDS)
    )
    if len(sigma_contract) != 3 or any(lo >= hi for lo, hi in sigma_contract):
        raise ValueError(
            "sigma_bounds must contain three increasing (lo, hi) component contracts."
        )

    t_ms = time_axis.samples_ms()
    baseline = t_ms < 0.0
    subject_curves = []
    used_subjects = []
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        if not (np.any(mask & y) and np.any(mask & ~y)):
            continue
        Xs = X[mask].copy()
        observed_subject = observed[mask]
        if baseline.any():
            Xs -= Xs[:, :, baseline].mean(axis=2, keepdims=True)
        ys = y[mask]
        class_means: list[np.ndarray] = []
        class_counts: list[np.ndarray] = []
        for label in (False, True):
            selected = ys == label
            counts = observed_subject[selected].sum(axis=0)
            sums = (
                Xs[selected] * observed_subject[selected, :, None]
            ).sum(axis=0)
            mean = np.full((X.shape[1], X.shape[2]), np.nan, dtype=np.float64)
            np.divide(sums, counts[:, None], out=mean, where=counts[:, None] > 0)
            class_means.append(mean)
            class_counts.append(counts)
        valid_channels = (class_counts[0] > 0) & (class_counts[1] > 0)
        if not bool(valid_channels.any()):
            continue
        curve = class_means[1] - class_means[0]
        curve[~valid_channels] = np.nan
        subject_curves.append(curve)
        used_subjects.append(subject)
    if len(subject_curves) < 2:
        raise ValueError(
            "ERP calibration needs at least two training subjects containing both classes."
        )

    stacked_curves = np.stack(subject_curves)
    channel_curves: list[np.ndarray] = []
    for channel in range(X.shape[1]):
        valid_subjects = np.isfinite(stacked_curves[:, channel]).all(axis=1)
        if bool(valid_subjects.any()):
            channel_curves.append(
                np.median(stacked_curves[valid_subjects, channel], axis=0)
            )
    if not channel_curves:
        raise ValueError("ERP calibration has no channel observed in both classes.")
    curve = np.median(np.stack(channel_curves), axis=0)
    p3b_idx = _window_indices(t_ms, *SEARCH_WINDOWS_MS["P3b"])
    i_p3b = int(p3b_idx[np.argmax(curve[p3b_idx])])
    p3b_amp = float(curve[i_p3b])
    p3b_tau = float(t_ms[i_p3b])
    p3b_fwhm = _fwhm(curve, t_ms, i_p3b)
    p3b_sigma = float(
        np.clip(p3b_fwhm / _FWHM_TO_SIGMA, sigma_contract[2][0], sigma_contract[2][1])
    )

    n2_idx = _window_indices(t_ms, *SEARCH_WINDOWS_MS["N2"])
    i_n2 = int(n2_idx[np.argmin(curve[n2_idx])])
    n2_amp = float(curve[i_n2])
    has_n2 = abs(n2_amp) >= 0.4 * max(abs(p3b_amp), 1e-12)
    n2_tau = float(t_ms[i_n2]) if has_n2 else 220.0
    n2_sigma = (
        float(
            np.clip(
                _fwhm(curve, t_ms, i_n2) / _FWHM_TO_SIGMA,
                sigma_contract[0][0],
                sigma_contract[0][1],
            )
        )
        if has_n2
        else float((sigma_contract[0][0] + sigma_contract[0][1]) / 2.0)
    )

    p3a_tau, p3a_sigma, has_p3a = (
        300.0,
        float((sigma_contract[1][0] + sigma_contract[1][1]) / 2.0),
        False,
    )
    if has_n2 and i_n2 < i_p3b and p3b_tau - t_ms[i_n2] > 150.0:
        segment = curve[i_n2:i_p3b]
        if len(segment) > 8:
            i_local = i_n2 + int(np.argmax(segment))
            prominence = curve[i_local] - max(curve[i_n2], 0.0)
            if t_ms[i_local] <= p3b_tau - 60.0 and prominence > 0.3 * max(p3b_amp, 0.0):
                p3a_tau = float(t_ms[i_local])
                p3a_sigma = float(
                    np.clip(
                        _fwhm(curve, t_ms, i_local) / _FWHM_TO_SIGMA,
                        sigma_contract[1][0],
                        sigma_contract[1][1],
                    )
                )
                has_p3a = True

    def tau_bounds(tau: float, sigma: float) -> tuple[float, float]:
        margin = max(80.0, 1.5 * sigma)
        return (
            max(float(time_axis.tmin_ms), tau - margin),
            min(float(time_axis.tmax_ms), tau + margin),
        )

    def sigma_bounds_for(sigma: float, lo: float, hi: float) -> tuple[float, float]:
        """Keep fold-calibrated learned sigma bounds inside the supplied contract."""
        return (
            max(float(lo), sigma / 2.0),
            min(float(hi), 2.0 * sigma),
        )

    sigmas = (n2_sigma, p3a_sigma, p3b_sigma)
    tau0_ms = (n2_tau, p3a_tau, p3b_tau)
    tau_bounds = _ordered_tau_bounds(
        tuple(tau_bounds(t, s) for t, s in zip(tau0_ms, sigmas, strict=True)),
        tau0_ms,
        tmin_ms=time_axis.tmin_ms,
        tmax_ms=time_axis.tmax_ms,
    )
    return {
        "calibration_scope": "outer_train_inner_subtrain",
        "n_trials": int(len(X)),
        "n_subjects": int(len(used_subjects)),
        "tau0_ms": tau0_ms,
        "tau0_bounds": tau_bounds,
        "sigma_bounds": tuple(
            sigma_bounds_for(sigma, lo, hi)
            for sigma, (lo, hi) in zip(sigmas, sigma_contract, strict=True)
        ),
        "sigma_contract": sigma_contract,
        "prior_source": "fold_calibration",
        "evidence": {
            "p3b": {
                "peak_ms": p3b_tau,
                "amplitude": p3b_amp,
                "fwhm_ms": p3b_fwhm,
                "sigma_ms": p3b_sigma,
            },
            "n2": {"detected": bool(has_n2), "peak_ms": n2_tau if has_n2 else None},
            "p3a": {"detected": bool(has_p3a), "peak_ms": p3a_tau if has_p3a else None},
        },
    }


@dataclass(frozen=True)
class FoldERPCalibrator:
    """Pickle/deepcopy-friendly callable used by baseline adapters.

    ``sigma_bounds`` is the canonical adult contract by default. Datasets
    with a documented broad component (GTN children) must pass their named
    override explicitly; they must not change the module default.
    """

    time_axis: EpochTimeAxis
    channel_names: tuple[str, ...]
    sigma_bounds: tuple[tuple[float, float], ...] = PCW_CANONICAL_SIGMA_BOUNDS
    channel_mask: tuple[bool, ...] | None = None

    accepts_trial_channel_mask = True

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> dict:
        effective_mask = trial_channel_mask
        if self.channel_mask is not None:
            static = np.asarray(self.channel_mask, dtype=bool)
            if static.shape != (X.shape[1],):
                raise ValueError("FoldERPCalibrator channel_mask must match X channels.")
            if effective_mask is None:
                effective_mask = np.broadcast_to(static, X.shape[:2])
            else:
                supplied = np.asarray(effective_mask)
                if supplied.dtype != np.dtype(bool) or supplied.shape != X.shape[:2]:
                    raise ValueError("trial_channel_mask must be a boolean (N,C) array.")
                effective_mask = supplied & static[None]
        return calibrate_erp_fold(
            X,
            y,
            subject_ids,
            time_axis=self.time_axis,
            channel_names=self.channel_names,
            sigma_bounds=self.sigma_bounds,
            trial_channel_mask=effective_mask,
        )
