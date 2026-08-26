from __future__ import annotations

import numpy as np
import pytest

from experiments.run_latency_identifiability import (
    SFREQ,
    _time_ms,
    make_synthetic_training_data,
)
from measurement.latency_measurement import LatencyMeasurement, detached_expected_window


def _target_subset(data, rows):
    return data.X[rows], data.y[rows], data.groups[rows], data.true_p3b_latency_ms[rows]


def _make_fitted_estimator(seed: int = 0) -> tuple[LatencyMeasurement, dict]:
    data = make_synthetic_training_data(
        n_subjects=6,
        n_target_per_subject=30,
        n_nontarget_per_subject=30,
        base_latency_ms=460.0,
        train_jitter_ms=40.0,
        noise_std=0.5,
        seed=seed,
    )
    estimator = LatencyMeasurement(
        anchor_tau0_ms=460.0,
        sfreq=SFREQ,
        time_ms=_time_ms(),
        grid_radius_ms=60.0,
        grid_step_ms=0.5,
        woody_iterations=3,
    ).fit(data.X, data.y, data.groups)
    return estimator, {"data": data}


def test_latency_measurement_passes_synthetic_gate() -> None:
    estimator, bundle = _make_fitted_estimator(seed=0)
    data = bundle["data"]
    target = np.flatnonzero(data.y == 1)
    target_subjects = np.unique(data.groups[target])
    calibration_subjects = target_subjects[: len(target_subjects) // 2]
    test_subjects = target_subjects[len(target_subjects) // 2 :]
    calibration = target[np.isin(data.groups[target], calibration_subjects)]
    test = target[np.isin(data.groups[target], test_subjects)]
    assert set(calibration_subjects.tolist()).isdisjoint(set(test_subjects.tolist()))

    estimator.calibrate_posterior_scale(
        data.X[calibration],
        data.true_p3b_latency_ms[calibration],
        target_coverage=0.90,
        scale_lo=0.5,
        scale_hi=12.0,
        n_candidates=81,
    )
    posterior = estimator.predict(data.X[test])
    truth = data.true_p3b_latency_ms[test]

    bias = float(np.mean(posterior.mean_ms - truth))
    rmse = float(np.sqrt(np.mean((posterior.mean_ms - truth) ** 2)))
    slope = float(np.polyfit(truth, posterior.mean_ms, 1)[0])
    coverage = float(
        np.mean((truth >= posterior.lower_ms) & (truth <= posterior.upper_ms))
    )

    assert abs(bias) < 5.0, {"bias": bias, "rmse": rmse}
    assert rmse < 10.0, {"bias": bias, "rmse": rmse}
    assert 0.9 <= slope <= 1.1, {"slope": slope}
    assert 0.85 <= coverage <= 0.95, {"coverage": coverage}


def test_latency_measurement_posterior_contract() -> None:
    estimator, bundle = _make_fitted_estimator(seed=1)
    data = bundle["data"]
    target = np.flatnonzero(data.y == 1)
    posterior = estimator.predict(data.X[target[:40]])
    posterior.validate()

    assert posterior.q.shape == (40, len(estimator.grid_ms))
    assert np.allclose(posterior.q.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(posterior.lower_ms <= posterior.mean_ms)
    assert np.all(posterior.mean_ms <= posterior.upper_ms)


def test_detached_expected_window_has_no_gradient() -> None:
    import torch

    estimator, bundle = _make_fitted_estimator(seed=3)
    data = bundle["data"]
    target = np.flatnonzero(data.y == 1)
    posterior = estimator.predict(data.X[target[:8]])
    window = torch.from_numpy(
        detached_expected_window(posterior, time_ms=_time_ms(), width_ms=50.0)
    ).to(dtype=torch.float32)
    weight = torch.ones_like(window, requires_grad=True)
    (window * weight).sum().backward()
    assert weight.grad is not None
    assert not window.requires_grad
    assert window.grad is None
    assert np.all(posterior.entropy >= 0.0)


def test_latency_population_tau0_uses_anchor() -> None:
    estimator, bundle = _make_fitted_estimator(seed=2)
    data = bundle["data"]
    target = np.flatnonzero(data.y == 1)
    posterior = estimator.predict(data.X[target])
    report = estimator.population_tau0(posterior, prior_mean_ms=460.0, prior_sd_ms=30.0)

    assert abs(report["tau0_ms"] - 460.0) < 5.0
    assert abs(report["sample_mean_ms"] - 460.0) < 5.0
    assert report["anchor_sensitivity_ms"] < 5.0


def test_latency_measurement_rejects_unfit_predict() -> None:
    estimator = LatencyMeasurement(
        anchor_tau0_ms=460.0,
        sfreq=SFREQ,
        time_ms=_time_ms(),
    )
    with pytest.raises(RuntimeError, match="must be fit"):
        estimator.predict(np.zeros((2, 3, len(_time_ms())), dtype=np.float32))


def test_nontarget_covariance_center_excludes_target_trials(monkeypatch) -> None:
    from measurement import latency_measurement as module

    captured: dict[str, np.ndarray] = {}

    class CaptureCovariance:
        def fit(self, values):
            captured["residuals"] = np.asarray(values).copy()
            self.covariance_ = np.eye(values.shape[1])
            return self

    monkeypatch.setattr(module, "LedoitWolf", CaptureCovariance)
    time_ms = np.arange(64, dtype=float) * (1000.0 / 64.0)
    waveform = np.exp(-0.5 * ((time_ms - 460.0) / 45.0) ** 2)
    X = np.zeros((10, 1, 64), dtype=np.float64)
    X[:8, 0] = 100.0 * waveform
    y = np.array([1] * 8 + [0, 0], dtype=np.int64)
    subjects = np.repeat("s", 10)
    estimator = LatencyMeasurement(
        anchor_tau0_ms=460.0,
        sfreq=64.0,
        time_ms=time_ms,
        grid_radius_ms=20.0,
        grid_step_ms=2.0,
    )

    estimator.fit(X, y, subjects)

    assert np.allclose(captured["residuals"][-2:], 0.0)
