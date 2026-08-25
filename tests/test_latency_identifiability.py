from __future__ import annotations

import numpy as np

from experiments.run_latency_identifiability import (
    N_TIME,
    _model_kwargs,
    _pool_fold_results,
    make_paired_latency_probe,
    make_synthetic_training_data,
    recovery_metrics,
)


def test_recovery_metrics_reports_signed_bias_and_rmse() -> None:
    metrics = recovery_metrics(np.asarray([465.0, 467.0]), 460.0)

    assert metrics["bias_ms"] == 6.0
    assert np.isclose(metrics["rmse_ms"], np.sqrt(37.0))
    assert metrics["mae_ms"] == 6.0
    assert metrics["n"] == 2


def test_pool_uses_configured_base_latency_for_tau0() -> None:
    condition = {
        "shift_ms": 0.0,
        "tau": {"bias_ms": 0.0, "rmse_ms": 0.0, "n": 1},
        "dtau": {"bias_ms": 0.0, "rmse_ms": 0.0, "n": 1},
        "paired_delta_dtau": {"bias_ms": 0.0, "rmse_ms": 0.0, "n": 1},
    }
    folds = [
        {"tau0_p3b_ms": 501.0, "conditions": {"+0": condition}},
        {"tau0_p3b_ms": 499.0, "conditions": {"+0": condition}},
    ]

    pooled = _pool_fold_results(folds, (0.0,), base_latency_ms=500.0)

    assert pooled["tau0_p3b_ms"]["bias_ms"] == 0.0
    assert pooled["tau0_p3b_ms"]["rmse_ms"] == 1.0


def test_model_kwargs_accept_deliberately_offset_p3b_tau0() -> None:
    assert _model_kwargs(p3b_tau0_init_ms=420.0)["tau0_ms"] == (220.0, 300.0, 420.0)


def test_paired_probe_reuses_noise_and_changes_only_p3b_latency() -> None:
    probes, pair_ids = make_paired_latency_probe(
        subject_ids=(2, 3),
        n_trials_per_subject=3,
        base_latency_ms=460.0,
        shifts_ms=(0.0, -20.0, 20.0, 40.0),
        noise_std=1.0,
        seed=7,
    )

    assert pair_ids.shape == (6,)
    assert all(values.shape == (6, 3, N_TIME) for values in probes.values())
    assert not np.array_equal(probes[0.0], probes[20.0])
    # Fz carries the weakest P3b injection; paired conditions should remain
    # much closer than independently generated trials.
    paired_difference = np.std(probes[20.0] - probes[0.0])
    independent_scale = np.std(probes[0.0])
    assert paired_difference < independent_scale


def test_synthetic_training_data_has_group_and_latency_contracts() -> None:
    data = make_synthetic_training_data(
        n_subjects=4,
        n_target_per_subject=3,
        n_nontarget_per_subject=5,
        base_latency_ms=460.0,
        train_jitter_ms=40.0,
        noise_std=1.0,
        seed=1,
    )

    assert data.X.shape == (32, 3, N_TIME)
    assert set(data.groups.tolist()) == {0, 1, 2, 3}
    assert np.isfinite(data.true_p3b_latency_ms[data.y == 1]).all()
    assert np.isnan(data.true_p3b_latency_ms[data.y == 0]).all()
    assert data.true_p3b_latency_ms[data.y == 1].min() >= 420.0
    assert data.true_p3b_latency_ms[data.y == 1].max() <= 500.0
