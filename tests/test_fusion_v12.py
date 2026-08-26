from __future__ import annotations

import numpy as np
import pytest

from baselines.fusion_v12 import (
    curve_improvement,
    e_process_diagnostics,
    fit_nested_fusion,
    prequential_e_process,
    replay_chain_stopping,
    replay_chain_stopping_from_contributions,
    replay_first_crossing,
    risk_coverage_curve,
    select_v12_fusion,
    trajectory_contributions,
    two_hypothesis_conformal_flags,
)


def test_replay_contracts_reject_fractional_indices_and_integer_masks() -> None:
    trajectories = np.zeros((2, 2, 3), dtype=float)
    with pytest.raises(ValueError, match="true_candidates must have an integer dtype"):
        replay_chain_stopping(trajectories, np.array([0.0, 1.0]), threshold=0.9)
    with pytest.raises(ValueError, match="sequence_lengths must have an integer dtype"):
        replay_chain_stopping(
            trajectories,
            np.array([0, 1]),
            threshold=0.9,
            sequence_lengths=np.array([2.5, 3.0]),
        )
    with pytest.raises(ValueError, match="valid_mask must have boolean dtype"):
        replay_chain_stopping_from_contributions(
            trajectories,
            np.array([0, 1]),
            valid_mask=np.ones((2, 3), dtype=np.int64),
            threshold=0.9,
        )


def test_nested_fusion_rejects_redundant_evidence() -> None:
    rng = np.random.default_rng(0)
    n_subjects = 8
    n_per = 120
    subject = np.repeat(np.arange(n_subjects), n_per)
    y = (rng.random(len(subject)) < 0.5).astype(float)
    base = np.where(y == 1.0, 1.0, -1.0) + rng.normal(size=len(y))
    evidence = 0.8 * base + rng.normal(size=len(y))
    report = fit_nested_fusion(base, evidence, y, subject, n_bootstrap=200)
    assert report["passed"] is False
    assert report["coefficient"] == 0.0
    assert report["c_ci"][0] <= 0.0 <= report["c_ci"][1] or not np.isfinite(report["c_ci"]).all()


def test_nested_fusion_accepts_incremental_evidence() -> None:
    rng = np.random.default_rng(1)
    n_subjects = 8
    n_per = 200
    subject = np.repeat(np.arange(n_subjects), n_per)
    y = (rng.random(len(subject)) < 0.5).astype(float)
    signal = np.where(y == 1.0, 1.0, -1.0)
    base = signal + rng.normal(size=len(y))
    unique = signal + rng.normal(scale=0.8, size=len(y))
    evidence = 0.3 * base + unique
    report = fit_nested_fusion(base, evidence, y, subject, n_bootstrap=200)
    assert report["passed"] is True
    assert report["coefficient"] > 0.0
    assert report["nll_improvement"] >= 0.005
    assert report["auc_non_inferior"] is True


def test_replay_first_crossing_metrics() -> None:
    sequences = np.asarray([[0.1, 1.5, 2.0], [-1.0, -0.2, 0.0], [0.2, 0.3, 0.4]])
    labels = np.asarray([1, 0, 1])
    report = replay_first_crossing(sequences, labels, threshold=1.0)
    assert report["decided_fraction"] == 1 / 3
    assert report["empirical_error"] == 0.0
    assert report["expected_flashes"] == 2.0


def test_replay_chain_stopping_accepts_variable_sequence_lengths() -> None:
    trajectories = np.asarray(
        [
            [[5.0, -5.0, 0.0], [-5.0, 5.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    trajectories[1, 0, 1] = 5.0
    trajectories[1, 1, 1] = -5.0
    report = replay_chain_stopping(
        trajectories,
        np.asarray([0, 0]),
        threshold=0.9,
        sequence_lengths=np.asarray([1, 3]),
    )
    assert report["n_decided"] == 2
    assert report["empirical_error"] == 0.0
    assert report["expected_flashes"] == 1.5


def test_two_hypothesis_conformal_flags() -> None:
    calibration = np.linspace(0.0, 10.0, 100)
    test = np.asarray([20.0, 5.0])
    flags = two_hypothesis_conformal_flags(calibration, test, alpha=0.05)
    assert flags.tolist() == [True, False]


def test_prequential_e_process_diagnostics() -> None:
    rng = np.random.default_rng(4)
    null = rng.normal(0.0, 1.0, 4000)
    llr = null - 0.5
    report = e_process_diagnostics(llr)
    assert report["passed"] is True
    assert report["nonnegative"] is True
    assert report["final_mean"] <= 1.05
    assert report["mean_e"] <= 1.05


def test_e_process_expectation_over_independent_null_sequences() -> None:
    """blueprint S0-7: E[e]<=1 must be estimated over independent sequences."""
    rng = np.random.default_rng(7)
    null_llr = rng.normal(0.0, 1.0, (400, 80)) - 0.5
    final_e = np.asarray([prequential_e_process(row)[-1] for row in null_llr])
    assert np.all(final_e >= 0.0)
    assert float(final_e.mean()) <= 1.05


def test_negative_pre_registered_sign_is_selectable() -> None:
    """Negative c is a valid pre-registered activation when CI<0 and c<0."""
    rng = np.random.default_rng(2)
    subject = np.repeat(np.arange(8), 150)
    y = (rng.random(len(subject)) < 0.5).astype(float)
    signal = np.where(y == 1.0, 1.0, -1.0)
    base = signal + rng.normal(size=len(y))
    evidence = -0.5 * signal + rng.normal(size=len(y))
    report = fit_nested_fusion(
        base, evidence, y, subject, n_bootstrap=200, expected_sign="negative"
    )
    selected = select_v12_fusion({"v": report}, density_selected_variant="v")
    assert report["passed"] is True
    assert report["coefficient"] < 0.0
    assert selected["passed"] is True
    assert selected["coefficient"] < 0.0


def test_trajectory_contributions_and_gated_replay() -> None:
    contributions = np.zeros((2, 2, 3), dtype=np.float64)
    contributions[0, 0, 0] = 2.0
    contributions[0, 0, 1] = 2.0
    contributions[0, 1, 0] = -2.0
    contributions[0, 1, 1] = -2.0
    contributions[1, 0, 1] = 5.0
    contributions[1, 1, 1] = -5.0
    true = np.asarray([0, 0])
    valid = np.asarray([[True, True, False], [False, True, False]])
    report = replay_chain_stopping_from_contributions(
        contributions,
        true,
        valid_mask=valid,
        threshold=0.9,
        sequence_lengths=np.asarray([2, 2]),
    )
    assert report["n_decided"] == 2
    assert report["empirical_error"] == 0.0
    assert report["expected_flashes"] == 1.5

    trajectory = np.cumsum(contributions, axis=2)
    rebuilt = trajectory_contributions(trajectory)
    assert np.allclose(rebuilt, contributions)


def test_risk_coverage_curve_reports_descriptive_points() -> None:
    contributions = np.zeros((2, 2, 4), dtype=np.float64)
    contributions[0, 0, :] = 2.0
    contributions[0, 1, :] = -2.0
    contributions[1, 0, 2] = 5.0
    contributions[1, 1, 2] = -5.0
    curve = risk_coverage_curve(
        contributions,
        np.asarray([0, 0]),
        valid_mask=np.ones((2, 4), dtype=bool),
        thresholds=(0.5, 0.9),
    )
    assert len(curve) == 2
    assert all(np.isfinite(point["risk"]) for point in curve)
    assert curve[1]["coverage"] <= curve[0]["coverage"] + 1e-12


def test_curve_improvement_compares_common_coverage() -> None:
    gated = [
        {"threshold": 0.9, "risk": 0.10, "coverage": 0.9, "undecided_fraction": 0.1, "expected_flashes": 4.0},
        {"threshold": 0.5, "risk": 0.30, "coverage": 1.0, "undecided_fraction": 0.0, "expected_flashes": 2.0},
    ]
    ungated = [
        {"threshold": 0.9, "risk": 0.20, "coverage": 0.9, "undecided_fraction": 0.1, "expected_flashes": 3.0},
        {"threshold": 0.5, "risk": 0.25, "coverage": 1.0, "undecided_fraction": 0.0, "expected_flashes": 2.0},
    ]
    report = curve_improvement(gated, ungated, target_coverage=0.9)
    assert report["available"] is True
    assert report["improved"] is True
    assert report["gated_risk"] == 0.10
