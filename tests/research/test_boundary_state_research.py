"""Independent counterexamples for the research design, not public-data efficacy."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.signal import butter

from data.finite_filter_operator import FiniteFilterOperator

MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments/boundary_state_research.py"
SPEC = importlib.util.spec_from_file_location("boundary_state_research", MODULE_PATH)
research = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = research
SPEC.loader.exec_module(research)


def _fixture():
    rng = np.random.default_rng(548734)
    n, m, rank = 60, 40, 4
    design = rng.normal(size=(n, rank))
    truth = rng.normal(size=(rank, 1))
    temporal = toeplitz(.7 ** np.arange(n))
    held_temporal = toeplitz(.7 ** np.arange(m))
    y = design @ truth + np.sqrt(.4) * np.linalg.cholesky(temporal) @ rng.normal(size=(n, 1))
    candidates = rng.normal(size=(3, m, rank))
    x = candidates[1] @ truth + np.sqrt(.4) * np.linalg.cholesky(held_temporal) @ rng.normal(size=(m, 1))
    return design, y, temporal, np.array([[.4]]), candidates, x, held_temporal


def _bf(evidence):
    return evidence - evidence[0]


def test_covariance_gauge_preserves_physical_prior_and_log_bayes_factors():
    d, y, k, spatial, candidates, x, kt = _fixture()
    ordinary = research.ScaleInvariantContinuousModel.fit_scale_invariant(d, y, k, spatial)
    gauged = research.ScaleInvariantContinuousModel.fit_scale_invariant(
        d, y, k, spatial, covariance_factor=200.
    )
    np.testing.assert_allclose(ordinary.kernel_mean, gauged.kernel_mean, atol=1e-12)
    np.testing.assert_allclose(gauged.kernel_row_covariance / 200, ordinary.kernel_row_covariance, atol=1e-12)
    np.testing.assert_allclose(_bf(ordinary.log_evidence(x, candidates, kt)),
                               _bf(gauged.log_evidence(x, candidates, kt)), atol=1e-9, rtol=0)
    # Independent normal-equation calculation of the same Gaussian density.
    inv_k = np.linalg.inv(k)
    row_cov = np.linalg.inv(d.T @ inv_k @ d + np.eye(d.shape[1]))
    mean = row_cov @ d.T @ inv_k @ y
    expected = []
    for a in candidates:
        covariance = spatial.item() * (kt + a @ row_cov @ a.T)
        residual = (x - a @ mean).ravel()
        expected.append(-.5 * (residual @ np.linalg.solve(covariance, residual)
                              + np.linalg.slogdet(covariance)[1]))
    np.testing.assert_allclose(ordinary.log_evidence(x, candidates, kt), expected, atol=1e-9, rtol=0)
    # Negative control: alpha times canonical lambda keeps effective lambda=1,
    # which silently changes the physical prior while noise covariance is fixed.
    wrong = research.ScaleInvariantContinuousModel.fit_scale_invariant(
        d, y, k, spatial, ridge_relative=200., covariance_factor=200.
    )
    assert np.max(np.abs(_bf(wrong.log_evidence(x, candidates, kt))
                         - _bf(ordinary.log_evidence(x, candidates, kt)))) > 1


def test_voltage_unit_change_keeps_ridge_fixed_and_rejects_naive_noise_scaling():
    d, y, k, spatial, candidates, x, kt = _fixture()
    ordinary = research.ScaleInvariantContinuousModel.fit_scale_invariant(d, y, k, spatial)
    scale = 1e6  # V to microvolt
    converted = research.ScaleInvariantContinuousModel.fit_scale_invariant(
        d, scale * y, k, scale**2 * spatial
    )
    np.testing.assert_allclose(_bf(converted.log_evidence(scale*x, candidates, kt)),
                               _bf(ordinary.log_evidence(x, candidates, kt)), atol=1e-9, rtol=0)
    bad = research.ScaleInvariantContinuousModel.fit_scale_invariant(
        d, scale*y, k, scale**2*spatial, ridge_relative=scale**2
    )
    assert np.max(np.abs(_bf(bad.log_evidence(scale*x, candidates, kt))
                         - _bf(ordinary.log_evidence(x, candidates, kt)))) > 1


def test_multichannel_density_and_candidate_permutation_are_consistent():
    rng = np.random.default_rng(273)
    d, y = rng.normal(size=(16, 3)), rng.normal(size=(16, 2))
    spatial = np.array([[2., .3], [.3, .7]])
    model = research.ScaleInvariantContinuousModel.fit_scale_invariant(d, y, np.eye(16), spatial)
    x, candidates = rng.normal(size=(9, 2)), rng.normal(size=(3, 9, 3))
    score = model.log_evidence(x, candidates, np.eye(9))
    np.testing.assert_array_equal(model.log_evidence(x, candidates[[2, 0, 1]], np.eye(9)), score[[2, 0, 1]])
    a = candidates[0]
    temporal = np.eye(9) + a @ model.kernel_row_covariance @ a.T
    residual = x - a @ model.kernel_mean
    # Matrix-normal expression independently checks the time-major Kronecker order.
    quadratic = np.trace(np.linalg.solve(spatial, residual.T @ np.linalg.solve(temporal, residual)))
    logdet = 2*np.linalg.slogdet(temporal)[1] + 9*np.linalg.slogdet(spatial)[1]
    assert score[0] == pytest.approx(-.5*(quadratic + logdet), abs=1e-10)


def test_projector_handles_duplicate_templates_and_soft_constraint_has_exact_solution():
    rows = research.erp_row_basis([[1., 0., 0., 0.], [2., 0., 0., 0.]])
    assert rows.shape == (1, 4)
    delta = np.array([[4., 3.], [2., 1.]])
    corrected = research.protect_delta(delta, rows, penalty=3.)
    np.testing.assert_allclose(corrected, [[1., 3.], [2., 1.]])
    # Independently verify stationarity, including the protected component.
    hessian = np.eye(4) + 3*rows.T @ rows
    np.testing.assert_allclose(hessian @ corrected.ravel(), delta.ravel())
    np.testing.assert_array_equal(research.protect_delta(delta, rows, penalty=0.), delta)
    with pytest.raises(ValueError, match="nonzero"):
        research.erp_row_basis(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="orthonormal"):
        research.protect_delta(delta, rows * 2, penalty=1.)


def _operator(padding=50):
    return FiniteFilterOperator(tuple(map(tuple, butter(2, [2., 15.], fs=64., btype="bandpass", output="sos"))), padding)


def test_deterministic_state_reconstructs_exact_output_without_extra_covariance():
    operator = _operator()
    x = np.random.default_rng(83).normal(size=(121, 2))
    zero, basis, states = operator.decompose(x)
    assert basis.shape == (121, 4*len(operator.sos))
    np.testing.assert_allclose(zero + basis @ states, operator.apply(x), atol=1e-10)


def test_common_grid_uses_actual_long_context_and_rejects_unobserved_scope():
    operator = _operator()
    x = np.random.default_rng(735).normal(size=(121, 2))
    rows = research.erp_row_basis(np.ones((1, 20)))
    analysis = research.BoundaryStateAnalysis.compute(
        x, operator, operator, rows, short_scope=(20, 100), output_scope=(30, 40), sfreq=64.
    )
    assert analysis.distance_seconds == pytest.approx(10/64)
    assert analysis.reference_role == "observed_long_context_diagnostic_only"
    assert analysis.pareto_diagnostics[0]["difference_mse_improvement"] == pytest.approx(1.)
    assert analysis.pareto_diagnostics[0]["erp_increment_error"] is None
    with pytest.raises(ValueError, match="inside"):
        research.BoundaryStateAnalysis.compute(
            x, operator, operator, rows, short_scope=(20, 140), output_scope=(30, 40), sfreq=64.
        )
    # Two possible full records have identical observed short segments. Their
    # long-filter targets differ: no deterministic short-only delta can fit both.
    alternate = x.copy()
    alternate[19] += 100
    np.testing.assert_array_equal(x[20:100], alternate[20:100])
    assert np.linalg.norm(operator.apply(x)[30:40] - operator.apply(alternate)[30:40]) > .01


def test_zero_delta_is_reported_as_undefined_fraction_without_policy_selection():
    operator = _operator()
    analysis = research.BoundaryStateAnalysis.compute(
        np.zeros((100, 1)), operator, operator, research.erp_row_basis(np.ones((1, 10))),
        short_scope=(0, 100), output_scope=(20, 30), sfreq=64.
    )
    assert analysis.zero_difference and analysis.erp_fraction is None
    assert not analysis.overlap_flag
    assert all(r["difference_mse_improvement"] is None for r in analysis.pareto_diagnostics)


def test_high_erp_overlap_can_repair_erp_and_is_not_proof_of_damage():
    rows = research.erp_row_basis([[1., 0.]])
    true_erp, short = np.array([[1., 0.]]), np.array([[.8, 0.]])
    delta = true_erp - short
    assert float(np.sum((rows @ delta.ravel())**2) / np.sum(delta**2)) == 1.
    np.testing.assert_array_equal(short + delta, true_erp)
    # Conversely, a correction independent of x has zero injection derivative,
    # even if the correction value points entirely along an ERP output direction.
    injected = short + np.array([[.3, 0.]])
    np.testing.assert_allclose((injected + delta) - (short + delta), injected - short)


def test_distance_schedule_is_monotone_stable_and_not_equal_to_maximum_at_zero():
    values = [research.distance_adaptive_erp_penalty(d) for d in (0., .5, 1., 2., 8., 1e9)]
    assert np.all(np.diff(values) < 0)
    assert values[0] == pytest.approx(88.07970779778825)
    assert values[2] == 50. and values[-1] == 0.
    with pytest.raises(ValueError, match="invalid"):
        research.distance_adaptive_erp_penalty(-1.)
    with pytest.raises(ValueError, match="invalid"):
        research.distance_adaptive_erp_penalty(1., tau=0.)


def test_dense_reference_budget_and_covariance_validation():
    d, y, k, spatial, candidates, x, kt = _fixture()
    with pytest.raises(ValueError, match="budget"):
        research.ScaleInvariantContinuousModel.fit_scale_invariant(d, y, k, spatial, byte_budget=8)
    model = research.ScaleInvariantContinuousModel.fit_scale_invariant(d, y, k, spatial)
    with pytest.raises(ValueError, match="budget"):
        model.log_evidence(x, candidates, kt, byte_budget=8)
    nonsymmetric = k.copy()
    nonsymmetric[0, 1] += .1
    with pytest.raises(ValueError, match="symmetric"):
        research.ScaleInvariantContinuousModel.fit_scale_invariant(d, y, nonsymmetric, spatial)


def test_plan_refuses_wrong_protocol_before_reading_data(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="C800"):
        research.make_plan(path)
