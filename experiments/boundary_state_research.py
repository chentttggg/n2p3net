"""Boundary redesign mathematics and experiment-plan scaffold, research only.

Long-context differences require actually observed context. The dense Gaussian
reference below tests covariance gauges; it does not estimate an AR model or
claim a calibrated physiological posterior. No production consumer imports it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402


def _finite(value, ndim, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty finite {ndim}-dimensional data")
    return result


def erp_row_basis(calibration_templates, *, relative_tolerance=1e-10):
    """Orthonormal rows on a declared time-major output grid, calibration only.

    Callers build templates using authorized calibration records, freeze their
    labels/support and align them to the output grid. Classifier coefficients
    and xDAWN spatial filters are not automatically such signal templates.
    """
    templates = _finite(calibration_templates, 2, "ERP templates")
    if not 0 < relative_tolerance < 1:
        raise ValueError("relative tolerance must lie in (0, 1)")
    _, singular, rows = np.linalg.svd(templates, full_matrices=False)
    if singular[0] <= np.finfo(float).tiny:
        raise ValueError("ERP templates have no identifiable nonzero direction")
    rows = rows[singular > relative_tolerance * singular[0]].copy()
    rows.setflags(write=False)
    return rows


def _erp_rows(rows, dimension):
    rows = _finite(rows, 2, "ERP row basis")
    if rows.shape[1] != dimension or not np.allclose(
        rows @ rows.T, np.eye(len(rows)), rtol=1e-10, atol=1e-12
    ):
        raise ValueError("ERP rows must be orthonormal on the exact output grid")
    return rows


def protect_delta(delta, erp_rows, *, penalty):
    """Solve min ||u-delta||^2 + penalty ||U u||^2 for orthonormal rows U.

    Output protection is distinct from K U_input = 0, which protects the
    response of a learned correction map to ERP input increments.
    """
    values = _finite(delta, 2, "delta")
    rows = _erp_rows(erp_rows, values.size)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    vector = values.reshape(-1)  # sample first, then physical channel
    return (vector - penalty / (1 + penalty) * (rows.T @ (rows @ vector))).reshape(
        values.shape
    )


def distance_adaptive_erp_penalty(distance_seconds, *, d_pivot=1., tau=.5, lambda_max=100.):
    """Unvalidated decreasing protection schedule; d is nearest output-support edge."""
    if (not np.isfinite([distance_seconds, d_pivot, tau, lambda_max]).all()
            or distance_seconds < 0 or d_pivot < 0 or tau <= 0 or lambda_max < 0):
        raise ValueError("invalid distance/transition/penalty")
    return float(lambda_max * expit((d_pivot - distance_seconds) / tau))


@dataclass(frozen=True)
class BoundaryStateAnalysis:
    """Observed-context diagnostic, without selecting a policy from held data."""

    distance_seconds: float
    erp_fraction: float | None
    overlap_flag: bool
    zero_difference: bool
    pareto_diagnostics: tuple[dict, ...]
    input_sha256: str
    reference_role: str = "observed_long_context_diagnostic_only"

    @classmethod
    def compute(cls, x_observed_full, operator_short, operator_long, erp_rows, *,
                short_scope, output_scope, sfreq, penalty_grid=(0., .01, .1, 1., 10., 100.),
                overlap_threshold=.5):
        """Compare R F_long x to R F_short C x on a common output grid.

        Scopes are absolute half-open sample intervals inside the supplied
        observed full context. Longer context is a diagnostic reference, never
        an imputed test input. Neither ERP truth nor a penalty is inferred here.
        """
        x = _finite(x_observed_full, 2, "observed full context")
        if not isinstance(operator_short, FiniteFilterOperator) or not isinstance(
            operator_long, FiniteFilterOperator
        ):
            raise ValueError("explicit finite operators required")
        if (len(short_scope) != 2 or len(output_scope) != 2
                or any(type(v) is not int for v in (*short_scope, *output_scope))):
            raise ValueError("scopes must contain integer half-open boundaries")
        begin, end = short_scope
        out_begin, out_end = output_scope
        if not 0 <= begin <= out_begin < out_end <= end <= len(x):
            raise ValueError("output must be inside short scope and observed full context")
        if not np.isfinite(sfreq) or sfreq <= 0 or not 0 < overlap_threshold < 1:
            raise ValueError("invalid sample rate or diagnostic overlap threshold")
        penalties = tuple(penalty_grid)
        if not penalties or not np.isfinite(penalties).all() or min(penalties) < 0:
            raise ValueError("nonempty finite nonnegative penalty grid required")
        short = operator_short.apply(x[begin:end])[out_begin - begin:out_end - begin]
        long = operator_long.apply(x)[out_begin:out_end]
        delta = long - short
        rows = _erp_rows(erp_rows, delta.size)
        energy = float(np.sum(delta**2))
        zero = energy <= np.finfo(float).tiny
        fraction = None if zero else float(np.sum((rows @ delta.reshape(-1))**2) / energy)
        diagnostics = []
        for penalty in penalties:
            correction = protect_delta(delta, rows, penalty=penalty)
            residual = delta - correction
            diagnostics.append({
                "penalty": float(penalty),
                "reference_difference_mse": float(np.mean(residual**2)),
                "difference_mse_improvement": None if zero else 1 - float(np.sum(residual**2)) / energy,
                "erp_aligned_correction_norm": float(np.linalg.norm(rows @ correction.reshape(-1))),
                "protection_map_condition": float(1 + penalty) if len(rows) < delta.size else 1.,
                "erp_increment_error": None,
                "white_noise_gain": None,
                "missing_metrics_reason": "requires_frozen_input_correction_map_and_independent_injections",
            })
        digest = hashlib.sha256()
        digest.update(x.tobytes())
        digest.update(rows.tobytes())
        digest.update(json.dumps([operator_short.record(), operator_long.record(),
                                  short_scope, output_scope, sfreq]).encode())
        return cls(min(out_begin - begin, end - out_end) / sfreq, fraction,
                   fraction is not None and fraction >= overlap_threshold, zero,
                   tuple(diagnostics), digest.hexdigest())


@dataclass(frozen=True)
class ScaleInvariantContinuousModel:
    """Small dense matrix-normal reference with an explicit covariance gauge.

    The canonical prior is W ~ MN(0, I / ridge_relative, Sigma). Covariance
    K -> alpha K, Sigma -> Sigma / alpha requires lambda -> lambda / alpha.
    K and Sigma must be independently frozen from calibration. This reference
    neither fits AR1/AR8 nor promotes a new physiological prior.
    """

    kernel_mean: np.ndarray
    kernel_row_covariance: np.ndarray
    channel_covariance: np.ndarray
    covariance_factor: float
    provenance: dict

    @classmethod
    def fit_scale_invariant(cls, design, samples, temporal_covariance, innovation_covariance, *,
                            ridge_relative=1., covariance_factor=1., byte_budget=8 * 1024 * 1024):
        design = _finite(design, 2, "calibration design")
        y = _finite(samples, 2, "calibration samples")
        temporal = _finite(temporal_covariance, 2, "canonical temporal covariance")
        spatial = _finite(innovation_covariance, 2, "canonical channel covariance")
        if (len(design) != len(y) or temporal.shape != (len(y), len(y))
                or spatial.shape != (y.shape[1], y.shape[1])):
            raise ValueError("design/sample/covariance grids do not match")
        if (not np.isfinite([ridge_relative, covariance_factor]).all()
                or ridge_relative <= 0 or covariance_factor <= 0):
            raise ValueError("positive finite ridge and covariance factor required")
        if type(byte_budget) is not int or byte_budget <= 0 or temporal.nbytes > byte_budget:
            raise ValueError("dense temporal reference exceeds diagnostic byte budget")
        for matrix in (temporal, spatial):
            if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0):
                raise ValueError("covariance must be symmetric")
            np.linalg.cholesky(matrix)
        root = np.linalg.cholesky(covariance_factor * temporal)
        d = solve_triangular(root, design, lower=True)
        observed = solve_triangular(root, y, lower=True)
        effective_ridge = ridge_relative / covariance_factor
        augmented = np.vstack([d, np.sqrt(effective_ridge) * np.eye(d.shape[1])])
        _, factor = np.linalg.qr(augmented, mode="reduced")
        inverse = solve_triangular(factor, np.eye(len(factor)))
        row_cov = inverse @ inverse.T
        mean = row_cov @ d.T @ observed
        arrays = [mean, row_cov, spatial / covariance_factor]
        for value in arrays:
            value.setflags(write=False)
        return cls(*arrays, covariance_factor, {
            "scope": "synthetic_or_calibration_frozen_noise_dense_reference",
            "ridge_relative": float(ridge_relative), "effective_ridge": float(effective_ridge),
            "scale_invariant": True, "gauge_rule": "lambda_divided_by_temporal_covariance_factor",
            "ar_estimated": False,
        })

    def log_evidence(self, samples, candidate_designs, temporal_covariance, *, byte_budget=8 * 1024 * 1024):
        """Exact small dense Gaussian, including candidate-dependent logdet."""
        y = _finite(samples, 2, "held samples")
        candidates = _finite(candidate_designs, 3, "candidate designs")
        temporal = _finite(temporal_covariance, 2, "held canonical temporal covariance")
        if (temporal.shape != (len(y), len(y)) or candidates.shape[1:] != (len(y), len(self.kernel_mean))
                or y.shape[1] != len(self.channel_covariance)):
            raise ValueError("candidate/sample/covariance grids do not match")
        if (type(byte_budget) is not int or byte_budget <= 0
                or 8 * y.size**2 > byte_budget):
            raise ValueError("dense predictive reference exceeds diagnostic byte budget")
        if not np.allclose(temporal, temporal.T, rtol=1e-12, atol=0):
            raise ValueError("held temporal covariance must be symmetric")
        np.linalg.cholesky(temporal)
        evidence = []
        for design in candidates:
            temporal_predictive = self.covariance_factor * temporal + design @ self.kernel_row_covariance @ design.T
            covariance = np.kron(temporal_predictive, self.channel_covariance)
            residual = (y - design @ self.kernel_mean).reshape(-1)
            root = np.linalg.cholesky(covariance)
            whitened = solve_triangular(root, residual, lower=True)
            evidence.append(-.5 * (float(whitened @ whitened) + 2 * np.log(np.diag(root)).sum()))
        return np.asarray(evidence)


def make_plan(baseline_plan):
    """Validate historical input identity without reading new test selections."""
    path = Path(baseline_plan)
    payload = path.read_bytes()
    base = json.loads(payload)
    if (base.get("authorized_labels_per_participant") != 800 or base.get("requested_fold_count") != 48
            or base.get("group_selection") != "first_six_complete_120_flash_selections"
            or base.get("sample_rate_hz") != 256 or base.get("channels") != "BNCI_native_8"):
        raise ValueError("baseline plan is not the declared C800 group protocol")
    sources = base["records"]
    if len(sources) != 8 or len({r["subject"] for r in sources}) != 8:
        raise ValueError("eight distinct physical participants required")
    for row in sources:
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("BNCI distributed MAT bytes changed")
    return {
        "schema": "n2p3_boundary_redesign_plan/1", "status": "framework_only_not_executed",
        "evidence_scope": "previously_observed_public_calibration_development",
        "baseline_plan_sha256": hashlib.sha256(payload).hexdigest(),
        "research_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": sources, "authorized_labels_per_participant": 800,
        "group_selection": base["group_selection"], "requested_fold_count": 48,
        "fit_labels_per_outer_fold": 600, "held_labels_per_outer_fold": 120,
        "unused_authorized_labels": 80, "physical_channels": base["channels"],
        "sample_rate_hz": base["sample_rate_hz"],
        "pre_roll_seconds": base["pre_roll_seconds"],
        "response_seconds": base["response_seconds"], "basis_rank": base["basis_rank"],
        "raw_rebuild_owner": "doc/assets/operator_research_20261004/run_calibration_study.py:selection_views",
        "erp_basis": "calibration_only_signal_templates_on_explicit_time_major_grid",
        "E1": {"distances_seconds": [0., .5, 1., 2., 4., 8.],
               "overlap_threshold_candidate": .5, "overlap_alone_proves_damage": False,
               "long_context_role": "diagnostic_reference_never_unknown_test_context"},
        "E2": {"ar_orders": [1, 8], "ridge_relative_grid": [.01, .1, 1., 10.],
               "arms": ["innovation_relative", "marginal_relative_existing_prior_change"],
               "gauge_check": "K_alpha_Sigma_over_alpha_lambda_over_alpha",
               "unit_check": "Y_times_s_Sigma_times_s_squared_same_lambda",
               "noise_estimation": "existing_research_OperatorGLS_frozen_calibration_only",
               "metrics": ["requested_hit@all", "coverage", "conditional_NLL", "maximum_confidence"]},
        "E3": {"distances_seconds": [0., .5, 1., 2., 4., 8.],
               "penalties": [0., .01, .1, 1., 10., 100.],
               "adaptive_schedule": {"d_pivot": 1., "tau": .5, "lambda_max": 100., "validated": False},
               "required_metrics": ["difference_MSE", "ERP_increment_error", "white_noise_gain", "condition_number"]},
        "selection": "inner_training_groups_only_or_fixed_grid_report_all_no_held_selection_choice",
        "unavailable_geometry": "keep_requested_case_with_reason_do_not_read_outside_C800",
        "acceptance": {"log_BF_gauge_atol": 1e-8, "ERP_increment_relative_error_candidate": .05,
                       "hit_floor_candidate": .75, "NLL_smoke_limit": float(2 * np.log(36)),
                       "promotion": "requires_paired_anchor_differences_coverage_uncertainty_and_independent_public_confirmation"},
        "kalman": "deferred_pending_observability_and_prediction_vs_retrospective_smoothing_contract",
        "full_E1_E2_E3_executed": False, "production_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(args.baseline_plan)
    # Never overwrite a previous plan/result or its execution identity.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "participants": len(result["records"]), "full_experiments_executed": False}))


if __name__ == "__main__":
    main()
