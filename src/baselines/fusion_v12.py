"""v12 InnovationAudit and DynamicStopping helpers.

These functions replace the legacy non-negative one-parameter fusion fit and
provide the pre-registered replay/conformal primitives for the S object.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def subject_balanced_bce(logits: np.ndarray, labels: np.ndarray, subject_ids: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    subject_ids = np.asarray(subject_ids).reshape(-1)
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    per_trial = -(labels * np.log(p) + (1.0 - labels) * np.log1p(-p))
    weights = np.zeros_like(per_trial)
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        weights[mask] = 1.0 / (mask.sum() * len(np.unique(subject_ids)))
    return float((weights * per_trial).sum())


def fit_nested_fusion(
    base: np.ndarray,
    evidence: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    *,
    n_bootstrap: int = 400,
    seed: int = 0,
    expected_sign: str = "positive",
) -> dict[str, object]:
    """Nested M0:a+bS vs M1:a+bS+cL with cluster bootstrap inference."""
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    evidence = np.asarray(evidence, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    subject_ids = np.asarray(subject_ids).reshape(-1)
    if not (base.shape == evidence.shape == labels.shape == subject_ids.shape):
        raise ValueError("base/evidence/labels/subject_ids must align.")
    if expected_sign not in ("positive", "negative"):
        raise ValueError("expected_sign must be 'positive' or 'negative'.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Nested fusion requires both classes.")
    subjects = np.unique(subject_ids)
    if len(subjects) < 4:
        nll = subject_balanced_bce(base, labels, subject_ids)
        return {
            "passed": False,
            "failure": "needs_at_least_four_validation_subjects",
            "coefficient": 0.0,
            "base_slope": 1.0,
            "base_intercept": 0.0,
            "evidence_slope": 0.0,
            "base_nll": nll,
            "fused_nll": nll,
            "nll_improvement": 0.0,
            "strict_majority": False,
            "c_ci": [float("nan"), float("nan")],
            "loo_coefficients": [],
            "auc_base": float("nan"),
            "auc_fused": float("nan"),
            "auc_non_inferior": True,
            "n_subjects": int(len(subjects)),
        }
    for subject in subjects:
        if np.unique(labels[subject_ids == subject]).size != 2:
            nll = subject_balanced_bce(base, labels, subject_ids)
            return {
                "passed": False,
                "failure": "every_validation_subject_needs_both_target_classes",
                "coefficient": 0.0,
                "base_slope": 1.0,
                "base_intercept": 0.0,
                "evidence_slope": 0.0,
                "base_nll": nll,
                "fused_nll": nll,
                "nll_improvement": 0.0,
                "strict_majority": False,
                "c_ci": [float("nan"), float("nan")],
                "loo_coefficients": [],
                "auc_base": float("nan"),
                "auc_fused": float("nan"),
                "auc_non_inferior": True,
                "n_subjects": int(len(subjects)),
            }

    design_0 = base[:, None]
    design_1 = np.column_stack((base, evidence))
    weights = np.zeros(len(labels))
    for subject in subjects:
        mask = subject_ids == subject
        weights[mask] = 1.0 / (mask.sum() * len(subjects))

    # Leave-one-subject-out fits and held-out predictions.
    loo_base_logits = np.zeros_like(base)
    loo_fused_logits = np.zeros_like(base)
    loo_coefficients: list[float] = []
    subject_base_nll: dict[object, float] = {}
    subject_fused_nll: dict[object, float] = {}
    for subject in subjects:
        held_out = subject_ids == subject
        fitting = ~held_out
        m0 = LogisticRegression(C=1e6, max_iter=2000).fit(
            design_0[fitting], labels[fitting], sample_weight=weights[fitting]
        )
        m1 = LogisticRegression(C=1e6, max_iter=2000).fit(
            design_1[fitting], labels[fitting], sample_weight=weights[fitting]
        )
        loo_base_logits[held_out] = m0.decision_function(design_0[held_out])
        loo_fused_logits[held_out] = m1.decision_function(design_1[held_out])
        loo_coefficients.append(float(m1.coef_[0, 1]))
        subject_base_nll[subject] = subject_balanced_bce(
            loo_base_logits[held_out], labels[held_out], subject_ids[held_out]
        )
        subject_fused_nll[subject] = subject_balanced_bce(
            loo_fused_logits[held_out], labels[held_out], subject_ids[held_out]
        )

    base_nll = float(np.mean(list(subject_base_nll.values())))
    fused_nll = float(np.mean(list(subject_fused_nll.values())))
    strict_majority = bool(
        np.mean(
            np.asarray([subject_fused_nll[s] < subject_base_nll[s] for s in subjects])
        )
        > 0.5
    )

    # Cluster bootstrap CI for c.
    rng = np.random.default_rng(seed)
    bootstrap_c: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        rows = np.concatenate([np.flatnonzero(subject_ids == s) for s in sampled])
        if np.unique(labels[rows]).size != 2:
            continue
        bootstrap_weights = np.zeros(len(rows))
        for s in sampled:
            mask = subject_ids[rows] == s
            bootstrap_weights[mask] = 1.0 / (mask.sum() * len(sampled))
        try:
            fit = LogisticRegression(C=1e6, max_iter=2000).fit(
                design_1[rows], labels[rows], sample_weight=bootstrap_weights
            )
        except ValueError:
            continue
        bootstrap_c.append(float(fit.coef_[0, 1]))
    if not bootstrap_c:
        ci_lower = float("nan")
        ci_upper = float("nan")
    else:
        ci_lower = float(np.percentile(bootstrap_c, 2.5))
        ci_upper = float(np.percentile(bootstrap_c, 97.5))

    pooled_auc_base = roc_auc_score(labels, loo_base_logits)
    pooled_auc_fused = roc_auc_score(labels, loo_fused_logits)
    auc_non_inferior = pooled_auc_fused >= pooled_auc_base - 0.005
    nll_improvement = (base_nll - fused_nll) / max(base_nll, 1e-12)
    sign_ok = (ci_lower > 0.0) if expected_sign == "positive" else (ci_upper < 0.0)
    passed = bool(
        np.isfinite(ci_lower)
        and sign_ok
        and strict_majority
        and nll_improvement >= 0.005
        and auc_non_inferior
    )

    if passed:
        final_fit = LogisticRegression(C=1e6, max_iter=2000).fit(
            design_1, labels, sample_weight=weights
        )
        coefficient = float(final_fit.coef_[0, 1])
        base_slope = float(final_fit.coef_[0, 0])
        base_intercept = float(final_fit.intercept_[0])
        evidence_slope = float(final_fit.coef_[0, 1])
    else:
        coefficient = 0.0
        base_slope = 1.0
        base_intercept = 0.0
        evidence_slope = 0.0
    return {
        "passed": passed,
        "coefficient": coefficient,
        "base_slope": base_slope,
        "base_intercept": base_intercept,
        "evidence_slope": evidence_slope,
        "base_nll": base_nll,
        "fused_nll": fused_nll,
        "nll_improvement": nll_improvement,
        "strict_majority": strict_majority,
        "c_ci": [ci_lower, ci_upper],
        "loo_coefficients": loo_coefficients,
        "auc_base": pooled_auc_base,
        "auc_fused": pooled_auc_fused,
        "auc_non_inferior": auc_non_inferior,
        "expected_sign": expected_sign,
        "n_subjects": int(len(subjects)),
    }



def _coefficient_sign_matches(report: dict[str, object]) -> bool:
    coefficient = float(report.get("coefficient", 0.0))
    expected_sign = str(report.get("expected_sign", "positive"))
    if expected_sign == "positive":
        return coefficient > 0.0
    if expected_sign == "negative":
        return coefficient < 0.0
    return False

def select_v12_fusion(
    candidate_reports: dict[str, dict[str, object]],
    *,
    density_selected_variant: str,
) -> dict[str, object]:
    passing = [
        variant
        for variant, report in candidate_reports.items()
        if bool(report.get("passed", False)) and _coefficient_sign_matches(report)

    ]
    selected_variant = (
        min(passing, key=lambda variant: float(candidate_reports[variant]["fused_nll"]))
        if passing
        else density_selected_variant
    )
    selected = dict(candidate_reports[selected_variant])
    if not passing:
        selected["coefficient"] = 0.0
        selected["passed"] = False
        selected["failure"] = "no_audit_eligible_variant_passed_nested_fusion"
    selected.update(
        {
            "variant": selected_variant,
            "density_selected_variant": density_selected_variant,
            "candidate_reports": candidate_reports,
            "selection_source": "nested_cluster_bootstrap",
        }
    )
    return selected


def replay_first_crossing(
    log_posterior_sequences: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    """Empirical replay metrics for a fixed posterior crossing rule.

    ``log_posterior_sequences`` is ``(N, T)`` for the target candidate logit
    trajectory. The function never tunes the threshold on labels.
    """
    sequences = np.asarray(log_posterior_sequences, dtype=np.float64)
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array.")
    if not set(np.unique(labels).tolist()) <= {0, 1}:
        raise ValueError("labels must contain only binary values 0 and 1.")
    if sequences.ndim != 2 or sequences.shape[0] != len(labels):
        raise ValueError("sequences must be (N,T) and align with labels.")
    n, t = sequences.shape
    decided = np.zeros(n, dtype=bool)
    stop_index = np.full(n, t - 1, dtype=int)
    for step in range(t):
        crossing = (~decided) & (sequences[:, step] >= threshold)
        stop_index[crossing] = step
        decided |= crossing
    decided_rows = np.flatnonzero(decided)
    error = float(np.mean(labels[decided_rows] != (sequences[decided_rows, stop_index[decided_rows]] > 0)))
    return {
        "passed": True,
        "threshold": float(threshold),
        "decided_fraction": float(np.mean(decided)),
        "undecided_fraction": float(np.mean(~decided)),
        "empirical_error": error,
        "expected_flashes": float(np.mean(stop_index[decided_rows] + 1)) if decided_rows.size else float("nan"),
        "n_sequences": n,
        "n_decided": int(decided_rows.size),
    }

def replay_chain_stopping(
    trajectories: np.ndarray,
    true_candidates: np.ndarray,
    *,
    threshold: float,
    sequence_lengths: np.ndarray | None = None,
) -> dict[str, object]:
    """Replay a fixed posterior-crossing rule over candidate trajectories.

    ``trajectories`` has shape ``(N, C, T)`` and contains log candidate scores.
    ``sequence_lengths`` optionally gives the valid acquisition prefix for each
    sequence; this is needed when subjects have different numbers of flashes.
    The returned metrics are descriptive only; they never imply fixed error
    control by themselves.
    """
    trajectories = np.asarray(trajectories, dtype=np.float64)
    true_candidates = np.asarray(true_candidates)
    if not np.issubdtype(true_candidates.dtype, np.integer):
        raise ValueError("true_candidates must have an integer dtype.")
    if trajectories.ndim != 3:
        raise ValueError("trajectories must be (N,C,T).")
    n, candidates, t = trajectories.shape
    if true_candidates.shape != (n,) or not np.all((true_candidates >= 0) & (true_candidates < candidates)):
        raise ValueError("true_candidates must be valid candidate indices.")
    if sequence_lengths is None:
        sequence_lengths = np.full(n, t, dtype=np.int64)
    else:
        sequence_lengths = np.asarray(sequence_lengths)
        if not np.issubdtype(sequence_lengths.dtype, np.integer):
            raise ValueError("sequence_lengths must have an integer dtype.")
        sequence_lengths = sequence_lengths.reshape(-1)
        if sequence_lengths.shape != (n,) or not np.all((sequence_lengths >= 1) & (sequence_lengths <= t)):
            raise ValueError("sequence_lengths must contain one valid length per sequence.")
    log_posterior = trajectories - np.logaddexp.reduce(trajectories, axis=1, keepdims=True)
    decided = np.zeros(n, dtype=bool)
    stop_index = sequence_lengths - 1
    predicted = np.full(n, -1, dtype=np.int64)
    for step in range(t):
        posterior = np.exp(log_posterior[:, :, step] - np.logaddexp.reduce(log_posterior[:, :, step], axis=1, keepdims=True))
        active = sequence_lengths > step
        crossing = (~decided) & active & (posterior.max(axis=1) >= threshold)
        stop_index[crossing] = step
        predicted[crossing] = posterior[crossing].argmax(axis=1)
        decided |= crossing
    decided_rows = np.flatnonzero(decided)
    error = float(np.mean(true_candidates[decided_rows] != predicted[decided_rows])) if decided_rows.size else float("nan")
    return {
        "threshold": float(threshold),
        "decided_fraction": float(np.mean(decided)),
        "undecided_fraction": float(np.mean(~decided)),
        "empirical_error": error,
        "expected_flashes": float(np.mean(stop_index[decided_rows] + 1)) if decided_rows.size else float("nan"),
        "n_sequences": n,
        "n_decided": int(decided_rows.size),
    }


def trajectory_contributions(trajectories: np.ndarray) -> np.ndarray:
    """Convert cumulative log-scores ``(N,C,T)`` to per-step contributions."""
    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("trajectories must be (N,C,T).")
    steps = np.empty_like(values)
    steps[:, :, 0] = values[:, :, 0]
    steps[:, :, 1:] = np.diff(values, axis=2)
    return steps


def replay_chain_stopping_from_contributions(
    contributions: np.ndarray,
    true_candidates: np.ndarray,
    *,
    valid_mask: np.ndarray | None,
    threshold: float,
    sequence_lengths: np.ndarray | None = None,
) -> dict[str, object]:
    """Replay posterior crossing when individual flash contributions are masked.

    ``contributions`` has shape ``(N,C,T)``; ``valid_mask`` is boolean
    ``(N,T)`` and marks flashes that are allowed to enter the posterior.
    This is the clean-reject gate replay required by blueprint 5.3/7.
    """
    contributions = np.asarray(contributions, dtype=np.float64)
    true = np.asarray(true_candidates)
    if not np.issubdtype(true.dtype, np.integer):
        raise ValueError("true_candidates must have an integer dtype.")
    if contributions.ndim != 3:
        raise ValueError("contributions must be (N,C,T).")
    n, candidates, t = contributions.shape
    if true.shape != (n,) or not np.all((true >= 0) & (true < candidates)):
        raise ValueError("true_candidates must be valid candidate indices.")
    if sequence_lengths is None:
        sequence_lengths = np.full(n, t, dtype=np.int64)
    else:
        sequence_lengths = np.asarray(sequence_lengths)
        if not np.issubdtype(sequence_lengths.dtype, np.integer):
            raise ValueError("sequence_lengths must have an integer dtype.")
        sequence_lengths = sequence_lengths.reshape(-1)
        if sequence_lengths.shape != (n,) or not np.all((sequence_lengths >= 1) & (sequence_lengths <= t)):
            raise ValueError("sequence_lengths must be valid per sequence.")
    if valid_mask is None:
        valid_mask = np.ones((n, t), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.dtype != np.dtype(bool):
            raise ValueError("valid_mask must have boolean dtype.")
        if valid_mask.shape != (n, t):
            raise ValueError("valid_mask must be (N,T).")

    masked = np.where(valid_mask[:, None, :], contributions, 0.0)
    trajectories = np.cumsum(masked, axis=2)
    log_posterior = trajectories - np.logaddexp.reduce(trajectories, axis=1, keepdims=True)
    decided = np.zeros(n, dtype=bool)
    stop_index = sequence_lengths - 1
    predicted = np.full(n, -1, dtype=np.int64)
    for step in range(t):
        posterior = np.exp(
            log_posterior[:, :, step]
            - np.logaddexp.reduce(log_posterior[:, :, step], axis=1, keepdims=True)
        )
        active = sequence_lengths > step
        crossing = (~decided) & active & (posterior.max(axis=1) >= threshold)
        stop_index[crossing] = step
        predicted[crossing] = posterior[crossing].argmax(axis=1)
        decided |= crossing
    decided_rows = np.flatnonzero(decided)
    error = (
        float(np.mean(true[decided_rows] != predicted[decided_rows]))
        if decided_rows.size
        else float("nan")
    )
    return {
        "threshold": float(threshold),
        "decided_fraction": float(np.mean(decided)),
        "undecided_fraction": float(np.mean(~decided)),
        "empirical_error": error,
        "expected_flashes": (
            float(np.mean(stop_index[decided_rows] + 1)) if decided_rows.size else float("nan")
        ),
        "n_sequences": int(n),
        "n_decided": int(decided_rows.size),
    }


def risk_coverage_curve(
    contributions: np.ndarray,
    true_candidates: np.ndarray,
    *,
    valid_mask: np.ndarray | None,
    sequence_lengths: np.ndarray | None = None,
    thresholds: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10)),
) -> list[dict[str, object]]:
    """Return (risk, coverage) pairs for one fixed acquisition/valid-mask rule.

    Risk is the empirical error among decided sequences; coverage is the
    decided fraction. The function only evaluates fixed thresholds, never
    tunes them on labels.
    """
    points: list[dict[str, object]] = []
    for threshold in thresholds:
        point = replay_chain_stopping_from_contributions(
            contributions,
            true_candidates,
            valid_mask=valid_mask,
            threshold=threshold,
            sequence_lengths=sequence_lengths,
        )
        points.append(
            {
                "threshold": float(threshold),
                "risk": (
                    point["empirical_error"]
                    if np.isfinite(point["empirical_error"])
                    else 1.0
                ),
                "coverage": point["decided_fraction"],
                "undecided_fraction": point["undecided_fraction"],
                "expected_flashes": point["expected_flashes"],
            }
        )
    return points


def curve_improvement(
    gated: list[dict[str, object]],
    ungated: list[dict[str, object]],
    *,
    target_coverage: float = 0.90,
) -> dict[str, object]:
    """Compare risk-coverage curves at the largest common coverage >= target."""
    gated = sorted(gated, key=lambda point: float(point["coverage"]))
    ungated = sorted(ungated, key=lambda point: float(point["coverage"]))
    gated_at = next(
        (point for point in gated if float(point["coverage"]) >= target_coverage), None
    )
    ungated_at = next(
        (point for point in ungated if float(point["coverage"]) >= target_coverage), None
    )
    if gated_at is None or ungated_at is None:
        return {"available": False, "improved": False}
    return {
        "available": True,
        "improved": bool(float(gated_at["risk"]) <= float(ungated_at["risk"])),
        "gated_risk": float(gated_at["risk"]),
        "ungated_risk": float(ungated_at["risk"]),
        "coverage": float(gated_at["coverage"]),
    }


def two_hypothesis_conformal_flags(
    typicality_calibration: np.ndarray,
    typicality_test: np.ndarray,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Both-hypothesis-reject flags from one-sided conformal p-values."""
    calibration = np.asarray(typicality_calibration, dtype=np.float64).reshape(-1)
    test = np.asarray(typicality_test, dtype=np.float64).reshape(-1)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0,1).")
    p_values = (1.0 + (calibration[None, :] >= test[:, None]).sum(axis=1)) / (
        len(calibration) + 1.0



    )
    return p_values < alpha


def prequential_e_process(llr: np.ndarray) -> np.ndarray:
    """Cumulative likelihood-ratio e-process from per-trial LLR values.

    Under a correctly specified null hypothesis each factor has expectation
    at most one and the product is nonnegative; the cumulative product is the
    anytime-valid evidence against the null.
    """
    llr = np.asarray(llr, dtype=np.float64).reshape(-1)
    factors = np.exp(np.clip(llr, -40.0, 40.0))
    return np.cumprod(factors)


def e_process_diagnostics(llr: np.ndarray) -> dict[str, object]:
    """Diagnostic gate for prequential e-process nonnegativity and E[e]<=1.

    A one-dimensional input is interpreted as one independent e-value per
    trial (valid for the per-trial prequential LLR audit). A two-dimensional
    input is interpreted as one sequence per row; ``E[e]`` is then estimated
    from the independent final cumulative values.
    """
    llr = np.asarray(llr, dtype=np.float64)
    if llr.size == 0:
        return {"passed": False, "n_trials": 0}
    if llr.ndim == 1:
        factors = np.exp(np.clip(llr, -40.0, 40.0))
        e_values = factors
        final_e = factors
    elif llr.ndim == 2:
        factors = np.exp(np.clip(llr, -40.0, 40.0))
        e_values = np.cumprod(factors, axis=1)
        final_e = e_values[:, -1]
    else:
        raise ValueError("llr must be one- or two-dimensional.")
    nonnegative = bool(np.min(e_values) >= 0.0)
    final_mean = float(np.mean(final_e))
    mean_factor = float(np.mean(factors))
    passed = bool(
        nonnegative
        and final_mean <= 1.0 + 0.05
        and mean_factor <= 1.0 + 0.05
    )
    return {
        "passed": passed,
        "nonnegative": nonnegative,
        "mean_e": mean_factor,
        "final_mean": final_mean,
        "n_trials": int(llr.shape[0]),
        "n_steps": int(llr.shape[1]) if llr.ndim == 2 else 1,
        "final_e": float(np.mean(final_e)),
    }
