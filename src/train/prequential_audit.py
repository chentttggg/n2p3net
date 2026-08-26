"""Subject-disjoint structure selection for the causal observation model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from models.innovation import CausalInnovationOutput
from train.contracts import GenerativeProfile
from train.prequential import (
    PREQUENTIAL_VARIANTS,
    PrequentialVariant,
    _low_rank_baseline,
    causal_adaptive_ar1_prediction,
    causal_ar_prediction,
    prequential_nll_per_trial,
)

PREDICTIVE_PARENTS: dict[PrequentialVariant, tuple[PrequentialVariant, ...]] = {
    "linear_ar": ("m0",),
    "m1": ("linear_ar",),
    "m2_diag": ("m1",),
    "m2_low_rank": ("m1",),
    "m3_low_rank_dynamic": ("m2_diag", "m2_low_rank"),
}
_NLL_IMPROVEMENT_ATOL = 1e-8
_NLL_IMPROVEMENT_RTOL = 1e-6


@dataclass(frozen=True)
class PrequentialAuditReport:
    passed: bool
    selected_variant: str
    neural_mean_retained: bool
    dynamic_diagonal_retained: bool
    low_rank_retained: bool
    nll_by_variant: dict[str, float]
    relative_improvement: dict[str, float]
    subject_win_fraction: dict[str, float]
    standardized_mean_abs_max: float
    class_conditional_mean_rms_max: float
    complex_mean_difference_rms: float
    standardized_covariance_error: float
    temporal_autocorrelation_max: float
    diagnostics_by_variant: dict[str, dict[str, float]]
    checks_by_variant: dict[str, dict[str, bool]]
    predictive_parent_by_variant: dict[str, tuple[str, ...]]
    relative_improvement_vs_parent: dict[str, float]
    subject_win_fraction_vs_parent: dict[str, float]
    eligible_variants: tuple[str, ...]
    checks: dict[str, bool]
    source_n_trials: int
    source_n_subjects: int
    profile_observed_channels: int
    minimum_observed_channels: int
    observation_pattern_supported: bool
    scope: str = "subject_disjoint_audit"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _relative_gain(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference.mean() - candidate.mean()) / reference.mean().abs().clamp_min(1e-8))


def _strict_score_improvement(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Require an NLL decrease larger than float32-scale numerical noise."""

    margin = _NLL_IMPROVEMENT_ATOL + _NLL_IMPROVEMENT_RTOL * reference.abs()
    return candidate < reference - margin


def _subject_means(values: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
    """Aggregate proper scores with one equal-weight value per subject."""

    return torch.stack(
        [values[subject_ids == subject].mean() for subject in torch.unique(subject_ids)]
    )


def _subject_win_fraction(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    subject_ids: torch.Tensor,
) -> float:
    wins: list[torch.Tensor] = []
    for subject in torch.unique(subject_ids):
        selected = subject_ids == subject
        wins.append(
            _strict_score_improvement(
                reference[selected].mean(),
                candidate[selected].mean(),
            )
        )
    return float(torch.stack(wins).float().mean()) if wins else 0.0


@dataclass(frozen=True)
class _ResidualDiagnostics:
    standardized_mean_abs_max: float
    class_conditional_mean_rms_max: float
    complex_mean_difference_rms: float
    standardized_covariance_error: float
    temporal_autocorrelation_max: float
    temporal_autocorrelation_lag: int
    temporal_autocorrelation_channel: int


def _standardized_residual_diagnostics(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    variant: PrequentialVariant,
    *,
    max_lag: int = 10,
) -> _ResidualDiagnostics:
    mean_model, covariance_model = PREQUENTIAL_VARIANTS[variant]
    labels = (labels.reshape(-1).to(observation.device) > 0.5).long()
    rows = torch.arange(observation.shape[0], device=observation.device)
    class_mean = profile.class_means.to(observation.device)[labels]
    mean = class_mean
    if mean_model in ("linear_ar", "neural"):
        centered = observation.detach().float() - class_mean.float()
        ar_prediction = causal_ar_prediction(
            centered,
            profile.ar_coefficients.to(observation.device),
        )
        ar_error = centered - ar_prediction
        mean = mean + ar_prediction + causal_adaptive_ar1_prediction(ar_error)
        if mean_model == "neural":
            mean = mean + innovation.history_correction[rows, labels]

    channels = profile.channel_mask.to(observation.device, dtype=torch.bool)
    residual = (observation.detach().float() - mean.float()).transpose(1, 2)[:, :, channels]
    variance_source = (
        profile.ar_channel_variances
        if mean_model in ("linear_ar", "neural")
        else profile.class_channel_variances
    )
    static = variance_source.to(observation.device)[labels]
    base_diagonal = static[:, None, :].expand(-1, observation.shape[-1], -1)
    static_factor = None
    if covariance_model in ("low_rank_static", "low_rank_dynamic"):
        low_rank_diagonal, low_rank_factor = _low_rank_baseline(
            profile, mean_model, labels, device=observation.device
        )
        base_diagonal = low_rank_diagonal[:, None, :].expand(-1, observation.shape[-1], -1)
        if low_rank_factor is not None:
            static_factor = low_rank_factor[:, None].expand(-1, observation.shape[-1], -1, -1)
    if covariance_model in ("static", "low_rank_static"):
        diagonal = base_diagonal[:, :, channels]
    else:
        log_scale = innovation.log_variance_scale[rows, labels].float().clamp(-5.0, 5.0)
        diagonal = (base_diagonal * log_scale.exp())[:, :, channels]
    factor = None
    if covariance_model == "low_rank_static":
        factor = static_factor
    elif covariance_model == "low_rank_dynamic":
        correction = innovation.factor_scale[rows, labels].float() * base_diagonal.sqrt()[..., None]
        factor = correction if static_factor is None else static_factor + correction
    if factor is not None:
        factor = factor[:, :, channels]

    covariance = torch.diag_embed(diagonal.clamp_min(1e-6))
    if factor is not None:
        covariance = covariance + factor @ factor.transpose(-1, -2)
    chol = torch.linalg.cholesky(covariance)
    standardized = torch.linalg.solve_triangular(chol, residual[..., None], upper=False).squeeze(-1)
    time_selected = profile.score_time_mask.to(observation.device) > 0.0
    z = standardized[:, time_selected]
    flattened = z.reshape(-1, z.shape[-1])
    mean_abs_max = flattened.mean(dim=0).abs().max()

    subject_ids = subject_ids.reshape(-1).to(observation.device)
    class_waveforms: list[torch.Tensor] = []
    for class_index in (0, 1):
        per_subject = [
            z[(subject_ids == subject) & (labels == class_index)].mean(dim=0)
            for subject in torch.unique(subject_ids)
        ]
        class_waveforms.append(torch.stack(per_subject).mean(dim=0))
    class_mean = torch.stack(class_waveforms)
    class_mean_rms_max = class_mean.square().mean(dim=(1, 2)).sqrt().max()
    complex_difference = torch.fft.fft(
        class_mean[1] - class_mean[0],
        dim=0,
        norm="ortho",
    )
    complex_mean_difference_rms = complex_difference.abs().square().mean().sqrt()

    centered = flattened - flattened.mean(dim=0)
    covariance_estimate = centered.T @ centered / max(flattened.shape[0] - 1, 1)
    identity = torch.eye(z.shape[-1], device=z.device)
    covariance_error = torch.linalg.matrix_norm(covariance_estimate - identity) / z.shape[-1]

    autocorrelations: list[torch.Tensor] = []
    for lag in range(1, min(max_lag, z.shape[1] - 1) + 1):
        left = z[:, :-lag]
        right = z[:, lag:]
        numerator = (left * right).sum(dim=(0, 1)).abs()
        denominator = (
            (left.square().sum(dim=(0, 1)) * right.square().sum(dim=(0, 1))).sqrt().clamp_min(1e-8)
        )
        autocorrelations.append(numerator / denominator)
    if autocorrelations:
        autocorrelation_grid = torch.stack(autocorrelations)
        flat_index = int(autocorrelation_grid.argmax())
        autocorrelation_lag = flat_index // autocorrelation_grid.shape[1] + 1
        autocorrelation_channel = flat_index % autocorrelation_grid.shape[1]
        autocorrelation_max = autocorrelation_grid.flatten()[flat_index]
    else:
        autocorrelation_max = torch.zeros((), device=observation.device)
        autocorrelation_lag = 0
        autocorrelation_channel = -1
    return _ResidualDiagnostics(
        standardized_mean_abs_max=float(mean_abs_max),
        class_conditional_mean_rms_max=float(class_mean_rms_max),
        complex_mean_difference_rms=float(complex_mean_difference_rms),
        standardized_covariance_error=float(covariance_error),
        temporal_autocorrelation_max=float(autocorrelation_max),
        temporal_autocorrelation_lag=autocorrelation_lag,
        temporal_autocorrelation_channel=autocorrelation_channel,
    )


@torch.no_grad()
def audit_prequential_model(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    *,
    observation_mask: torch.Tensor | None = None,
    min_relative_improvement: float = 0.002,
    min_subject_win_fraction: float = 0.50,
    standardized_mean_abs_max: float = 0.25,
    class_conditional_mean_rms_max: float = 0.25,
    complex_mean_difference_rms_max: float = 0.35,
    standardized_covariance_error_max: float = 0.50,
    temporal_autocorrelation_max: float = 0.25,
) -> PrequentialAuditReport:
    """Select a calibrated causal density using one untouched audit split.

    Proper-score improvement is necessary but not sufficient: every candidate
    must also produce approximately centered, spatially standardized and
    temporally white innovations. Selection by NLL happens only after those
    absolute checks, preventing a flexible scale/covariance head from winning
    by reweighting colored residuals.
    """

    if min_relative_improvement < 0.0 or not 0.0 <= min_subject_win_fraction <= 1.0:
        raise ValueError("Invalid prequential improvement thresholds.")
    subject_ids = subject_ids.reshape(-1).to(observation.device)
    if subject_ids.numel() != observation.shape[0]:
        raise ValueError("subject_ids must align with audit trials.")
    if torch.unique(subject_ids).numel() < 2:
        raise ValueError("Prequential audit requires at least two untouched subjects.")
    binary_labels = labels.reshape(-1).to(observation.device) > 0.5
    for subject in torch.unique(subject_ids):
        selected = subject_ids == subject
        if not bool(binary_labels[selected].any()) or not bool((~binary_labels[selected]).any()):
            raise ValueError("Every audit subject must contain both target classes.")

    profile_mask = profile.channel_mask.to(observation.device, dtype=torch.bool)
    if observation_mask is None:
        effective_mask = profile_mask[None].expand(observation.shape[0], -1)
    else:
        runtime_mask = observation_mask.to(observation.device, dtype=torch.bool)
        if runtime_mask.shape == (observation.shape[1],):
            runtime_mask = runtime_mask[None].expand(observation.shape[0], -1)
        if runtime_mask.shape != observation.shape[:2]:
            raise ValueError("observation_mask must be (C,) or (B,C).")
        effective_mask = runtime_mask & profile_mask[None]
    supported_observation_pattern = bool(
        (effective_mask == profile_mask[None]).all() and effective_mask.any(dim=1).all()
    )

    variants = tuple(PREQUENTIAL_VARIANTS)
    per_trial = {
        name: prequential_nll_per_trial(
            observation,
            innovation,
            profile,
            labels=labels,
            variant=name,
            observation_mask=effective_mask,
        )
        for name in variants
    }
    per_subject = {name: _subject_means(values, subject_ids) for name, values in per_trial.items()}
    nll = {name: float(values.mean()) for name, values in per_subject.items()}

    diagnostics_by_variant: dict[str, dict[str, float]] = {}
    checks_by_variant: dict[str, dict[str, bool]] = {}
    gain_vs_m0: dict[str, float] = {}
    wins_vs_m0: dict[str, float] = {}
    gain_vs_parent: dict[str, float] = {}
    wins_vs_parent: dict[str, float] = {}
    eligible: list[PrequentialVariant] = []
    for diagnostic_variant in variants:
        values = (
            _standardized_residual_diagnostics(
                observation,
                innovation,
                profile,
                labels,
                subject_ids,
                diagnostic_variant,
            )
            if supported_observation_pattern
            else _ResidualDiagnostics(
                standardized_mean_abs_max=float("inf"),
                class_conditional_mean_rms_max=float("inf"),
                complex_mean_difference_rms=float("inf"),
                standardized_covariance_error=float("inf"),
                temporal_autocorrelation_max=float("inf"),
                temporal_autocorrelation_lag=0,
                temporal_autocorrelation_channel=-1,
            )
        )
        diagnostics_by_variant[diagnostic_variant] = {
            "standardized_mean_abs_max": values.standardized_mean_abs_max,
            "class_conditional_mean_rms_max": values.class_conditional_mean_rms_max,
            "complex_mean_difference_rms": values.complex_mean_difference_rms,
            "standardized_covariance_error": values.standardized_covariance_error,
            "temporal_autocorrelation_max": values.temporal_autocorrelation_max,
            "temporal_autocorrelation_lag": float(values.temporal_autocorrelation_lag),
            "temporal_autocorrelation_channel": float(values.temporal_autocorrelation_channel),
        }
        gain = _relative_gain(per_subject["m0"], per_subject[diagnostic_variant])
        win = _subject_win_fraction(per_trial["m0"], per_trial[diagnostic_variant], subject_ids)
        gain_vs_m0[diagnostic_variant] = gain
        wins_vs_m0[diagnostic_variant] = win
        parents = PREDICTIVE_PARENTS.get(diagnostic_variant, ())
        if not parents:
            parent_gain = 0.0
            parent_win = 0.0
        else:
            parent_gain = min(
                _relative_gain(per_subject[parent], per_subject[diagnostic_variant])
                for parent in parents
            )
            parent_win = min(
                _subject_win_fraction(
                    per_trial[parent],
                    per_trial[diagnostic_variant],
                    subject_ids,
                )
                for parent in parents
            )
        gain_vs_parent[diagnostic_variant] = parent_gain
        wins_vs_parent[diagnostic_variant] = parent_win
        candidate_checks = {
            "supported_observation_pattern": supported_observation_pattern,
            "finite_nll": bool(torch.isfinite(per_trial[diagnostic_variant]).all()),
            "predictive_skill_vs_m0": bool(
                diagnostic_variant == "m0"
                or (gain >= min_relative_improvement and win > min_subject_win_fraction)
            ),
            "incremental_predictive_skill_vs_parent": bool(
                diagnostic_variant == "m0"
                or (
                    bool(parents)
                    and parent_gain >= min_relative_improvement
                    and parent_win > min_subject_win_fraction
                )
            ),
            "standardized_zero_mean": (
                values.standardized_mean_abs_max <= standardized_mean_abs_max
            ),
            "class_conditional_mean_neutrality": (
                values.class_conditional_mean_rms_max <= class_conditional_mean_rms_max
            ),
            "complex_mean_difference_neutrality": (
                values.complex_mean_difference_rms <= complex_mean_difference_rms_max
            ),
            "standardized_identity_covariance": (
                values.standardized_covariance_error <= standardized_covariance_error_max
            ),
            "temporal_whiteness": (
                values.temporal_autocorrelation_max <= temporal_autocorrelation_max
            ),
        }
        checks_by_variant[diagnostic_variant] = candidate_checks
        if all(candidate_checks.values()):
            eligible.append(diagnostic_variant)

    # Dict insertion order is the declared complexity order, so exact NLL
    # ties prefer the simpler candidate.
    selected: PrequentialVariant = min(eligible, key=lambda name: nll[name]) if eligible else "m0"
    mean_abs = diagnostics_by_variant[selected]["standardized_mean_abs_max"]
    class_mean_rms = diagnostics_by_variant[selected]["class_conditional_mean_rms_max"]
    complex_mean_difference = diagnostics_by_variant[selected]["complex_mean_difference_rms"]
    covariance_error = diagnostics_by_variant[selected]["standardized_covariance_error"]
    autocorrelation = diagnostics_by_variant[selected]["temporal_autocorrelation_max"]
    selected_checks = checks_by_variant[selected]
    selected_mean, selected_covariance = PREQUENTIAL_VARIANTS[selected]
    keep_neural_mean = selected_mean == "neural"
    keep_low_rank = selected_covariance in ("low_rank_static", "low_rank_dynamic")
    keep_diagonal = selected_covariance in ("dynamic_diag", "low_rank_dynamic")

    mean_gain = _relative_gain(per_subject["linear_ar"], per_subject["m1"])
    mean_win = _subject_win_fraction(per_trial["linear_ar"], per_trial["m1"], subject_ids)
    diagonal_gain = _relative_gain(per_subject["m1"], per_subject["m2_diag"])
    diagonal_win = _subject_win_fraction(per_trial["m1"], per_trial["m2_diag"], subject_ids)
    low_rank_static_gain = _relative_gain(per_subject["m1"], per_subject["m2_low_rank"])
    low_rank_static_win = _subject_win_fraction(
        per_trial["m1"], per_trial["m2_low_rank"], subject_ids
    )
    low_rank_dynamic_gain = gain_vs_parent["m3_low_rank_dynamic"]
    low_rank_dynamic_win = wins_vs_parent["m3_low_rank_dynamic"]
    m0_gain = gain_vs_m0[selected]
    m0_win = wins_vs_m0[selected]
    checks = {key: bool(value) for key, value in selected_checks.items()}
    return PrequentialAuditReport(
        passed=bool(eligible) and all(checks.values()),
        selected_variant=selected,
        neural_mean_retained=keep_neural_mean,
        dynamic_diagonal_retained=keep_diagonal,
        low_rank_retained=keep_low_rank,
        nll_by_variant=nll,
        relative_improvement={
            "neural_mean_vs_linear_ar": mean_gain,
            "dynamic_diagonal_vs_neural_static": diagonal_gain,
            "low_rank_static_vs_neural_static": low_rank_static_gain,
            "low_rank_dynamic_vs_both_parents": low_rank_dynamic_gain,
            "selected_vs_m0": m0_gain,
        },
        subject_win_fraction={
            "neural_mean_vs_linear_ar": mean_win,
            "dynamic_diagonal_vs_neural_static": diagonal_win,
            "low_rank_static_vs_neural_static": low_rank_static_win,
            "low_rank_dynamic_vs_both_parents": low_rank_dynamic_win,
            "selected_vs_m0": m0_win,
        },
        standardized_mean_abs_max=mean_abs,
        class_conditional_mean_rms_max=class_mean_rms,
        complex_mean_difference_rms=complex_mean_difference,
        standardized_covariance_error=covariance_error,
        temporal_autocorrelation_max=autocorrelation,
        diagnostics_by_variant=diagnostics_by_variant,
        checks_by_variant=checks_by_variant,
        predictive_parent_by_variant={
            variant: parents for variant, parents in PREDICTIVE_PARENTS.items()
        },
        relative_improvement_vs_parent=gain_vs_parent,
        subject_win_fraction_vs_parent=wins_vs_parent,
        eligible_variants=tuple(eligible),
        checks=checks,
        source_n_trials=int(observation.shape[0]),
        source_n_subjects=int(torch.unique(subject_ids).numel()),
        profile_observed_channels=int(profile_mask.sum()),
        minimum_observed_channels=int(effective_mask.sum(dim=1).min()),
        observation_pattern_supported=supported_observation_pattern,
    )
