"""Strict-past Gaussian scoring and fold-local generative profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from models.innovation import CausalInnovationOutput
from train.contracts import GenerativeProfile

PrequentialVariant = Literal[
    "m0",
    "linear_ar",
    "m1",
    "m2_diag",
    "m2_low_rank",
    "m3_low_rank_dynamic",
]


PREQUENTIAL_VARIANTS: dict[PrequentialVariant, tuple[str, str]] = {
    "m0": ("zero", "static"),
    "linear_ar": ("linear_ar", "static"),
    "m1": ("neural", "static"),
    "m2_diag": ("neural", "dynamic_diag"),
    "m2_low_rank": ("neural", "low_rank_static"),
    "m3_low_rank_dynamic": ("neural", "low_rank_dynamic"),
}


@dataclass(frozen=True)
class PrequentialTrialScore:
    """Additive density evidence and its optimization-scale normalization."""

    nll_per_time: torch.Tensor
    nll_sum: torch.Tensor
    nll_per_observed_scalar: torch.Tensor
    observed_scalar_count: torch.Tensor
    valid: torch.Tensor


def causal_ar_prediction(
    signal: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Apply a fold-fixed VAR(p) without reading the current sample."""

    if signal.dim() != 3 or coefficients.dim() != 3:
        raise ValueError("signal and coefficients must be (B,C,T) and (order,C,C).")
    if coefficients.shape[1:] != (signal.shape[1], signal.shape[1]):
        raise ValueError("VAR coefficient channels must match signal channels.")
    order = min(int(coefficients.shape[0]), signal.shape[-1] - 1)
    if order <= 0:
        return torch.zeros_like(signal)
    # Left padding by ``order`` places coefficient 0 on x[t-1], never x[t].
    weight = coefficients[:order].to(device=signal.device, dtype=signal.dtype)
    weight = weight.permute(1, 2, 0).flip(-1)
    return F.conv1d(F.pad(signal, (order, 0)), weight)[..., : signal.shape[-1]]


def causal_adaptive_ar1_prediction(
    innovation: torch.Tensor,
    *,
    min_history: int = 8,
    stability_limit: float = 0.95,
) -> torch.Tensor:
    """Predict each channel's next innovation from a strict-past online fit.

    The coefficient at ``t`` is estimated only from innovation pairs ending
    before ``t``. This adapts a fold-level VAR to each trial's baseline prefix
    without labels, future samples or persistent subject parameters.
    """

    if innovation.dim() != 3:
        raise ValueError("innovation must be (B,C,T).")
    if min_history < 1 or not 0.0 < stability_limit < 1.0:
        raise ValueError("Invalid adaptive AR(1) stability settings.")
    prediction = torch.zeros_like(innovation)
    if innovation.shape[-1] < 3:
        return prediction
    lagged_products = innovation[:, :, 1:] * innovation[:, :, :-1]
    lagged_squares = innovation[:, :, :-1].square()
    cumulative_products = lagged_products.cumsum(dim=-1)
    cumulative_squares = lagged_squares.cumsum(dim=-1)
    # At time t>=2, index t-2 includes pairs ending at t-1 and excludes e_t.
    numerator = cumulative_products[:, :, :-1]
    denominator = cumulative_squares[:, :, :-1].clamp_min(1e-6)
    coefficient = (numerator / denominator).clamp(-stability_limit, stability_limit)
    counts = torch.arange(
        1,
        coefficient.shape[-1] + 1,
        device=innovation.device,
    )
    coefficient = coefficient * (counts >= min_history).to(coefficient.dtype)
    prediction[:, :, 2:] = coefficient * innovation[:, :, 1:-1]
    return prediction


def _estimate_low_rank_covariance(
    residual: torch.Tensor,
    selected: torch.Tensor,
    time_mask: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    rank: int,
    variance_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a fold-fixed diagonal-plus-low-rank covariance by eigentruncation."""

    channels = residual.shape[1]
    diagonal = torch.ones(channels, device=residual.device, dtype=torch.float32)
    factor = torch.zeros(channels, rank, device=residual.device, dtype=torch.float32)
    samples = residual[selected][:, channel_mask][:, :, time_mask].permute(0, 2, 1)
    samples = samples.reshape(-1, int(channel_mask.sum())).float()
    samples = samples - samples.mean(dim=0, keepdim=True)
    covariance = samples.T @ samples / max(samples.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    active_rank = min(rank, covariance.shape[0])
    retained_values = eigenvalues[-active_rank:].clamp_min(0.0)
    retained_vectors = eigenvectors[:, -active_rank:]
    selected_factor = retained_vectors * retained_values.sqrt()[None]
    remainder = covariance - selected_factor @ selected_factor.T
    selected_diagonal = remainder.diagonal().clamp_min(float(variance_floor))
    diagonal[channel_mask] = selected_diagonal
    factor[channel_mask, :active_rank] = selected_factor
    return diagonal, factor


@torch.no_grad()
def estimate_generative_profile(
    observation: torch.Tensor,
    labels: torch.Tensor,
    *,
    sfreq: float,
    tmin_ms: float,
    channel_mask: torch.Tensor | None = None,
    score_interval_ms: tuple[float, float] = (0.0, 800.0),
    variance_floor: float = 1e-3,
    ar_order: int = 32,
    ar_ridge: float = 1e-3,
    covariance_rank: int = 2,
) -> GenerativeProfile:
    """Fit class templates and static M0 variances on optimization data only."""

    if observation.dim() != 3:
        raise ValueError("observation must be (N,C,T).")
    y = labels.reshape(-1).to(device=observation.device)
    if y.numel() != observation.shape[0]:
        raise ValueError("observation and labels must have equal trial counts.")
    if variance_floor <= 0.0 or ar_ridge <= 0.0 or ar_order < 1:
        raise ValueError("variance_floor and ar_ridge must be positive.")
    if covariance_rank not in (1, 2):
        raise ValueError("covariance_rank must be 1 or 2.")
    ar_order = min(int(ar_order), observation.shape[-1] - 1)
    positive = y > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("Generative profile requires both target classes.")
    if channel_mask is None:
        channel_mask = torch.ones(observation.shape[1], dtype=torch.bool, device=observation.device)
    else:
        channel_mask = channel_mask.to(device=observation.device, dtype=torch.bool)
    if channel_mask.shape != (observation.shape[1],) or not bool(channel_mask.any()):
        raise ValueError("channel_mask must select at least one observation channel.")

    start_ms, stop_ms = (float(value) for value in score_interval_ms)
    if not start_ms < stop_ms:
        raise ValueError("score_interval_ms must have positive width.")
    times_ms = tmin_ms + torch.arange(
        observation.shape[-1], device=observation.device, dtype=torch.float32
    ) * (1000.0 / float(sfreq))
    score_time_mask = ((times_ms >= start_ms) & (times_ms < stop_ms)).float()
    if not bool(score_time_mask.any()):
        raise ValueError("score_interval_ms does not overlap the observation epoch.")

    work = observation.detach().float()
    class_means = torch.stack((work[negative].mean(dim=0), work[positive].mean(dim=0)))
    class_variances: list[torch.Tensor] = []
    time_weight = score_time_mask[None, None]
    time_count = score_time_mask.sum().clamp_min(1.0)
    for class_index, selected in enumerate((negative, positive)):
        residual = work[selected] - class_means[class_index][None]
        variance = (residual.square() * time_weight).sum(dim=(0, 2)) / (
            float(selected.sum()) * time_count
        )
        class_variances.append(variance.clamp_min(float(variance_floor)))
    class_channel_variances = torch.stack(class_variances)

    label_indices = positive.long()
    centered = work - class_means[label_indices]
    ar_time_mask = score_time_mask[ar_order:] > 0.0
    lagged = torch.stack(
        [centered[:, :, ar_order - lag : -lag] for lag in range(1, ar_order + 1)],
        dim=2,
    ).permute(0, 3, 2, 1)
    predictors = lagged[:, ar_time_mask].reshape(-1, ar_order * observation.shape[1])
    responses = (
        centered[:, :, ar_order:].transpose(1, 2)[:, ar_time_mask].reshape(-1, observation.shape[1])
    )
    gram = predictors.T @ predictors
    ridge_scale = float(ar_ridge) * gram.diagonal().mean().clamp_min(1e-6)
    ridge = ridge_scale * torch.eye(
        ar_order * observation.shape[1], device=observation.device, dtype=torch.float32
    )
    coefficients_flat = torch.linalg.solve(gram + ridge, predictors.T @ responses).T
    ar_coefficients = coefficients_flat.reshape(
        observation.shape[1], ar_order, observation.shape[1]
    ).permute(1, 0, 2)
    ar_prediction = causal_ar_prediction(centered, ar_coefficients)
    ar_error = centered - ar_prediction
    adaptive_prediction = causal_adaptive_ar1_prediction(ar_error)
    ar_error = ar_error - adaptive_prediction
    ar_variances: list[torch.Tensor] = []
    ar_weight = score_time_mask[None, None]
    ar_time_count = score_time_mask.sum().clamp_min(1.0)
    for selected in (negative, positive):
        variance = (ar_error[selected].square() * ar_weight).sum(dim=(0, 2)) / (
            float(selected.sum()) * ar_time_count
        )
        ar_variances.append(variance.clamp_min(float(variance_floor)))
    ar_channel_variances = torch.stack(ar_variances)

    centered_residual = work - class_means[label_indices]
    class_low_rank = [
        _estimate_low_rank_covariance(
            centered_residual,
            selected,
            score_time_mask > 0.0,
            channel_mask,
            rank=covariance_rank,
            variance_floor=variance_floor,
        )
        for selected in (negative, positive)
    ]
    ar_low_rank = [
        _estimate_low_rank_covariance(
            ar_error,
            selected,
            score_time_mask > 0.0,
            channel_mask,
            rank=covariance_rank,
            variance_floor=variance_floor,
        )
        for selected in (negative, positive)
    ]
    class_low_rank_diagonal = torch.stack([parts[0] for parts in class_low_rank])
    class_low_rank_factor = torch.stack([parts[1] for parts in class_low_rank])
    ar_low_rank_diagonal = torch.stack([parts[0] for parts in ar_low_rank])
    ar_low_rank_factor = torch.stack([parts[1] for parts in ar_low_rank])

    class_channel_variances[:, ~channel_mask] = 1.0
    ar_channel_variances[:, ~channel_mask] = 1.0
    ar_coefficients[:, ~channel_mask] = 0.0
    ar_coefficients[:, :, ~channel_mask] = 0.0
    class_means[:, ~channel_mask] = 0.0

    profile = GenerativeProfile(
        class_means=class_means,
        class_channel_variances=class_channel_variances,
        ar_coefficients=ar_coefficients,
        ar_channel_variances=ar_channel_variances,
        target_rate=y.float().mean(),
        channel_mask=channel_mask,
        score_time_mask=score_time_mask,
        sfreq=float(sfreq),
        tmin_ms=float(tmin_ms),
        n_time=int(observation.shape[-1]),
        source_n_trials=int(observation.shape[0]),
        class_low_rank_diagonal=class_low_rank_diagonal,
        class_low_rank_factor=class_low_rank_factor,
        ar_low_rank_diagonal=ar_low_rank_diagonal,
        ar_low_rank_factor=ar_low_rank_factor,
    )
    profile.validate(n_channels=observation.shape[1])
    return profile


def _class_indices(
    batch_size: int,
    device: torch.device,
    *,
    labels: torch.Tensor | None,
    hypothesis: int | None,
) -> torch.Tensor:
    if (labels is None) == (hypothesis is None):
        raise ValueError("Provide exactly one of labels or hypothesis.")
    if hypothesis is not None:
        if hypothesis not in (0, 1):
            raise ValueError("hypothesis must be 0 or 1.")
        return torch.full((batch_size,), hypothesis, dtype=torch.long, device=device)
    assert labels is not None
    indices = (labels.reshape(-1).to(device=device) > 0.5).long()
    if indices.numel() != batch_size:
        raise ValueError("labels must match the innovation batch size.")
    return indices


def _effective_observation_mask(
    observation: torch.Tensor,
    profile: GenerativeProfile,
    observation_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return the per-trial intersection of runtime and fold-profile channels."""

    batch_size, channels, _ = observation.shape
    profile_mask = profile.channel_mask.to(observation.device, dtype=torch.bool)
    if observation_mask is None:
        runtime_mask = profile_mask[None].expand(batch_size, -1)
    else:
        runtime_mask = observation_mask.to(observation.device, dtype=torch.bool)
        if runtime_mask.shape == (channels,):
            runtime_mask = runtime_mask[None].expand(batch_size, -1)
        if runtime_mask.shape != (batch_size, channels):
            raise ValueError("observation_mask must be (C,) or (B,C).")
    return runtime_mask & profile_mask[None]


def _gaussian_nll_per_time(
    residual: torch.Tensor,
    diagonal: torch.Tensor,
    factor: torch.Tensor | None,
) -> torch.Tensor:
    """Return Gaussian NLL as ``(B,T)`` using a stable Woodbury solve."""

    if residual.shape != diagonal.shape or residual.dim() != 3:
        raise ValueError("residual and diagonal must share shape (B,T,C).")
    diagonal = diagonal.clamp_min(1e-6)
    diagonal_inverse = diagonal.reciprocal()
    quadratic = (residual.square() * diagonal_inverse).sum(dim=-1)
    log_determinant = diagonal.log().sum(dim=-1)
    if factor is not None:
        if factor.shape[:3] != residual.shape:
            raise ValueError("factor must be (B,T,C,rank).")
        rank = factor.shape[-1]
        if rank == 1:
            u = factor[..., 0]
            middle = 1.0 + (u.square() * diagonal_inverse).sum(dim=-1)
            projected = (u * diagonal_inverse * residual).sum(dim=-1)
            quadratic = quadratic - projected.square() / middle
            log_determinant = log_determinant + middle.log()
        elif rank == 2:
            u, v = factor.unbind(dim=-1)
            a = (u.square() * diagonal_inverse).sum(dim=-1)
            b = (u * v * diagonal_inverse).sum(dim=-1)
            c = (v.square() * diagonal_inverse).sum(dim=-1)
            p0 = (u * diagonal_inverse * residual).sum(dim=-1)
            p1 = (v * diagonal_inverse * residual).sum(dim=-1)
            determinant = ((1.0 + a) * (1.0 + c) - b.square()).clamp_min(1e-12)
            q0 = ((1.0 + c) * p0 - b * p1) / determinant
            q1 = ((1.0 + a) * p1 - b * p0) / determinant
            quadratic = quadratic - p0 * q0 - p1 * q1
            log_determinant = log_determinant + determinant.log()
        else:
            projected = torch.einsum("btcr,btc->btr", factor, diagonal_inverse * residual)
            gram = torch.einsum("btcr,btc,btcs->btrs", factor, diagonal_inverse, factor)
            eye = torch.eye(rank, device=residual.device, dtype=residual.dtype)[None, None]
            chol = torch.linalg.cholesky(eye + gram)
            solved = torch.cholesky_solve(projected[..., None], chol).squeeze(-1)
            quadratic = quadratic - (projected * solved).sum(dim=-1)
            log_determinant = log_determinant + 2.0 * torch.log(
                torch.diagonal(chol, dim1=-2, dim2=-1)
            ).sum(dim=-1)
    constant = residual.shape[-1] * torch.log(
        torch.tensor(2.0 * torch.pi, device=residual.device, dtype=residual.dtype)
    )
    return 0.5 * (quadratic + log_determinant + constant)


def _low_rank_baseline(
    profile: GenerativeProfile,
    mean_model: str,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the fold-fixed covariance baseline for one mean family."""

    if profile.class_low_rank_diagonal is None:
        variance = (
            profile.ar_channel_variances
            if mean_model in ("linear_ar", "neural")
            else profile.class_channel_variances
        )
        return variance.to(device)[indices], None
    diagonal = (
        profile.ar_low_rank_diagonal
        if mean_model in ("linear_ar", "neural")
        else profile.class_low_rank_diagonal
    )
    factor = (
        profile.ar_low_rank_factor
        if mean_model in ("linear_ar", "neural")
        else profile.class_low_rank_factor
    )
    assert diagonal is not None and factor is not None
    return diagonal.to(device)[indices], factor.to(device)[indices]


def _validate_prequential_shapes(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
) -> tuple[int, int, int]:
    if observation.dim() != 3 or observation.shape[-1] != profile.n_time:
        raise ValueError("observation must be (B,C,T) and match GenerativeProfile.")
    batch_size, channels, n_time = observation.shape
    n_hypotheses = innovation.history_correction.shape[1]
    if n_hypotheses not in (1, 2):
        raise ValueError("innovation must contain one labeled or two class hypotheses.")
    expected_scale_shape = (batch_size, n_hypotheses, n_time, channels)
    if innovation.history_correction.shape != (batch_size, n_hypotheses, channels, n_time):
        raise ValueError("innovation.history_correction must be (B,H,C,T).")
    if innovation.log_variance_scale.shape != expected_scale_shape:
        raise ValueError("innovation.log_variance_scale must be (B,H,T,C).")
    if innovation.factor_scale.shape[:4] != expected_scale_shape:
        raise ValueError("innovation.factor_scale must be (B,H,T,C,rank).")
    return batch_size, channels, n_time


def _innovation_indices(
    innovation: CausalInnovationOutput, class_indices: torch.Tensor
) -> torch.Tensor:
    """Map requested classes to full or labeled hypothesis slots."""

    if innovation.history_correction.shape[1] == 2:
        return class_indices
    labels = innovation.hypothesis_labels
    if labels is None:
        raise ValueError("Single-hypothesis innovation is missing hypothesis_labels.")
    labels = labels.to(device=class_indices.device, dtype=torch.long).reshape(-1)
    if labels.shape != class_indices.shape or bool((labels != class_indices).any()):
        raise ValueError("Single-hypothesis innovation does not match requested labels.")
    return torch.zeros_like(class_indices)


def _prepare_prequential_mean(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
    indices: torch.Tensor,
    effective_mask: torch.Tensor,
    *,
    mean_model: str,
) -> torch.Tensor:
    batch_size = observation.shape[0]
    rows = torch.arange(batch_size, device=observation.device)
    innovation_indices = _innovation_indices(innovation, indices)
    class_mean = profile.class_means.to(observation.device)[indices]
    mean = class_mean
    if mean_model in ("linear_ar", "neural"):
        mask_values = effective_mask[:, :, None].to(dtype=torch.float32)
        centered = (observation.detach().float() - class_mean.float()) * mask_values
        ar_prediction = (
            causal_ar_prediction(
                centered,
                profile.ar_coefficients.to(observation.device),
            )
            * mask_values
        )
        ar_error = (centered - ar_prediction) * mask_values
        adaptive_prediction = causal_adaptive_ar1_prediction(ar_error) * mask_values
        mean = mean + ar_prediction + adaptive_prediction
        if mean_model == "neural":
            mean = mean + innovation.history_correction[rows, innovation_indices] * mask_values
    return mean


def _score_prepared_mean(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
    indices: torch.Tensor,
    effective_mask: torch.Tensor,
    mean: torch.Tensor,
    *,
    mean_model: str,
    covariance_model: str,
    mask_is_homogeneous: bool = False,
) -> PrequentialTrialScore:
    batch_size, _, n_time = observation.shape
    rows = torch.arange(batch_size, device=observation.device)
    innovation_indices = _innovation_indices(innovation, indices)

    with torch.amp.autocast(device_type=observation.device.type, enabled=False):
        residual = (observation.detach().float() - mean.float()).transpose(1, 2)
        variance_source = (
            profile.ar_channel_variances
            if mean_model in ("linear_ar", "neural")
            else profile.class_channel_variances
        )
        static = variance_source.to(observation.device)[indices]
        base_diagonal = static[:, None, :].expand(-1, n_time, -1)
        static_factor = None
        if covariance_model in ("low_rank_static", "low_rank_dynamic"):
            low_rank_diagonal, low_rank_factor = _low_rank_baseline(
                profile, mean_model, indices, device=observation.device
            )
            base_diagonal = low_rank_diagonal[:, None, :].expand(-1, n_time, -1)
            if low_rank_factor is not None:
                static_factor = low_rank_factor[:, None].expand(-1, n_time, -1, -1)
        if covariance_model in ("static", "low_rank_static"):
            diagonal = base_diagonal
        else:
            log_scale = innovation.log_variance_scale[rows, innovation_indices].float().clamp(
                -5.0, 5.0
            )
            diagonal = base_diagonal * log_scale.exp()
        factor = None
        if covariance_model == "low_rank_static":
            factor = static_factor
        elif covariance_model == "low_rank_dynamic":
            correction = (
                innovation.factor_scale[rows, innovation_indices].float()
                * base_diagonal.sqrt()[..., None]
            )
            factor = correction if static_factor is None else static_factor + correction
        per_time = torch.zeros(
            batch_size,
            n_time,
            device=observation.device,
            dtype=torch.float32,
        )
        patterns = effective_mask[:1] if mask_is_homogeneous else torch.unique(effective_mask, dim=0)
        for pattern in patterns:
            selected_rows = (effective_mask == pattern[None]).all(dim=1)
            if not bool(pattern.any()):
                continue
            pattern_factor = factor[selected_rows][:, :, pattern] if factor is not None else None
            pattern_nll = _gaussian_nll_per_time(
                residual[selected_rows][:, :, pattern],
                diagonal[selected_rows][:, :, pattern],
                pattern_factor,
            )
            per_time[selected_rows] = pattern_nll
        time_weight = profile.score_time_mask.to(observation.device, dtype=torch.float32)
        weighted_per_time = per_time * time_weight[None]
        nll_sum = weighted_per_time.sum(dim=1)
        observed_channels = effective_mask.sum(dim=1).to(dtype=torch.float32)
        observed_scalar_count = observed_channels * time_weight.sum()
        valid = observed_scalar_count > 0.0
        normalized = torch.where(
            valid,
            nll_sum / observed_scalar_count.clamp_min(1.0),
            torch.full_like(nll_sum, torch.nan),
        )
        return PrequentialTrialScore(
            nll_per_time=weighted_per_time,
            nll_sum=nll_sum,
            nll_per_observed_scalar=normalized,
            observed_scalar_count=observed_scalar_count,
            valid=valid,
        )


def prequential_score_per_trial(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
    *,
    labels: torch.Tensor | None = None,
    hypothesis: int | None = None,
    variant: PrequentialVariant = "m3_low_rank_dynamic",
    observation_mask: torch.Tensor | None = None,
    mask_is_homogeneous: bool | None = None,
) -> PrequentialTrialScore:
    """Score a masked Gaussian marginal without conflating loss and evidence scales."""

    profile.validate(n_channels=observation.shape[1])
    batch_size, _, _ = _validate_prequential_shapes(observation, innovation, profile)
    if variant not in PREQUENTIAL_VARIANTS:
        raise ValueError(f"Unknown prequential variant {variant!r}.")
    mean_model, covariance_model = PREQUENTIAL_VARIANTS[variant]
    effective_mask = _effective_observation_mask(observation, profile, observation_mask)
    if mask_is_homogeneous is None:
        mask_is_homogeneous = observation_mask is None or observation_mask.dim() == 1
    finite_observed = torch.isfinite(observation) | ~effective_mask[:, :, None]
    if not bool(finite_observed.all()):
        raise ValueError(
            "Observed likelihood samples must be finite; mark missing channels explicitly."
        )

    indices = _class_indices(
        batch_size,
        observation.device,
        labels=labels,
        hypothesis=hypothesis,
    )
    mean = _prepare_prequential_mean(
        observation,
        innovation,
        profile,
        indices,
        effective_mask,
        mean_model=mean_model,
    )
    return _score_prepared_mean(
        observation,
        innovation,
        profile,
        indices,
        effective_mask,
        mean,
        mean_model=mean_model,
        covariance_model=covariance_model,
        mask_is_homogeneous=mask_is_homogeneous,
    )


def prequential_nll_per_trial(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
    *,
    labels: torch.Tensor | None = None,
    hypothesis: int | None = None,
    variant: PrequentialVariant = "m3_low_rank_dynamic",
    observation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return normalized proper scores; zero-observation trials are NaN, never perfect."""

    return prequential_score_per_trial(
        observation,
        innovation,
        profile,
        labels=labels,
        hypothesis=hypothesis,
        variant=variant,
        observation_mask=observation_mask,
    ).nll_per_observed_scalar


def prequential_nll(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
    labels: torch.Tensor,
    *,
    variant: PrequentialVariant = "m3_low_rank_dynamic",
    observation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean proper score used to train the causal generative subgraph."""

    per_trial = prequential_nll_per_trial(
        observation,
        innovation,
        profile,
        labels=labels,
        variant=variant,
        observation_mask=observation_mask,
    )
    positive = labels.reshape(-1).to(per_trial.device) > 0.5
    valid = torch.isfinite(per_trial)
    if not bool(valid.any()):
        raise ValueError("Prequential NLL found no observed likelihood samples.")
    if bool((positive & valid).any()) and bool(((~positive) & valid).any()):
        return 0.5 * (per_trial[positive & valid].mean() + per_trial[(~positive) & valid].mean())
    return per_trial[valid].mean()


def _class_balanced_prequential_nll(
    score: PrequentialTrialScore,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Reduce per-trial scores with the public class-balancing semantics."""

    per_trial = score.nll_per_observed_scalar
    positive = labels.reshape(-1).to(per_trial.device) > 0.5
    valid = torch.isfinite(per_trial)
    if not bool(valid.any()):
        raise ValueError("Prequential NLL found no observed likelihood samples.")
    if bool((positive & valid).any()) and bool(((~positive) & valid).any()):
        return 0.5 * (per_trial[positive & valid].mean() + per_trial[(~positive) & valid].mean())
    return per_trial[valid].mean()


def nested_prequential_training_loss(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
    labels: torch.Tensor,
    *,
    covariance_weight: float,
    observation_mask: torch.Tensor | None = None,
    validate_profile: bool = True,
    mask_is_homogeneous: bool | None = None,
) -> torch.Tensor:
    """Train causal mean and covariance with a nested composite proper score.

    The fixed-diagonal ``m1`` score is always present, so a flexible variance
    head cannot hide a poor conditional mean. The dynamic low-rank score is
    introduced gradually by the trainer. A positive weighted sum of the two
    Gaussian log scores remains proper for their shared conditional mean.
    """

    if not 0.0 <= covariance_weight <= 1.0:
        raise ValueError("covariance_weight must lie in [0,1].")
    if validate_profile:
        profile.validate(n_channels=observation.shape[1])
    batch_size, _, _ = _validate_prequential_shapes(observation, innovation, profile)
    effective_mask = _effective_observation_mask(observation, profile, observation_mask)
    if mask_is_homogeneous is None:
        mask_is_homogeneous = observation_mask is None or observation_mask.dim() == 1
    finite_observed = torch.isfinite(observation) | ~effective_mask[:, :, None]
    if not bool(finite_observed.all()):
        raise ValueError(
            "Observed likelihood samples must be finite; mark missing channels explicitly."
        )
    indices = _class_indices(
        batch_size,
        observation.device,
        labels=labels,
        hypothesis=None,
    )
    mean = _prepare_prequential_mean(
        observation,
        innovation,
        profile,
        indices,
        effective_mask,
        mean_model="neural",
    )
    mean_trial_score = _score_prepared_mean(
        observation,
        innovation,
        profile,
        indices,
        effective_mask,
        mean,
        mean_model="neural",
        covariance_model="static",
        mask_is_homogeneous=mask_is_homogeneous,
    )
    mean_score = _class_balanced_prequential_nll(mean_trial_score, labels)
    if covariance_weight == 0.0:
        return mean_score
    covariance_trial_score = _score_prepared_mean(
        observation,
        innovation,
        profile,
        indices,
        effective_mask,
        mean,
        mean_model="neural",
        covariance_model="low_rank_dynamic",
        mask_is_homogeneous=mask_is_homogeneous,
    )
    covariance_score = _class_balanced_prequential_nll(covariance_trial_score, labels)
    return (mean_score + covariance_weight * covariance_score) / (1.0 + covariance_weight)


def prequential_log_likelihood_ratio(
    observation: torch.Tensor,
    innovation: CausalInnovationOutput,
    profile: GenerativeProfile,
    *,
    variant: PrequentialVariant,
    observation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``log p(x|y=1) - log p(x|y=0)`` for each trial."""

    negative = prequential_score_per_trial(
        observation,
        innovation,
        profile,
        hypothesis=0,
        variant=variant,
        observation_mask=observation_mask,
    )
    positive = prequential_score_per_trial(
        observation,
        innovation,
        profile,
        hypothesis=1,
        variant=variant,
        observation_mask=observation_mask,
    )
    return negative.nll_sum - positive.nll_sum
