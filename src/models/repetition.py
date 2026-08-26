"""LEGACY_V11: causal probabilistic evidence accumulation across GTN repetitions.

Only ``NEURAL_RIDE_V11_LEGACY`` instantiates this module. The active v12
recipe uses ``models.repetition_v12.AdditiveRepetitionEvidence``; the legacy
soft-BCE(0.9/0.1) reliability anchor is kept for historical comparison only.

The trial classifier produces an evidence variable ``e``.  This module models
its class-conditional density instead of multiplying its logit by a learned
quality weight.  Artifact contamination is class independent:

    p(e | z, h) = rho * p_clean(e | z, h) + (1-rho) * p_artifact(e)

Consequently the conditional LLR is exactly zero when ``rho == 0``.  A small
GRU carries acquisition-order dependence; it never emits an unconstrained set
score.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

QUALITY_FEATURE_NAMES = (
    "baseline_variance",
    "baseline_peak_to_peak",
    "baseline_drift",
    "line_noise_ratio",
    "high_frequency_ratio",
    "temporal_jump_ratio",
    "flatline_fraction",
    "missing_channel_fraction",
)


def corrected_trial_evidence(
    weighted_logits: torch.Tensor,
    *,
    pos_weight: float,
    train_prior: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Remove weighted-BCE and training-prior offsets, then temperature scale."""

    if pos_weight <= 0.0:
        raise ValueError("pos_weight must be positive.")
    if not 0.0 < train_prior < 1.0:
        raise ValueError("train_prior must lie strictly between zero and one.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    prior_log_odds = math.log(train_prior / (1.0 - train_prior))
    offset = math.log(pos_weight) + prior_log_odds
    return (weighted_logits - offset) / temperature


def extract_quality_features(
    signal: torch.Tensor,
    *,
    sfreq: float,
    baseline_n: int,
    reference_slice: tuple[int, int] | None = None,
    reconstruction_residual: torch.Tensor | None = None,
    channel_mask: torch.Tensor | None = None,
    check_finite: bool = True,
) -> torch.Tensor:
    """Compute class-independent per-trial artifact indicators.

    Inputs are expected in the dataset's physical signal space, before the
    model's per-trial baseline standardization. Log amplitudes and scale-free
    ratios are subsequently robust-normalized on optimization subjects.
    """

    if signal.dim() != 3:
        raise ValueError(f"signal must be (B,C,T), got {tuple(signal.shape)}.")
    if sfreq <= 0.0:
        raise ValueError("sfreq must be positive.")
    batch, channels, n_time = signal.shape
    if n_time < 4:
        raise ValueError("quality extraction needs at least four time samples.")
    if channel_mask is None:
        observed_mask = torch.ones(batch, channels, dtype=torch.bool, device=signal.device)
    else:
        observed_mask = channel_mask.to(device=signal.device, dtype=torch.bool)
        if observed_mask.shape == (channels,):
            observed_mask = observed_mask[None].expand(batch, -1)
        if observed_mask.shape != (batch, channels):
            raise ValueError("channel_mask must be (C,) or (B,C).")
    channel_weight = observed_mask.float()
    observed_count = channel_weight.sum(dim=1).clamp_min(1.0)
    finite = torch.isfinite(signal) | ~observed_mask[:, :, None]
    if check_finite and not bool(finite.all()):
        raise ValueError(
            "Observed quality-feature samples must be finite; mark missing channels explicitly."
        )
    x = torch.where(
        observed_mask[:, :, None],
        signal.float(),
        torch.zeros_like(signal, dtype=torch.float32),
    )
    if reference_slice is None:
        baseline_n = int(baseline_n)
        if baseline_n < 2:
            baseline_n = max(2, min(n_time // 5, n_time - 1))
        baseline_n = min(baseline_n, n_time - 1)
        baseline = x[..., :baseline_n]
    else:
        if len(reference_slice) != 2:
            raise ValueError("reference_slice must be (start, stop).")
        start, stop = (int(reference_slice[0]), int(reference_slice[1]))
        if start < 0 or stop > n_time or stop - start < 2:
            raise ValueError("reference_slice must contain at least two in-range samples.")
        baseline = x[..., start:stop]
        baseline_n = stop - start
    if reconstruction_residual is not None:
        if reconstruction_residual.shape != signal.shape:
            raise ValueError("reconstruction_residual must match signal shape.")

    # Reliability inputs must be nuisance indicators, not proxies for the
    # evoked target response. Amplitude summaries are therefore pre-stimulus.
    positive_floor = torch.finfo(x.dtype).tiny
    baseline_variance = torch.log(
        (
            (baseline.var(dim=-1, unbiased=False) * channel_weight).sum(dim=1) / observed_count
        ).clamp_min(positive_floor)
    )
    baseline_peak_to_peak = torch.log(
        (
            ((baseline.amax(dim=-1) - baseline.amin(dim=-1)).abs() * channel_weight).sum(dim=1)
            / observed_count
        ).clamp_min(positive_floor)
    )
    split = max(1, baseline_n // 2)
    baseline_drift = torch.log(
        (
            (baseline[..., :split].mean(dim=-1) - baseline[..., -split:].mean(dim=-1))
            .abs()
            .mul(channel_weight)
            .sum(dim=1)
            / observed_count
        ).clamp_min(positive_floor)
    )

    spectrum = torch.fft.rfft(x, dim=-1)
    power = (spectrum.abs().square() * channel_weight[:, :, None]).sum(dim=1) / observed_count[
        :, None
    ]
    freqs_np = np.fft.rfftfreq(n_time, d=1.0 / float(sfreq))
    positive_np = freqs_np > 0.0
    nuisance_np = freqs_np >= min(15.0, 0.45 * float(sfreq))
    if not nuisance_np.any():
        nuisance_np = positive_np.copy()
    total_power = power[:, torch.as_tensor(nuisance_np, device=x.device)].mean(dim=1).clamp_min(positive_floor)
    line_np = ((freqs_np >= 48.0) & (freqs_np <= 52.0)) | ((freqs_np >= 58.0) & (freqs_np <= 62.0))
    if not line_np.any():
        line_np = positive_np.copy()
    high_np = (freqs_np >= min(30.0, 0.30 * float(sfreq))) & positive_np
    if not high_np.any():
        high_np = positive_np.copy()
    line_mask = torch.as_tensor(line_np, device=x.device)
    high_mask = torch.as_tensor(high_np, device=x.device)

    def _band_ratio(mask: torch.Tensor) -> torch.Tensor:
        return torch.log1p(100.0 * power[:, mask].mean(dim=1) / total_power)

    line_noise_ratio = _band_ratio(line_mask)
    high_frequency_ratio = _band_ratio(high_mask)

    absolute_diff = x.diff(dim=-1).abs()
    mean_jump = (
        absolute_diff.mean(dim=-1).mul(channel_weight).sum(dim=1) / observed_count
    ).clamp_min(positive_floor)
    max_jump = absolute_diff.amax(dim=-1).mul(channel_weight).sum(dim=1) / observed_count
    temporal_jump_ratio = torch.log1p(max_jump / mean_jump)

    channel_std = x.std(dim=-1, unbiased=False)
    reference_std = channel_std.median(dim=1).values.clamp_min(positive_floor)
    flatline_fraction = (
        (channel_std <= (0.02 * reference_std[:, None]).clamp_min(1e-12)) & observed_mask
    ).float().sum(dim=1) / observed_count
    missing_fraction = 1.0 - channel_weight.mean(dim=1)

    features = torch.stack(
        (
            baseline_variance,
            baseline_peak_to_peak,
            baseline_drift,
            line_noise_ratio,
            high_frequency_ratio,
            temporal_jump_ratio,
            flatline_fraction,
            missing_fraction,
        ),
        dim=-1,
    )
    if check_finite and not bool(torch.isfinite(features).all()):
        raise ValueError("Quality feature extraction produced non-finite values.")
    return features


def _student_t_log_prob(
    value: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    df: torch.Tensor,
) -> torch.Tensor:
    """Stable Student-t log density with broadcast-compatible tensors."""

    scale = scale.clamp_min(1e-4)
    df = df.clamp_min(2.01)
    standardized = (value - loc) / scale
    return (
        torch.lgamma((df + 1.0) / 2.0)
        - torch.lgamma(df / 2.0)
        - 0.5 * (torch.log(df) + math.log(math.pi))
        - torch.log(scale)
        - 0.5 * (df + 1.0) * torch.log1p(standardized.square() / df)
    )


def _mixture_log_prob(
    clean_log_prob: torch.Tensor,
    artifact_log_prob: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Log mixture that preserves exact endpoints at rho=0 and rho=1."""

    rho = reliability.clamp(0.0, 1.0)
    neg_inf = torch.full_like(rho, -torch.inf)
    log_clean_weight = torch.where(rho > 0.0, torch.log(rho), neg_inf)
    log_artifact_weight = torch.where(rho < 1.0, torch.log1p(-rho), neg_inf)
    return torch.logaddexp(
        log_clean_weight[..., None] + clean_log_prob,
        log_artifact_weight[..., None] + artifact_log_prob[..., None],
    )


@dataclass(frozen=True)
class SequenceEvidenceOutput:
    observed_log_prob: torch.Tensor
    conditional_llr: torch.Tensor
    reliability: torch.Tensor
    posterior_clean_probability: torch.Tensor


class RepetitionEvidenceModel(nn.Module):
    """Small causal GRU with clean and shared-artifact Student-t densities."""

    def __init__(
        self,
        *,
        n_quality_features: int = len(QUALITY_FEATURE_NAMES),
        hidden_size: int = 24,
        reliability_hidden: int = 16,
    ) -> None:
        super().__init__()
        if n_quality_features < 1 or hidden_size < 1:
            raise ValueError("quality feature count and hidden size must be positive.")
        self.n_quality_features = int(n_quality_features)
        self.hidden_size = int(hidden_size)

        self.reliability_net = nn.Sequential(
            nn.Linear(self.n_quality_features, reliability_hidden),
            nn.SiLU(),
            nn.Linear(reliability_hidden, 1),
        )
        nn.init.zeros_(self.reliability_net[-1].weight)
        nn.init.constant_(self.reliability_net[-1].bias, math.log(0.9 / 0.1))

        self.gru = nn.GRU(
            self.n_quality_features + 3,
            self.hidden_size,
            batch_first=True,
        )
        self.clean_context = nn.Sequential(
            nn.Linear(self.hidden_size + self.n_quality_features, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6),
        )
        nn.init.zeros_(self.clean_context[-1].weight)
        nn.init.zeros_(self.clean_context[-1].bias)

        self.clean_loc = nn.Parameter(torch.tensor([-1.0, 1.0]))
        self.clean_raw_scale = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.clean_raw_df = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.artifact_raw_scale = nn.Parameter(torch.tensor(0.0))
        self.artifact_raw_df = nn.Parameter(torch.tensor(0.5))

        self.register_buffer("artifact_loc", torch.tensor(0.0))
        self.register_buffer("quality_center", torch.zeros(self.n_quality_features))
        self.register_buffer("quality_scale", torch.ones(self.n_quality_features))
        self.register_buffer("quality_normalizer_fitted", torch.tensor(False))
        self.register_buffer("evidence_pos_weight", torch.tensor(1.0))
        self.register_buffer("evidence_train_prior", torch.tensor(0.5))
        self.register_buffer("evidence_temperature", torch.tensor(1.0))

    def set_evidence_calibration(
        self,
        *,
        pos_weight: float,
        train_prior: float,
        temperature: float = 1.0,
    ) -> None:
        corrected_trial_evidence(
            torch.zeros(()),
            pos_weight=pos_weight,
            train_prior=train_prior,
            temperature=temperature,
        )
        self.evidence_pos_weight.fill_(float(pos_weight))
        self.evidence_train_prior.fill_(float(train_prior))
        self.evidence_temperature.fill_(float(temperature))

    def correct_evidence(self, weighted_logits: torch.Tensor) -> torch.Tensor:
        prior_log_odds = torch.logit(self.evidence_train_prior.clamp(1e-6, 1.0 - 1e-6))
        offset = torch.log(self.evidence_pos_weight.clamp_min(1e-6)) + prior_log_odds
        return (weighted_logits - offset) / self.evidence_temperature.clamp_min(1e-4)

    @torch.no_grad()
    def fit_quality_normalizer(self, quality: torch.Tensor) -> None:
        """Freeze a robust transform estimated from optimization subjects only."""

        values = quality.detach().float().reshape(-1, self.n_quality_features)
        if values.shape[0] < 2 or not bool(torch.isfinite(values).all()):
            raise ValueError("quality normalization needs at least two finite rows.")
        center = values.median(dim=0).values
        q25 = torch.quantile(values, 0.25, dim=0)
        q75 = torch.quantile(values, 0.75, dim=0)
        scale = ((q75 - q25) / 1.349).clamp_min(1e-3)
        self.quality_center.copy_(center.to(self.quality_center))
        self.quality_scale.copy_(scale.to(self.quality_scale))
        self.quality_normalizer_fitted.fill_(True)

    def normalize_quality(self, quality: torch.Tensor) -> torch.Tensor:
        return ((quality - self.quality_center) / self.quality_scale).clamp(-8.0, 8.0)

    def reliability(self, quality: torch.Tensor) -> torch.Tensor:
        if quality.shape[-1] != self.n_quality_features:
            raise ValueError(
                f"quality needs {self.n_quality_features} features, got {quality.shape[-1]}."
            )
        return torch.sigmoid(self.reliability_net(self.normalize_quality(quality)).squeeze(-1))

    def reliability_identification_loss(self, quality: torch.Tensor) -> torch.Tensor:
        """Anchor rho with class-blind, feature-space artifact interventions."""

        normalized = self.normalize_quality(quality)
        clean_mask = normalized.abs().amax(dim=-1) < 3.0
        clean_loss = normalized.sum() * 0.0
        if bool(clean_mask.any()):
            clean_logits = self.reliability_net(normalized[clean_mask]).squeeze(-1)
            clean_loss = functional.binary_cross_entropy_with_logits(
                clean_logits, torch.full_like(clean_logits, 0.9)
            )
        corrupted = normalized.detach().clone()
        rows = torch.arange(corrupted.shape[0], device=corrupted.device)
        feature = rows.remainder(self.n_quality_features)
        corrupted[rows, feature] = torch.maximum(
            corrupted[rows, feature], corrupted.new_full((len(rows),), 6.0)
        )
        artifact_logits = self.reliability_net(corrupted).squeeze(-1)
        artifact_loss = functional.binary_cross_entropy_with_logits(
            artifact_logits, torch.full_like(artifact_logits, 0.1)
        )
        return 0.5 * (clean_loss + artifact_loss)

    def _class_log_prob(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        hidden: torch.Tensor,
        *,
        reliability_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context_quality = self.normalize_quality(quality)
        raw = self.clean_context(torch.cat((hidden, context_quality), dim=-1)).view(-1, 2, 3)
        clean_loc = self.clean_loc[None] + 2.0 * torch.tanh(raw[..., 0])
        clean_scale = 0.05 + 2.95 * torch.sigmoid(self.clean_raw_scale[None] + raw[..., 1])
        clean_df = 4.01 + functional.softplus(self.clean_raw_df[None] + raw[..., 2])
        clean_log_prob = _student_t_log_prob(evidence[:, None], clean_loc, clean_scale, clean_df)

        artifact_scale = functional.softplus(self.artifact_raw_scale) + 3.05
        artifact_df = 2.01 + 1.99 * torch.sigmoid(self.artifact_raw_df)
        artifact_log_prob = _student_t_log_prob(
            evidence,
            self.artifact_loc.expand_as(evidence),
            artifact_scale.expand_as(evidence),
            artifact_df.expand_as(evidence),
        )
        rho = (
            self.reliability(quality)
            if reliability_override is None
            else reliability_override.to(device=evidence.device, dtype=evidence.dtype)
        )
        mixture = _mixture_log_prob(clean_log_prob, artifact_log_prob, rho)
        return mixture, rho, clean_log_prob, artifact_log_prob

    def _sequence_log_prob(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        labels: torch.Tensor,
        *,
        reliability_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate paths using a posterior-clean robust recurrent filter."""

        if labels.dim() != 2:
            raise ValueError("labels must be (n_paths, sequence_length).")
        n_paths, length = labels.shape
        if evidence.shape != (length,) or quality.shape != (
            length,
            self.n_quality_features,
        ):
            raise ValueError("evidence/quality must align with labels' sequence length.")
        output = self.forward_batched_sequences(
            evidence[None].expand(n_paths, -1),
            quality[None].expand(n_paths, -1, -1),
            labels,
            evidence.new_full((n_paths,), length, dtype=torch.long),
            reliability_override=(
                None
                if reliability_override is None
                else reliability_override[None].expand(n_paths, -1)
            ),
        )
        return (
            output.observed_log_prob,
            output.conditional_llr,
            output.reliability,
            output.posterior_clean_probability,
        )

    def forward_batched_sequences(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
        *,
        reliability_override: torch.Tensor | None = None,
    ) -> SequenceEvidenceOutput:
        """Evaluate padded causal paths together while masking ended sequences."""

        if evidence.dim() != 2 or labels.shape != evidence.shape:
            raise ValueError("evidence and labels must share shape (n_paths, max_length).")
        n_paths, max_length = evidence.shape
        if quality.shape != (n_paths, max_length, self.n_quality_features):
            raise ValueError("quality must be (n_paths, max_length, n_quality_features).")
        lengths = lengths.to(device=evidence.device, dtype=torch.long).reshape(-1)
        if lengths.numel() != n_paths or bool(((lengths < 1) | (lengths > max_length)).any()):
            raise ValueError("lengths must contain one valid length per path.")
        labels = labels.to(device=evidence.device, dtype=evidence.dtype)
        active_steps = torch.arange(max_length, device=evidence.device)[None] < lengths[:, None]
        if bool((((labels < 0) | (labels > 1)) & active_steps).any()):
            raise ValueError("labels must be binary on active sequence steps.")
        if reliability_override is None:
            rho_all = self.reliability(quality.reshape(-1, self.n_quality_features)).reshape(
                n_paths, max_length
            )
        else:
            rho_all = torch.as_tensor(
                reliability_override, device=evidence.device, dtype=evidence.dtype
            )
            if rho_all.numel() == 1:
                rho_all = rho_all.expand(n_paths, max_length)
            elif rho_all.shape != (n_paths, max_length):
                raise ValueError("reliability_override must be scalar or (n_paths, max_length).")
        normalized_quality = self.normalize_quality(quality)
        artifact_scale = functional.softplus(self.artifact_raw_scale) + 3.05
        artifact_df = 2.01 + 1.99 * torch.sigmoid(self.artifact_raw_df)
        artifact_log_prob = _student_t_log_prob(
            evidence,
            self.artifact_loc.expand_as(evidence),
            artifact_scale.expand_as(evidence),
            artifact_df.expand_as(evidence),
        )

        hidden = evidence.new_zeros((1, n_paths, self.hidden_size))
        observed_steps: list[torch.Tensor] = []
        llr_steps: list[torch.Tensor] = []
        rho_steps: list[torch.Tensor] = []
        posterior_steps: list[torch.Tensor] = []
        for step in range(max_length):
            step_evidence = evidence[:, step]
            step_quality = normalized_quality[:, step]
            raw = self.clean_context(torch.cat((hidden.squeeze(0), step_quality), dim=-1)).view(
                -1, 2, 3
            )
            clean_loc = self.clean_loc[None] + 2.0 * torch.tanh(raw[..., 0])
            clean_scale = 0.05 + 2.95 * torch.sigmoid(self.clean_raw_scale[None] + raw[..., 1])
            clean_df = 4.01 + functional.softplus(self.clean_raw_df[None] + raw[..., 2])
            clean_log_prob = _student_t_log_prob(
                step_evidence[:, None], clean_loc, clean_scale, clean_df
            )
            rho = rho_all[:, step]
            mixture = _mixture_log_prob(clean_log_prob, artifact_log_prob[:, step], rho)
            step_labels = labels[:, step].long()
            selected = torch.gather(mixture, 1, step_labels[:, None]).squeeze(1)
            selected_clean = torch.gather(clean_log_prob, 1, step_labels[:, None]).squeeze(1)
            log_rho = torch.where(
                rho > 0.0,
                torch.log(rho),
                torch.full_like(rho, -torch.inf),
            )
            posterior_clean = torch.exp(log_rho + selected_clean - selected).clamp(0.0, 1.0)
            gru_input = torch.cat(
                (
                    posterior_clean[:, None] * (step_evidence / 5.0).clamp(-4.0, 4.0)[:, None],
                    posterior_clean[:, None] * step_quality,
                    labels[:, step, None],
                    posterior_clean[:, None],
                ),
                dim=-1,
            )
            _, next_hidden = self.gru(gru_input[:, None], hidden)
            active = active_steps[:, step]
            hidden = torch.where(active[None, :, None], next_hidden, hidden.detach())
            observed_steps.append(torch.where(active, selected, torch.zeros_like(selected)))
            llr_steps.append(
                torch.where(active, mixture[:, 1] - mixture[:, 0], torch.zeros_like(selected))
            )
            rho_steps.append(torch.where(active, rho, torch.zeros_like(rho)))
            posterior_steps.append(
                torch.where(active, posterior_clean, torch.zeros_like(posterior_clean))
            )
        return SequenceEvidenceOutput(
            observed_log_prob=torch.stack(observed_steps, dim=1),
            conditional_llr=torch.stack(llr_steps, dim=1),
            reliability=torch.stack(rho_steps, dim=1),
            posterior_clean_probability=torch.stack(posterior_steps, dim=1),
        )

    def forward_sequence(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        labels: torch.Tensor,
        *,
        reliability_override: float | torch.Tensor | None = None,
    ) -> SequenceEvidenceOutput:
        """Evaluate one observed label path in acquisition order."""

        evidence = evidence.reshape(-1)
        labels = labels.reshape(-1).to(device=evidence.device, dtype=evidence.dtype)
        if quality.shape != (evidence.numel(), self.n_quality_features):
            raise ValueError("quality must be (sequence_length, n_quality_features).")
        if labels.numel() != evidence.numel() or bool(((labels < 0) | (labels > 1)).any()):
            raise ValueError("labels must be aligned binary values.")
        override = None
        if reliability_override is not None:
            override = torch.as_tensor(
                reliability_override, device=evidence.device, dtype=evidence.dtype
            ).expand(evidence.numel())
        observed, llr, rho, posterior_clean = self._sequence_log_prob(
            evidence,
            quality,
            labels[None],
            reliability_override=override,
        )
        return SequenceEvidenceOutput(
            observed_log_prob=observed[0],
            conditional_llr=llr[0],
            reliability=rho[0],
            posterior_clean_probability=posterior_clean[0],
        )

    def candidate_log_scores(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        stimulus_digits: torch.Tensor,
        *,
        digit_vocab: Sequence[int] = tuple(range(1, 10)),
        log_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the probability chain rule to every candidate digit."""

        trajectory, reliability = self.candidate_log_score_trajectory(
            evidence,
            quality,
            stimulus_digits,
            digit_vocab=digit_vocab,
            log_prior=log_prior,
        )
        return trajectory[:, -1], reliability

    def candidate_log_score_trajectory(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        stimulus_digits: torch.Tensor,
        *,
        digit_vocab: Sequence[int] = tuple(range(1, 10)),
        log_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cumulative candidate scores at every acquisition checkpoint."""

        evidence = evidence.reshape(-1)
        stimulus_digits = stimulus_digits.reshape(-1).to(evidence.device, dtype=torch.long)
        if quality.shape != (evidence.numel(), self.n_quality_features):
            raise ValueError("quality must align with the evidence sequence.")
        if stimulus_digits.numel() != evidence.numel():
            raise ValueError("stimulus_digits must align with evidence.")
        vocab = torch.as_tensor(digit_vocab, device=evidence.device, dtype=torch.long)
        if vocab.dim() != 1 or vocab.numel() < 2 or vocab.unique().numel() != vocab.numel():
            raise ValueError("digit_vocab must contain unique candidate values.")
        n_candidates = int(vocab.numel())
        if log_prior is None:
            scores = evidence.new_full((n_candidates,), -math.log(n_candidates))
        else:
            scores = log_prior.to(device=evidence.device, dtype=evidence.dtype).reshape(-1)
            if scores.numel() != n_candidates:
                raise ValueError("log_prior must have one value per candidate.")

        labels = (stimulus_digits[None] == vocab[:, None]).to(evidence.dtype)
        selected, _, rho, _ = self._sequence_log_prob(evidence, quality, labels)
        trajectory = scores[:, None] + selected.cumsum(dim=1)
        return trajectory, rho[0]
