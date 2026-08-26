"""v12 RepetitionEvidence object: additive LLR backbone plus shrinkable state residual.

The legacy ``RepetitionEvidenceModel`` remains importable for historical
comparisons only. New research code must use ``AdditiveRepetitionEvidence``:

* the backbone scores every flash with class-conditional densities whose
  context depends only on ``(evidence, quality)`` -- no candidate-specific
  hidden state -- so the candidate score is an additive LLR over all positive
  and negative flashes;
* the optional state residual is initialized at exactly zero and must prove
  held-out incremental log-score value before its gain is allowed to move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from models.reliability_v12 import FidelityEstimator
from models.repetition import (
    QUALITY_FEATURE_NAMES,
    _mixture_log_prob,
    _student_t_log_prob,
    corrected_trial_evidence,
)


@dataclass(frozen=True)
class AdditiveSequenceEvidenceOutput:
    observed_log_prob: torch.Tensor
    conditional_llr: torch.Tensor
    reliability: torch.Tensor

    state_residual_energy: torch.Tensor


class AdditiveLLRBackbone(nn.Module):
    """Static, quality-conditioned class densities.

    The clean density parameters are functions of normalized quality features
    only. There is deliberately no recurrent hidden input, which makes the
    per-flash ``conditional_llr`` additive across candidate paths.
    """

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

        # Fidelity/quality gate. Its probability semantics are finalized in the
        # Q object; here it is only the mixture weight required by the density.
        self.fidelity_estimator = FidelityEstimator(
            n_features=self.n_quality_features,
            hidden_size=reliability_hidden,
        )

        self.clean_context = nn.Sequential(
            nn.Linear(self.n_quality_features, self.hidden_size),
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

    @torch.no_grad()
    def fit_quality_normalizer(self, quality: torch.Tensor) -> None:
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
        self.fidelity_estimator.fit_normalizer(quality)

    def normalize_quality(self, quality: torch.Tensor) -> torch.Tensor:
        return ((quality - self.quality_center) / self.quality_scale).clamp(-8.0, 8.0)

    def fidelity_score(self, quality: torch.Tensor) -> torch.Tensor:
        if quality.shape[-1] != self.n_quality_features:
            raise ValueError(
                f"quality needs {self.n_quality_features} features, got {quality.shape[-1]}."
            )
        return self.fidelity_estimator(quality)

    def fidelity(self, quality: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fidelity_score(quality)).clamp(0.0, 1.0)

    def fidelity_margin_rank_loss(self, quality: torch.Tensor) -> torch.Tensor:
        return self.fidelity_estimator.fidelity_margin_rank_loss(quality)

    def class_log_probs(
        self, evidence: torch.Tensor, quality: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(clean_log_prob (N,2), artifact_log_prob (N,))``."""
        context_quality = self.normalize_quality(quality)
        raw = self.clean_context(context_quality).view(-1, 2, 3)
        clean_loc = self.clean_loc[None] + 2.0 * torch.tanh(raw[..., 0])
        clean_scale = 0.05 + 2.95 * torch.sigmoid(self.clean_raw_scale[None] + raw[..., 1])
        clean_df = 4.01 + functional.softplus(self.clean_raw_df[None] + raw[..., 2])
        clean_log_prob = _student_t_log_prob(
            evidence[:, None], clean_loc, clean_scale, clean_df
        )
        artifact_scale = functional.softplus(self.artifact_raw_scale) + 3.05
        artifact_df = 2.01 + 1.99 * torch.sigmoid(self.artifact_raw_df)
        artifact_log_prob = _student_t_log_prob(
            evidence,
            self.artifact_loc.expand_as(evidence),
            artifact_scale.expand_as(evidence),
            artifact_df.expand_as(evidence),
        )
        return clean_log_prob, artifact_log_prob


class StateResidual(nn.Module):
    """Candidate-path recurrent correction initialized at exactly zero."""

    def __init__(self, *, n_quality_features: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_quality_features = int(n_quality_features)
        self.gru = nn.GRU(n_quality_features + 2, self.hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size + n_quality_features, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # Fail-closed: the residual is exactly zero until a gate enables it.
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        labels: torch.Tensor,

        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(delta (N,2), next_hidden, energy)``. Context is exactly
        ``(evidence, quality, 1(d==c))`` per blueprint 3.2; posterior-clean is
        not an input to the recurrent correction.
        """
        normalized_quality = quality
        gru_input = torch.cat(
            (
                (evidence / 5.0).clamp(-4.0, 4.0)[:, None],
                normalized_quality,
                labels[:, None],

            ),
            dim=-1,
        )
        _, next_hidden = self.gru(gru_input[:, None], hidden)
        raw = self.head(torch.cat((hidden.squeeze(0), normalized_quality), dim=-1))
        delta = self.gain * torch.tanh(raw)
        energy = (delta * delta).mean()
        return delta, next_hidden, energy


class AdditiveRepetitionEvidence(nn.Module):
    """Additive LLR repetition evidence with an optional shrink-to-zero state residual."""

    def __init__(
        self,
        *,
        n_quality_features: int = len(QUALITY_FEATURE_NAMES),
        hidden_size: int = 24,
        reliability_hidden: int = 16,
        state_residual: bool = False,
    ) -> None:
        super().__init__()
        self.n_quality_features = int(n_quality_features)
        self.hidden_size = int(hidden_size)
        self.backbone = AdditiveLLRBackbone(
            n_quality_features=self.n_quality_features,
            hidden_size=self.hidden_size,
            reliability_hidden=reliability_hidden,
        )
        self.state_residual = (
            StateResidual(n_quality_features=self.n_quality_features, hidden_size=self.hidden_size)
            if state_residual
            else None
        )
        self.register_buffer("evidence_pos_weight", torch.tensor(1.0))
        self.register_buffer("evidence_train_prior", torch.tensor(0.5))
        self.register_buffer("evidence_temperature", torch.tensor(1.0))

    # Calibration contract mirrors the legacy evidence model.
    def set_evidence_calibration(
        self, *, pos_weight: float, train_prior: float, temperature: float = 1.0
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

    def fit_quality_normalizer(self, quality: torch.Tensor) -> None:
        self.backbone.fit_quality_normalizer(quality)

    def fidelity(self, quality: torch.Tensor) -> torch.Tensor:
        return self.backbone.fidelity(quality)

    def fidelity_score(self, quality: torch.Tensor) -> torch.Tensor:
        return self.backbone.fidelity_score(quality)

    def fidelity_margin_rank_loss(self, quality: torch.Tensor) -> torch.Tensor:
        return self.backbone.fidelity_margin_rank_loss(quality)

    def reliability(self, quality: torch.Tensor) -> torch.Tensor:
        """Compatibility alias until the Q object replaces this semantic."""
        return self.fidelity(quality)

    def state_residual_gain_value(self) -> float:
        if self.state_residual is None:
            return 0.0
        return float(self.state_residual.gain.detach().cpu())

    @torch.no_grad()
    def set_state_residual_gain(self, gain: float) -> None:
        if self.state_residual is None:
            raise RuntimeError("state residual is not instantiated.")
        self.state_residual.gain.fill_(float(gain))

    @torch.no_grad()
    def prepare_state_residual_for_refit(self, gain: float = 0.1) -> None:
        """Give the residual a small nonzero gain so its parameters receive gradients."""
        if self.state_residual is not None:
            self.state_residual.gain.fill_(float(gain))

    def forward_batched_sequences(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
        *,
        reliability_override: torch.Tensor | None = None,
    ) -> AdditiveSequenceEvidenceOutput:
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
            rho_all = self.fidelity(quality.reshape(-1, self.n_quality_features)).reshape(
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

        normalized_quality = self.backbone.normalize_quality(quality)
        hidden = evidence.new_zeros((1, n_paths, self.hidden_size))
        observed_steps: list[torch.Tensor] = []
        llr_steps: list[torch.Tensor] = []
        rho_steps: list[torch.Tensor] = []

        residual_energy_steps: list[torch.Tensor] = []

        for step in range(max_length):
            step_evidence = evidence[:, step]
            step_quality = normalized_quality[:, step]
            step_labels = labels[:, step].long()
            rho = rho_all[:, step]

            clean_log_prob, artifact_log_prob = self.backbone.class_log_probs(
                step_evidence, step_quality
            )

            energy = step_evidence.sum() * 0.0

            if self.state_residual is not None:
                delta, next_hidden, energy = self.state_residual(
                    step_evidence,
                    step_quality,
                    labels[:, step],

                    hidden,
                )
                clean_log_prob = clean_log_prob + delta
            else:
                next_hidden = hidden.detach()

            mixture = _mixture_log_prob(clean_log_prob, artifact_log_prob, rho)
            selected = torch.gather(mixture, 1, step_labels[:, None]).squeeze(1)

            active = active_steps[:, step]
            hidden = torch.where(active[None, :, None], next_hidden, hidden.detach())
            observed_steps.append(torch.where(active, selected, torch.zeros_like(selected)))
            llr_steps.append(
                torch.where(active, mixture[:, 1] - mixture[:, 0], torch.zeros_like(selected))
            )
            rho_steps.append(torch.where(active, rho, torch.zeros_like(rho)))
            residual_energy_steps.append(torch.where(active, energy, torch.zeros_like(energy)))

        return AdditiveSequenceEvidenceOutput(
            observed_log_prob=torch.stack(observed_steps, dim=1),
            conditional_llr=torch.stack(llr_steps, dim=1),
            reliability=torch.stack(rho_steps, dim=1),
            state_residual_energy=torch.stack(residual_energy_steps, dim=1).mean(),
        )

    def _sequence_log_prob(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        labels: torch.Tensor,
        *,
        reliability_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

            output.state_residual_energy,
        )

    def forward_sequence(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        labels: torch.Tensor,
        *,
        reliability_override: torch.Tensor | None = None,
    ) -> AdditiveSequenceEvidenceOutput:
        evidence = evidence.reshape(-1)
        labels = labels.reshape(-1).to(device=evidence.device, dtype=evidence.dtype)
        if quality.shape != (evidence.numel(), self.n_quality_features):
            raise ValueError("quality must be (sequence_length, n_quality_features).")
        override = None
        if reliability_override is not None:
            override = torch.as_tensor(
                reliability_override, device=evidence.device, dtype=evidence.dtype
            ).expand(evidence.numel())
        observed, llr, rho, energy = self._sequence_log_prob(
            evidence, quality, labels[None], reliability_override=override
        )
        return AdditiveSequenceEvidenceOutput(
            observed_log_prob=observed[0],
            conditional_llr=llr[0],
            reliability=rho[0],

            state_residual_energy=energy,
        )

    def candidate_log_score_trajectory(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        stimulus_digits: torch.Tensor,
        *,
        digit_vocab: tuple[int, ...] = tuple(range(1, 10)),
        log_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cumulative additive LLR over all positive and negative flashes.

        The returned values are *centered candidate scores*: they omit the
        candidate-independent ``sum_t log p0(e_t)`` baseline. For a common
        acquisition prefix this baseline is identical for every candidate, so
        argmax and cross-entropy are invariant; the values are not normalized
        chain log-probabilities and must not be fed to NLL/Brier directly.
        Use ``candidate_chain_scores_with_clean_probability`` for the full
        probability path."""
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
        _, llr, rho, _ = self._sequence_log_prob(evidence, quality, labels)
        trajectory = scores[:, None] + (llr * labels).cumsum(dim=1)
        return trajectory, rho[0]

    def candidate_log_scores(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        stimulus_digits: torch.Tensor,
        *,
        digit_vocab: tuple[int, ...] = tuple(range(1, 10)),
        log_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory, reliability = self.candidate_log_score_trajectory(
            evidence,
            quality,
            stimulus_digits,
            digit_vocab=digit_vocab,
            log_prior=log_prior,
        )
        return trajectory[:, -1], reliability


    @torch.no_grad()
    def candidate_chain_scores_with_clean_probability(
        self,
        evidence: torch.Tensor,
        quality: torch.Tensor,
        stimulus_digits: torch.Tensor,
        clean_probability: torch.Tensor,
        *,
        digit_vocab: tuple[int, ...] = tuple(range(1, 10)),
        log_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full digit-chain scores using an externally calibrated clean probability.

        This is the blueprint-4.2 acceptance path: the same additive backbone
        densities are mixed with the calibrated ``rho`` per flash, and the
        complete candidate path score is returned for chain softmax/NLL/decision
        re-verification. It intentionally does not mutate the fitted model.
        """
        evidence = evidence.reshape(-1)
        stimulus_digits = stimulus_digits.reshape(-1).to(evidence.device, dtype=torch.long)
        rho = torch.as_tensor(
            clean_probability, device=evidence.device, dtype=evidence.dtype
        ).reshape(-1)
        if quality.shape != (evidence.numel(), self.n_quality_features):
            raise ValueError("quality must align with the evidence sequence.")
        if stimulus_digits.numel() != evidence.numel() or rho.numel() != evidence.numel():
            raise ValueError("stimulus_digits and clean_probability must align with evidence.")
        if not bool(((rho >= 0.0) & (rho <= 1.0)).all()):
            raise ValueError("clean_probability must lie in [0,1].")
        vocab = torch.as_tensor(digit_vocab, device=evidence.device, dtype=torch.long)
        if vocab.dim() != 1 or vocab.numel() < 2 or vocab.unique().numel() != vocab.numel():
            raise ValueError("digit_vocab must contain unique candidate values.")
        n_candidates = int(vocab.numel())
        labels = (stimulus_digits[None] == vocab[:, None]).to(evidence.dtype)
        observed, _, _, _ = self._sequence_log_prob(
            evidence,
            quality,
            labels,
            reliability_override=rho,
        )
        scores = observed.sum(dim=1)
        if log_prior is None:
            return scores
        prior = log_prior.to(device=scores.device, dtype=scores.dtype).reshape(-1)
        if prior.numel() != n_candidates:
            raise ValueError("log_prior must have one value per candidate.")
        return scores + prior



def state_residual_gate_decision(
    subject_deltas: np.ndarray,
    *,
    n_bootstrap: int = 400,
    seed: int = 0,
) -> dict[str, object]:
    """Decide whether the state residual earned held-out log-score value.

    ``subject_deltas`` are per-audit-subject differences
    ``score(residual) - score(backbone)``. The gate is fail-closed:
    nonfinite scores, fewer than four subjects, a non-strict majority, or a
    cluster-bootstrap CI lower bound <= 0 all disable the residual.
    """
    deltas = np.asarray(subject_deltas, dtype=np.float64).reshape(-1)
    if len(deltas) < 4 or not bool(np.isfinite(deltas).all()):
        return {
            "passed": False,
            "strict_majority": False,
            "mean_delta": float("nan") if not len(deltas) else float(np.nanmean(deltas)),
            "ci_lower": float("nan"),
            "subject_deltas": deltas.tolist(),
            "n_subjects": int(len(deltas)),
        }
    mean_delta = float(deltas.mean())
    strict_majority = bool((deltas > 0.0).sum() > 0.5 * len(deltas))
    rng = np.random.default_rng(seed)
    bootstrap_means = np.asarray(
        [
            float(rng.choice(deltas, size=len(deltas), replace=True).mean())
            for _ in range(n_bootstrap)
        ]
    )
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    passed = bool(strict_majority and ci_lower > 0.0)
    return {
        "passed": passed,
        "strict_majority": strict_majority,
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "subject_deltas": deltas.tolist(),
        "n_subjects": int(len(deltas)),
    }
