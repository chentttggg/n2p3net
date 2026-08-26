from __future__ import annotations

import math

import pytest
import torch

from models.repetition_v12 import (
    AdditiveRepetitionEvidence,
    state_residual_gate_decision,
)
from train.contracts import SetMetadata
from train.repetition_v12_objective import additive_repetition_multi_k_objective


def test_state_residual_gate_fail_closed_counterexamples() -> None:
    import numpy as np

    positive = state_residual_gate_decision(np.asarray([1.0, 2.0, 3.0, 4.0]), n_bootstrap=200)
    assert positive["passed"] is True
    assert positive["strict_majority"] is True
    assert positive["ci_lower"] > 0.0

    mixed = state_residual_gate_decision(np.asarray([-1.0, -2.0, -3.0, 3.0]), n_bootstrap=200)
    assert mixed["passed"] is False

    single = state_residual_gate_decision(np.asarray([10.0]), n_bootstrap=200)
    assert single["passed"] is False

    nonfinite = state_residual_gate_decision(
        np.asarray([1.0, np.nan, 2.0]), n_bootstrap=200
    )
    assert nonfinite["passed"] is False


def test_fidelity_margin_rank_loss_is_the_v12_public_name() -> None:
    model = AdditiveRepetitionEvidence(hidden_size=8)
    quality = torch.randn(6, model.n_quality_features)

    loss = model.fidelity_margin_rank_loss(quality)

    assert torch.isfinite(loss)
    assert not hasattr(model, "fidelity_identification_loss")


def test_additive_backbone_candidate_score_is_cumulative_llr() -> None:
    model = AdditiveRepetitionEvidence(hidden_size=8)
    model.eval()
    digits = torch.tensor([1, 2, 3, 1, 2, 3])
    evidence = torch.tensor([-1.5, 2.0, -1.5, -1.5, 1.0, -1.5])
    quality = torch.zeros(6, 8)
    trajectory, reliability = model.candidate_log_score_trajectory(
        evidence, quality, digits, digit_vocab=(1, 2, 3)
    )
    labels = (digits[None] == torch.tensor([1, 2, 3])[:, None]).float()
    _, llr, _, _ = model._sequence_log_prob(evidence, quality, labels)
    expected = -math.log(3.0) + (llr * labels).cumsum(dim=1)
    assert torch.allclose(trajectory, expected, atol=1e-7)
    assert reliability.shape == (len(digits),)


def test_additive_candidate_chain_uses_all_flashes() -> None:
    model = AdditiveRepetitionEvidence(hidden_size=8)
    model.eval()
    digits = torch.tensor([1, 2, 3, 1, 2, 3])
    evidence = torch.where(digits == 2, 1.5, -1.5).float()
    scores, _ = model.candidate_log_scores(
        evidence, torch.zeros(6, 8), digits, digit_vocab=(1, 2, 3)
    )
    assert int(scores.argmax()) == 1
    assert scores[1] > scores[0] and scores[1] > scores[2]


def test_additive_backbone_is_exchangeable_without_residual() -> None:
    torch.manual_seed(2)
    model = AdditiveRepetitionEvidence(hidden_size=8, state_residual=False).eval()
    evidence = torch.tensor([-1.5, 2.0, -1.5, -1.5, 1.0, -1.5])
    digits = torch.tensor([1, 2, 3, 1, 2, 3])
    quality = torch.randn(6, 8)
    order = torch.tensor([5, 4, 3, 2, 1, 0])
    first, _ = model.candidate_log_scores(evidence, quality, digits, digit_vocab=(1, 2, 3))
    second, _ = model.candidate_log_scores(
        evidence[order], quality[order], digits[order], digit_vocab=(1, 2, 3)
    )
    assert torch.allclose(first, second, atol=1e-7)


def test_enabled_state_residual_breaks_exchangeability() -> None:
    torch.manual_seed(3)
    model = AdditiveRepetitionEvidence(hidden_size=8, state_residual=True).eval()
    torch.nn.init.normal_(model.state_residual.head[-1].weight, std=0.1)
    torch.nn.init.normal_(model.state_residual.gru.weight_ih_l0, std=0.1)
    with torch.no_grad():
        model.state_residual.gain.fill_(0.1)
    evidence = torch.tensor([-1.5, 2.0, -1.5, -1.5, 1.0, -1.5])
    digits = torch.tensor([1, 2, 3, 1, 2, 3])
    quality = torch.randn(6, 8)
    order = torch.tensor([5, 4, 3, 2, 1, 0])
    first, _ = model.candidate_log_scores(evidence, quality, digits, digit_vocab=(1, 2, 3))
    second, _ = model.candidate_log_scores(
        evidence[order], quality[order], digits[order], digit_vocab=(1, 2, 3)
    )
    assert not torch.allclose(first, second)


def test_state_residual_is_exactly_zero_by_default() -> None:
    torch.manual_seed(0)
    plain = AdditiveRepetitionEvidence(hidden_size=8, state_residual=False)
    residual = AdditiveRepetitionEvidence(hidden_size=8, state_residual=True)
    residual.load_state_dict(
        {key: value for key, value in plain.state_dict().items() if key in residual.state_dict()},
        strict=False,
    )
    evidence = torch.randn(3, 5)
    quality = torch.randn(3, 5, 8)
    labels = torch.randint(0, 2, (3, 5)).float()
    lengths = torch.tensor([5, 4, 3])
    plain_out = plain.forward_batched_sequences(evidence, quality, labels, lengths)
    residual_out = residual.forward_batched_sequences(evidence, quality, labels, lengths)
    assert torch.equal(residual.state_residual.gain, torch.zeros(()))
    assert torch.equal(plain_out.observed_log_prob, residual_out.observed_log_prob)
    assert torch.equal(plain_out.conditional_llr, residual_out.conditional_llr)
    assert residual_out.state_residual_energy == 0.0


def test_state_residual_gain_changes_outputs_when_enabled() -> None:
    torch.manual_seed(1)
    model = AdditiveRepetitionEvidence(hidden_size=8, state_residual=True)
    torch.nn.init.normal_(model.state_residual.head[-1].weight, std=0.1)
    torch.nn.init.normal_(model.state_residual.gru.weight_ih_l0, std=0.1)
    evidence = torch.randn(1, 4)
    quality = torch.randn(1, 4, 8)
    labels = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    lengths = torch.tensor([4])
    before = model.forward_batched_sequences(evidence, quality, labels, lengths)
    with torch.no_grad():
        model.state_residual.gain.fill_(0.1)
    after = model.forward_batched_sequences(evidence, quality, labels, lengths)
    assert not torch.allclose(before.conditional_llr, after.conditional_llr)


def test_additive_objective_gradients_and_coverage() -> None:
    model = AdditiveRepetitionEvidence(hidden_size=8, state_residual=True)
    model.set_evidence_calibration(pos_weight=8.0, train_prior=1.0 / 9.0)
    digits = torch.tensor([1, 2, 3, 1, 2, 3])
    labels = (digits == 2).float()
    logits = torch.where(labels > 0.5, 1.0, -1.0).requires_grad_()
    quality = torch.zeros(6, 8)
    metadata = SetMetadata(
        stimulus_digits=digits,
        group_ids=torch.zeros(6, dtype=torch.long),
        repetition_ranks=torch.tensor([0, 0, 0, 1, 1, 1]),
        sequence_ranks=torch.arange(6),
    )
    digit_loss, conditional_nll, coverage, residual_energy = (
        additive_repetition_multi_k_objective(
            logits,
            quality,
            labels,
            metadata,
            model,
            evidence_ks=(1, 2),
            evidence_weights=(0.5, 0.5),
            digit_vocab=(1, 2, 3),
            state_residual_l2_weight=1e-3,
        )
    )
    total = digit_loss + 0.1 * conditional_nll + residual_energy
    total.backward()
    assert coverage == {1: 1, 2: 1}
    assert torch.isfinite(total)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())



def test_additive_objective_survives_partial_k_prevalidated_batch() -> None:
    """Batch-wide ``prevalidated_kmax`` must be the minimum covered K.

    A loader may emit K1 prefixes for every group while K2 is requested.
    Claiming the loader-wide max K here would skip the per-K emptiness guard
    and feed an empty slice to cross-entropy (NaN).
    """
    from train.preloaded import GTNSetDataLoader

    torch.manual_seed(4)
    model = AdditiveRepetitionEvidence(hidden_size=8)
    model.set_evidence_calibration(pos_weight=1.0, train_prior=1.0 / 9.0)

    def partial_digits(group: int) -> torch.Tensor:
        digits = torch.arange(1, 10)
        return torch.cat((digits, digits[:1] + group))

    digits = torch.cat((partial_digits(0), partial_digits(1)))
    groups = torch.cat(
        (torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long))
    )
    labels = ((groups == 0) & (digits == 3)) | ((groups == 1) & (digits == 7))
    loader = GTNSetDataLoader(
        torch.randn(len(digits), 3, 16),
        labels.float(),
        digits,
        groups,
        evidence_ks=(1, 2),
        batch_size=20,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert batch.set_metadata.prevalidated_kmax == 1

    logits = torch.zeros(batch.y.numel()).requires_grad_()
    quality = torch.zeros(batch.y.numel(), model.n_quality_features)
    digit_loss, conditional_nll, coverage, _ = additive_repetition_multi_k_objective(
        logits,
        quality,
        batch.y,
        batch.set_metadata,
        model,
        evidence_ks=(1, 2),
        evidence_weights=(0.5, 0.5),
    )
    total = digit_loss + 0.1 * conditional_nll
    total.backward()

    assert coverage == {1: 2, 2: 0}
    assert torch.isfinite(digit_loss)
    assert torch.isfinite(total)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_candidate_chain_scores_with_external_clean_probability() -> None:
    """blueprint 4.2: calibrated rho enters the full digit-chain mixture path."""
    torch.manual_seed(5)
    model = AdditiveRepetitionEvidence(hidden_size=8).eval()
    digits = torch.tensor([1, 2, 3, 1])
    evidence = torch.tensor([-1.2, 1.4, -1.2, -1.2])
    quality = torch.randn(4, 8)
    rho = torch.tensor([0.9, 0.4, 0.8, 0.7])
    labels = (digits[None] == torch.tensor([1, 2, 3])[:, None]).float()
    state_before = {key: value.clone() for key, value in model.state_dict().items()}

    scores = model.candidate_chain_scores_with_clean_probability(
        evidence, quality, digits, rho, digit_vocab=(1, 2, 3)
    )
    with torch.no_grad():
        clean_log_prob, artifact_log_prob = model.backbone.class_log_probs(
            evidence, model.backbone.normalize_quality(quality)
        )
    expected = torch.zeros(3)
    for candidate in range(3):
        for step in range(4):
            label = int(labels[candidate, step])
            mixture = rho[step] * torch.exp(clean_log_prob[step, label]) + (
                1.0 - rho[step]
            ) * torch.exp(artifact_log_prob[step])
            expected[candidate] += torch.log(mixture)
    assert torch.allclose(scores, expected, atol=1e-6)
    state_after = model.state_dict()
    assert state_before.keys() == state_after.keys()
    assert all(torch.equal(state_before[key], state_after[key]) for key in state_before)


def test_additive_objective_rejects_missing_sequence_ranks() -> None:
    model = AdditiveRepetitionEvidence(hidden_size=8)
    digits = torch.tensor([1, 2, 3])
    metadata = SetMetadata(
        stimulus_digits=digits,
        group_ids=torch.zeros(3, dtype=torch.long),
        repetition_ranks=torch.zeros(3, dtype=torch.long),
        sequence_ranks=None,
    )
    with pytest.raises(ValueError, match="sequence_ranks"):
        additive_repetition_multi_k_objective(
            torch.zeros(3),
            torch.zeros(3, 8),
            (digits == 2).float(),
            metadata,
            model,
            evidence_ks=(1,),
            evidence_weights=(1.0,),
            digit_vocab=(1, 2, 3),
        )
