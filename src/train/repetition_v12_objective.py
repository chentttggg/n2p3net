"""v12 additive-LLR repetition objective.

Unlike the legacy loss, the candidate trajectory is the cumulative
``conditional_llr`` produced by a shared static backbone. The optional state
residual contributes a differentiable energy that is shrunk to zero by
default and must earn its weight through the R-object gate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from models.repetition_v12 import AdditiveRepetitionEvidence
from train.contracts import SetMetadata


def additive_repetition_multi_k_objective(
    weighted_logits: torch.Tensor,
    quality_features: torch.Tensor,
    y: torch.Tensor,
    metadata: SetMetadata,
    evidence_model: AdditiveRepetitionEvidence,
    *,
    evidence_ks: Sequence[int] = (1, 3, 5),
    evidence_weights: Sequence[float] = (0.34, 0.33, 0.33),
    digit_vocab: Sequence[int] = tuple(range(1, 10)),
    state_residual_l2_weight: float = 0.0,
    fidelity_aux_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int], torch.Tensor]:
    """Return ``(digit_loss, conditional_nll, coverage, residual_energy)``."""
    logits = weighted_logits.reshape(-1)
    labels = y.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if quality_features.shape != (logits.numel(), evidence_model.n_quality_features):
        raise ValueError("quality_features must align with weighted_logits.")
    if not metadata.prevalidated:
        metadata.validate(logits.numel())
    if metadata.sequence_ranks is None:
        raise ValueError("Additive repetition modeling requires acquisition-order sequence_ranks.")

    ks = tuple(int(k) for k in evidence_ks)
    if not ks or any(k < 1 for k in ks) or tuple(sorted(set(ks))) != ks:
        raise ValueError("evidence_ks must be unique positive integers in ascending order.")
    if len(evidence_weights) != len(ks) or any(float(w) < 0.0 for w in evidence_weights):
        raise ValueError("evidence_weights must be non-negative and match evidence_ks.")
    weights = torch.as_tensor(evidence_weights, device=logits.device, dtype=logits.dtype)
    if not bool(weights.sum() > 0.0):
        raise ValueError("At least one multi-K evidence weight must be positive.")
    weights = weights / weights.sum()
    kmax = metadata.prevalidated_kmax if metadata.prevalidated else None
    all_ks_covered = kmax is not None and max(ks) <= int(kmax)

    vocab = torch.as_tensor(digit_vocab, device=logits.device, dtype=torch.long)
    evidence = evidence_model.correct_evidence(logits)
    main_groups = torch.unique(metadata.group_ids[metadata.group_ids >= 0], sorted=True)
    if main_groups.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, zero, {k: 0 for k in ks}, zero

    group_rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for group in main_groups:
        rows = torch.nonzero(metadata.group_ids == group, as_tuple=False).flatten()
        order = torch.argsort(metadata.sequence_ranks[rows], stable=True)
        rows = rows[order]
        group_digits = metadata.stimulus_digits[rows]
        group_labels = labels[rows]
        positive_digits = torch.unique(group_digits[group_labels > 0.5])
        if positive_digits.numel() != 1 or not bool((positive_digits[0] == vocab).any()):
            continue
        group_rows.append(rows)
        targets.append(torch.nonzero(vocab == positive_digits[0], as_tuple=False).flatten()[0])

    if not group_rows:
        raise ValueError("Additive conditional NLL found no valid observed GTN sequence.")

    lengths = torch.as_tensor(
        [rows.numel() for rows in group_rows], device=logits.device, dtype=torch.long
    )
    padded_evidence = torch.nn.utils.rnn.pad_sequence(
        [evidence[rows] for rows in group_rows], batch_first=True
    )
    padded_quality = torch.nn.utils.rnn.pad_sequence(
        [quality_features[rows] for rows in group_rows], batch_first=True
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        [labels[rows] for rows in group_rows], batch_first=True
    )
    padded_digits = torch.nn.utils.rnn.pad_sequence(
        [metadata.stimulus_digits[rows] for rows in group_rows], batch_first=True
    )
    n_groups, max_length = padded_evidence.shape
    n_candidates = int(vocab.numel())
    candidate_labels = (padded_digits[:, None] == vocab[None, :, None]).to(logits.dtype)
    all_evidence = torch.cat(
        (padded_evidence, padded_evidence[:, None].expand(-1, n_candidates, -1).flatten(0, 1))
    )
    all_quality = torch.cat(
        (
            padded_quality,
            padded_quality[:, None].expand(-1, n_candidates, -1, -1).flatten(0, 1),
        )
    )
    all_labels = torch.cat((padded_labels, candidate_labels.flatten(0, 1)))
    all_lengths = torch.cat((lengths, lengths[:, None].expand(-1, n_candidates).reshape(-1)))

    sequence = evidence_model.forward_batched_sequences(
        all_evidence, all_quality, all_labels, all_lengths
    )

    active = torch.arange(max_length, device=logits.device)[None] < lengths[:, None]
    conditional_nll = -sequence.observed_log_prob[:n_groups][active].mean()
    if fidelity_aux_weight > 0.0:
        main_quality = quality_features[metadata.group_ids >= 0]
        conditional_nll = conditional_nll + fidelity_aux_weight * (
            evidence_model.fidelity_margin_rank_loss(main_quality)
        )
    if state_residual_l2_weight > 0.0:
        # blueprint 3.2: shrink residual deltas, not GRU weights.
        conditional_nll = conditional_nll + state_residual_l2_weight * (
            sequence.state_residual_energy
        )

    candidate_llr = sequence.conditional_llr[n_groups:].reshape(
        n_groups, n_candidates, max_length
    )
    candidate_contribution = candidate_llr * candidate_labels
    trajectory = candidate_contribution.cumsum(dim=-1) - math.log(n_candidates)
    occurrence_counts = candidate_labels.long().cumsum(dim=-1)
    target_tensor = torch.stack(targets).to(device=logits.device, dtype=torch.long)
    digit_loss = logits.sum() * 0.0
    active_weight = weights.sum() * 0.0
    coverage: dict[int, int] = {}
    for index, k in enumerate(ks):
        has_k = occurrence_counts[:, :, -1].ge(k).all(dim=1)
        if not all_ks_covered and not bool(has_k.any()):
            coverage[k] = 0
            continue
        first_k = occurrence_counts.ge(k).long().argmax(dim=-1)
        checkpoint = first_k.max(dim=1).values
        rows = torch.nonzero(has_k, as_tuple=False).flatten()
        scores = trajectory[rows, :, checkpoint[rows]]
        digit_loss = digit_loss + weights[index] * F.cross_entropy(scores, target_tensor[rows])
        active_weight = active_weight + weights[index]
        coverage[k] = int(rows.numel())
    if not all_ks_covered and not bool(active_weight > 0.0):
        raise ValueError("Additive repetition objective found no requested online checkpoint.")
    return digit_loss / active_weight, conditional_nll, coverage, sequence.state_residual_energy
