"""Unified candidate-evidence aggregation core (audit F-03, 2026-09-02).

Every runner consumes this one implementation of per-decision candidate
scoring, winner selection, and tie abstention. The previous three-path fork
(``models.decision.decide`` subject-digit post-processing,
``transfer.evaluation._aggregate_scores`` generic candidate aggregation, and
the row/column accumulation in ``transfer.candidate_decision``) implemented
the same mathematics three times with drifting semantics; this module is the
single remaining contract.

Semantics (frozen with the archived v3 evidence chain):

- ``sum`` accumulates log-likelihood-ratio-style evidence; ``mean`` is the
  count-neutral default that keeps random candidate occurrence counts out of
  the ranking; ``tempered_evidence`` with ``count_power`` in ``[0,1]``
  interpolates the two (``0`` = mean, ``1`` = sum).
- ``trim0.2`` drops ``floor(0.2 n)`` extreme values per candidate before
  summing (no trim below five trials); ``precision`` is an inverse-variance
  weighted mean requiring finite positive per-trial variances.
- Empty candidates score ``-inf`` and never win.
- Exact ties (``rtol=atol=1e-12``) abstain: the winner is ``None``. Nothing
  may break ties by candidate label.
- Candidate codes outside the declared vocabulary fail closed instead of
  being silently dropped from the bucket aggregation (audit D8).
- With ``center_logits``, each decision group's logits are centered on their
  own (weighted) mean before accumulation, so a constant classifier bias
  cannot become ``c * n_d`` when per-candidate counts are unequal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EVIDENCE_AGGREGATIONS = frozenset({"sum", "mean", "tempered_evidence"})
EXTENDED_EVIDENCE_AGGREGATIONS = EVIDENCE_AGGREGATIONS | {"trim0.2", "precision"}
DEFAULT_EVIDENCE_AGGREGATION = "mean"
DEFAULT_EVIDENCE_COUNT_POWER = 0.5
MISSING_CANDIDATE_POLICIES = frozenset({"exclude", "abstain"})

# Retained alias: callers that only accept the three count-based modes.
COUNT_AGGREGATIONS = EVIDENCE_AGGREGATIONS


def count_tempered_evidence_scores(
    weighted_sums: np.ndarray,
    weight_sums: np.ndarray,
    squared_weight_sums: np.ndarray,
    *,
    count_power: float = DEFAULT_EVIDENCE_COUNT_POWER,
) -> np.ndarray:
    """Return weighted means tempered by effective evidence count.

    With unit weights this is ``mean(logit) * n**count_power``. Therefore
    ``count_power=0`` is mean aggregation and ``count_power=1`` is sum.
    """

    weighted_sums = np.asarray(weighted_sums, dtype=float)
    weight_sums = np.asarray(weight_sums, dtype=float)
    squared_weight_sums = np.asarray(squared_weight_sums, dtype=float)
    if not weighted_sums.shape == weight_sums.shape == squared_weight_sums.shape:
        raise ValueError("weighted sums and weight moments must have identical shapes.")
    if not np.isfinite(weighted_sums).all():
        raise ValueError("weighted evidence sums contain NaN/inf.")
    if not np.isfinite(count_power) or not 0.0 <= count_power <= 1.0:
        raise ValueError("count_power must be finite and in [0, 1].")
    if np.any(weight_sums < 0.0) or np.any(squared_weight_sums < 0.0):
        raise ValueError("evidence weight moments must be non-negative.")

    scores = np.full(weighted_sums.shape, -np.inf, dtype=float)
    nonempty = (weight_sums > 0.0) & (squared_weight_sums > 0.0)
    effective_count = np.zeros(weighted_sums.shape, dtype=float)
    effective_count[nonempty] = (
        weight_sums[nonempty] ** 2 / squared_weight_sums[nonempty]
    )
    weighted_mean = np.zeros(weighted_sums.shape, dtype=float)
    weighted_mean[nonempty] = weighted_sums[nonempty] / weight_sums[nonempty]
    scores[nonempty] = weighted_mean[nonempty] * effective_count[nonempty] ** count_power
    return scores


@dataclass(frozen=True)
class CandidateScores:
    """One decision's per-candidate scores and abstention-aware winner."""

    scores: dict[int, float]
    counts: dict[int, int]
    predicted: int | None


def _validated_vocabulary(
    vocabulary: Sequence[int],
    codes: np.ndarray,
) -> np.ndarray:
    vocab = np.asarray(vocabulary)
    if vocab.ndim != 1 or len(vocab) == 0 or len(np.unique(vocab)) != len(vocab):
        raise ValueError(
            "candidate vocabulary must be a unique non-empty one-dimensional array."
        )
    if len(codes) and not np.isin(codes, vocab).all():
        outside = sorted(set(codes.tolist()) - set(vocab.tolist()))
        raise ValueError(
            f"candidate codes outside the declared vocabulary fail closed: {outside}."
        )
    return vocab


def candidate_evidence_scores(
    logits: Sequence[float],
    codes: Sequence[int],
    vocabulary: Sequence[int],
    *,
    aggregation: str = DEFAULT_EVIDENCE_AGGREGATION,
    evidence_count_power: float = DEFAULT_EVIDENCE_COUNT_POWER,
    logit_variances: Sequence[float] | None = None,
    trial_weights: Sequence[float] | None = None,
    missing_candidate_policy: str = "exclude",
) -> CandidateScores:
    """Aggregate one decision's trial evidence into per-candidate scores.

    Parameters
    ----------
    logits, codes : trial-level evidence and candidate membership.
    vocabulary : the frozen candidate set; codes outside it fail closed.
    aggregation : one of ``EXTENDED_EVIDENCE_AGGREGATIONS``.
    evidence_count_power : ``beta`` for ``tempered_evidence`` in ``[0,1]``.
    logit_variances : per-trial predictive variances (``precision`` only).
    trial_weights : optional non-negative label-free reliability weights.
    missing_candidate_policy : ``"exclude"`` lets candidates with no surviving
        trial simply never win (the ``decide``/endpoint semantics);
        ``"abstain"`` makes any missing candidate void the whole decision
        (the hit@R requested-denominator semantics: a missing candidate is
        incomplete evidence, not evidence against that candidate).

    Returns
    -------
    CandidateScores
        Per-candidate scores (``-inf`` for empty candidates), observed trial
        counts, and the argmax winner or ``None`` on an exact tie or abstain.
    """

    if aggregation not in EXTENDED_EVIDENCE_AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {sorted(EXTENDED_EVIDENCE_AGGREGATIONS)}.")
    if missing_candidate_policy not in MISSING_CANDIDATE_POLICIES:
        raise ValueError(
            f"missing_candidate_policy must be one of {sorted(MISSING_CANDIDATE_POLICIES)}."
        )
    values = np.asarray(logits, dtype=float)
    membership = np.asarray(codes)
    if values.ndim != 1 or membership.ndim != 1 or len(values) != len(membership):
        raise ValueError("logits and candidate codes must be aligned one-dimensional vectors.")
    if len(values) and not np.isfinite(values).all():
        raise ValueError("logits contain NaN/inf.")
    vocab = _validated_vocabulary(vocabulary, membership.astype(membership.dtype, copy=False))
    variances = None if logit_variances is None else np.asarray(logit_variances, dtype=float)
    if variances is not None and variances.shape != values.shape:
        raise ValueError("logit_variances must align with logits.")
    if aggregation == "precision" and variances is None:
        raise ValueError("precision aggregation requires per-trial predictive variances.")
    weights = (
        np.ones(len(values), dtype=float)
        if trial_weights is None
        else np.asarray(trial_weights, dtype=float)
    )
    if weights.shape != values.shape:
        raise ValueError("trial_weights must align with logits.")
    if len(weights) and not np.isfinite(weights).all():
        raise ValueError("trial_weights must be finite.")
    if len(weights) and (np.any(weights < 0.0) or not np.any(weights > 0.0)):
        raise ValueError("trial_weights must be non-negative with at least one positive entry.")

    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for candidate in vocab.tolist():
        selected = membership == candidate
        count = int(selected.sum())
        counts[int(candidate)] = count
        if not count:
            scores[int(candidate)] = -np.inf
            continue
        selected_values = values[selected]
        selected_weights = weights[selected]
        positive = selected_weights > 0.0
        if not positive.any():
            scores[int(candidate)] = -np.inf
            continue
        if aggregation == "sum":
            scores[int(candidate)] = float(np.sum(selected_weights * selected_values))
        elif aggregation == "mean":
            scores[int(candidate)] = float(
                np.sum(selected_weights * selected_values) / np.sum(selected_weights)
            )
        elif aggregation == "tempered_evidence":
            scores[int(candidate)] = float(
                count_tempered_evidence_scores(
                    np.asarray([np.sum(selected_weights * selected_values)]),
                    np.asarray([np.sum(selected_weights)]),
                    np.asarray([np.sum(selected_weights**2)]),
                    count_power=evidence_count_power,
                )[0]
            )
        elif aggregation == "trim0.2":
            ordered = np.sort(selected_values)
            trim = int(np.floor(0.2 * len(ordered)))
            kept = ordered[trim : len(ordered) - trim] if trim else ordered
            scores[int(candidate)] = float(np.sum(kept))
        else:
            assert aggregation == "precision" and variances is not None
            candidate_variances = variances[selected]
            if not np.isfinite(candidate_variances).all() or np.any(
                candidate_variances <= 0.0
            ):
                raise ValueError("predictive variances must be finite and positive.")
            inverse = 1.0 / candidate_variances
            scores[int(candidate)] = float(
                np.dot(inverse, selected_values) / inverse.sum()
            )

    finite_scores = np.asarray([scores[int(candidate)] for candidate in vocab.tolist()], dtype=float)
    if missing_candidate_policy == "abstain" and bool(np.isneginf(finite_scores).any()):
        predicted = None
    else:
        maximum = float(np.max(finite_scores))
        tied = np.flatnonzero(np.isclose(finite_scores, maximum, rtol=1e-12, atol=1e-12))
        predicted = int(vocab[tied[0]]) if len(tied) == 1 else None
    return CandidateScores(scores=scores, counts=counts, predicted=predicted)


@dataclass(frozen=True)
class GroupDecisions:
    """Per-decision-group winners over a frozen candidate vocabulary."""

    group_ids: np.ndarray
    predicted: np.ndarray  # object dtype; None on tie or all-empty groups
    scores_by_group: dict[str, dict[int, float]]


def decide_by_group(
    logits: Sequence[float],
    codes: Sequence[int],
    group_ids: Sequence,
    vocabulary: Sequence[int],
    *,
    center_logits: bool = True,
    aggregation: str = DEFAULT_EVIDENCE_AGGREGATION,
    evidence_count_power: float = DEFAULT_EVIDENCE_COUNT_POWER,
    trial_weights: Sequence[float] | None = None,
) -> GroupDecisions:
    """Decide one candidate per decision group from trial-level evidence.

    Groups are the decision units (subjects, selections, sessions). With
    ``center_logits`` each group's logits are centered on their own weighted
    mean before accumulation so constant classifier bias cannot leak into the
    ranking as ``c * n_d`` under unequal per-candidate counts.
    """

    values = np.asarray(logits, dtype=float)
    membership = np.asarray(codes)
    groups = np.asarray(group_ids).astype(str)
    if not (len(values) == len(membership) == len(groups)):
        raise ValueError("logits, codes, and group_ids must be aligned.")
    if len(values) and not np.isfinite(values).all():
        raise ValueError("logits contain NaN/inf.")
    if aggregation not in EVIDENCE_AGGREGATIONS:
        raise ValueError(
            f"group decisions accept {sorted(EVIDENCE_AGGREGATIONS)}, got {aggregation!r}."
        )
    vocab = _validated_vocabulary(vocabulary, membership.astype(membership.dtype, copy=False))
    weights = (
        np.ones(len(values), dtype=float)
        if trial_weights is None
        else np.asarray(trial_weights, dtype=float)
    )
    if weights.shape != values.shape:
        raise ValueError("trial_weights must align with logits.")
    if len(weights) and not np.isfinite(weights).all():
        raise ValueError("trial_weights must be finite.")
    if len(weights) and (np.any(weights < 0.0) or not np.any(weights > 0.0)):
        raise ValueError("trial_weights must be non-negative with at least one positive entry.")

    unique_groups = np.unique(groups)
    aligned = values
    if center_logits and len(values):
        aligned = values.astype(float, copy=True)
        for group in unique_groups:
            mask = groups == group
            weight_sum = float(weights[mask].sum())
            if weight_sum > 0.0 and int(mask.sum()) > 1:
                aligned[mask] -= float(np.sum(weights[mask] * values[mask])) / weight_sum

    predicted = np.full(len(unique_groups), None, dtype=object)
    scores_by_group: dict[str, dict[int, float]] = {}
    for index, group in enumerate(unique_groups):
        mask = groups == group
        result = candidate_evidence_scores(
            aligned[mask],
            membership[mask],
            vocab,
            aggregation=aggregation,
            evidence_count_power=evidence_count_power,
            trial_weights=weights[mask],
        )
        predicted[index] = result.predicted
        scores_by_group[str(group)] = result.scores
    return GroupDecisions(
        group_ids=unique_groups,
        predicted=predicted,
        scores_by_group=scores_by_group,
    )
