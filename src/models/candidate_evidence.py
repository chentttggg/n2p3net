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
- Empty or zero-effective-weight candidates make the decision abstain; the
  frozen vocabulary never shrinks to the surviving evidence.
- Top score contrasts within the fixed absolute ``1e-12`` floor abstain:
  the winner is ``None``. The common score origin and candidate labels do not
  set the tie tolerance.
- Candidate codes outside the declared vocabulary fail closed instead of
  being silently dropped from the bucket aggregation (audit D8).
- With ``center_logits``, each decision group's logits are centered on their
  own (weighted) mean before accumulation, so a constant classifier bias
  cannot become ``c * n_d`` when per-candidate counts are unequal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real

import numpy as np
import torch

from data.candidate_task import MembershipDecision
from models.candidate_ties import candidate_top_tie_mask

EVIDENCE_AGGREGATIONS = frozenset({"sum", "mean", "tempered_evidence"})
EXTENDED_EVIDENCE_AGGREGATIONS = EVIDENCE_AGGREGATIONS | {"trim0.2", "precision"}
DEFAULT_EVIDENCE_AGGREGATION = "mean"
DEFAULT_EVIDENCE_COUNT_POWER = 0.5

MEMBERSHIP_SCORE_METHOD = "membership_llr_sum/1"


def _exact_identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} requires an exact nonempty string identity")
    return value


def _evidence_digest(value: str | None, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase sha256 digest")


def _log_normalize(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise"):
        try:
            shifted = values - np.max(values)
            return shifted - np.log(np.exp(shifted).sum())
        except FloatingPointError as error:
            raise ValueError("log evidence differences exceed floating-point support") from error


def _real_evidence_values(value):
    values = np.asarray(value)
    if np.iscomplexobj(values):
        raise ValueError("candidate log evidence and probabilities require real values")
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class CandidateExpertIdentity:
    """An expert's own input and fitted state, independent of common raw support.

    A target-only expert has no source-model digest. A failed fit may have no
    model digest; its declared fit procedure and numeric input remain identified.
    Normalized scores do not by themselves certify probability calibration.
    """

    expert_id: str
    numerical_input_digest: str
    model_digest: str | None
    source_digest: str | None
    fit_digest: str
    score_kind: str
    probability_calibrated: bool

    def __post_init__(self) -> None:
        for name in ("expert_id", "score_kind"):
            _exact_identity(getattr(self, name), name)
        for name in ("numerical_input_digest", "fit_digest"):
            _evidence_digest(getattr(self, name), name)
        for name in ("model_digest", "source_digest"):
            _evidence_digest(getattr(self, name), name, optional=True)
        if not isinstance(self.probability_calibrated, bool):
            raise ValueError("probability_calibrated must be explicitly boolean")

    def record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvidence:
    """Validated, label-free candidate probabilities for one physical decision.

    Construction and artifact reload are the sole validation boundary. Arrays
    own immutable byte storage, so consumers do not revalidate or rehash them.
    ``-inf`` denotes true zero probability; finite log probabilities remain
    available when their exponentials underflow. Complete tied distributions
    are available for risk evaluation but provide no operational prediction.
    """

    candidate_vocabulary: tuple[str, ...]
    log_probability: np.ndarray | None
    participant_key: str
    decision_id: str
    physical_support_digest: str
    expert_identities: tuple[CandidateExpertIdentity, ...]
    evidence_level: int
    expected_event_counts: tuple[int, ...]
    observed_event_counts: tuple[int, ...]
    failure_reason: str | None = None
    failure_kind: str | None = None
    predicted: str | None = field(init=False)
    status: str = field(init=False)
    top_candidate_indices: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.candidate_vocabulary, (str, bytes)):
            raise ValueError("candidate vocabulary must be a sequence of exact identities")
        vocabulary = tuple(self.candidate_vocabulary)
        if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("candidate vocabulary must contain at least two unique identities")
        for candidate in vocabulary:
            _exact_identity(candidate, "candidate vocabulary")
        object.__setattr__(self, "candidate_vocabulary", vocabulary)
        for name in ("participant_key", "decision_id"):
            _exact_identity(getattr(self, name), name)
        _evidence_digest(self.physical_support_digest, "physical_support_digest")
        identities = tuple(self.expert_identities)
        if not identities or any(not isinstance(value, CandidateExpertIdentity) for value in identities):
            raise ValueError("expert_identities requires validated expert identities")
        if len({value.expert_id for value in identities}) != len(identities):
            raise ValueError("expert identities must be unique")
        object.__setattr__(self, "expert_identities", identities)
        if isinstance(self.evidence_level, bool) or not isinstance(self.evidence_level, Integral) or self.evidence_level < 1:
            raise ValueError("evidence_level must be a positive integer")
        object.__setattr__(self, "evidence_level", int(self.evidence_level))
        for name in ("expected_event_counts", "observed_event_counts"):
            counts = tuple(getattr(self, name))
            if len(counts) != len(vocabulary) or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in counts
            ):
                raise ValueError(f"{name} must be nonnegative integers aligned with the vocabulary")
            object.__setattr__(self, name, tuple(int(value) for value in counts))
        if any(observed > expected for observed, expected in
               zip(self.observed_event_counts, self.expected_event_counts, strict=True)):
            raise ValueError("observed event counts cannot exceed the declared physical schedule")
        predicted = None
        top_candidate_indices = ()
        if self.log_probability is None:
            if self.failure_kind not in {"fit_failure", "not_available", "incomplete", "abstain"}:
                raise ValueError("missing probability requires an explicit failure kind")
            _exact_identity(self.failure_reason, "failure_reason")
            status = self.failure_kind
        else:
            if self.failure_reason is not None or self.failure_kind is not None:
                raise ValueError("failure evidence cannot publish candidate probabilities")
            if not self.physical_coverage_complete:
                raise ValueError("candidate probability requires complete physical coverage")
            if any(value.model_digest is None for value in identities):
                raise ValueError("candidate probabilities require each fitted model identity")
            values = _real_evidence_values(self.log_probability)
            if values.shape != (len(vocabulary),):
                raise ValueError("log probability shape must align with the frozen vocabulary")
            if np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any():
                raise ValueError("log probabilities must be finite or true negative infinity")
            maximum = np.max(values)
            normalization = maximum + np.log(np.exp(values - maximum).sum())
            if not np.isclose(normalization, 0., rtol=0., atol=1e-10):
                raise ValueError("log probabilities must be normalized on the complete vocabulary")
            values = np.frombuffer(values.tobytes(), dtype=np.float64)
            object.__setattr__(self, "log_probability", values)
            winners = np.flatnonzero(candidate_top_tie_mask(values))
            top_candidate_indices = tuple(int(index) for index in winners)
            status = "tie" if len(winners) != 1 else "available"
            if status == "available":
                predicted = vocabulary[int(winners[0])]
        object.__setattr__(self, "predicted", predicted)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "top_candidate_indices", top_candidate_indices)

    @property
    def probability(self) -> np.ndarray | None:
        return None if self.log_probability is None else np.exp(self.log_probability)

    @property
    def probability_available(self) -> bool:
        return self.log_probability is not None

    @property
    def operational_available(self) -> bool:
        return self.predicted is not None

    @property
    def tie(self) -> bool:
        return self.status == "tie"

    @property
    def physical_coverage_complete(self) -> bool:
        return (all(value > 0 for value in self.expected_event_counts)
                and self.expected_event_counts == self.observed_event_counts)

    @classmethod
    def from_log_evidence(cls, log_evidence, *, temperature=1., prior=None, **identity):
        """Normalize a continuous joint likelihood once, without trial reaggregation."""
        count = len(identity["candidate_vocabulary"])
        values = _real_evidence_values(log_evidence)
        if values.shape != (count,) or np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any():
            raise ValueError("log evidence must cover the frozen candidate vocabulary")
        if isinstance(temperature, bool) or not isinstance(temperature, Real) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a positive finite frozen scalar")
        if prior is None:
            log_prior = np.zeros(count)
        else:
            prior = _real_evidence_values(prior)
            if prior.shape != (count,) or not np.isfinite(prior).all() or np.any(prior < 0) or not np.isclose(prior.sum(), 1., rtol=0., atol=1e-10):
                raise ValueError("candidate prior must sum to one on the frozen vocabulary")
            log_prior = np.full(count, -np.inf)
            np.log(prior, out=log_prior, where=prior > 0)
        expected = tuple(identity["expected_event_counts"])
        observed = tuple(identity["observed_event_counts"])
        if expected != observed or not all(value > 0 for value in expected):
            reason = "no_usable_evidence" if not any(observed) else "incomplete_candidate_coverage"
            return cls(log_probability=None, failure_kind="incomplete", failure_reason=reason, **identity)
        with np.errstate(over="raise", invalid="raise"):
            try:
                scaled = (values - np.max(values)) / temperature + log_prior
            except FloatingPointError as error:
                raise ValueError("log evidence differences exceed floating-point support") from error
        if not np.isfinite(scaled).any():
            raise ValueError("candidate likelihood and prior have disjoint support")
        return cls(log_probability=_log_normalize(scaled), **identity)

    @classmethod
    def from_membership_scores(cls, scores: MembershipScores, decision: MembershipDecision, **identity):
        """Preserve the existing membership scorer's full vocabulary and failure semantics."""
        if scores.candidate_vocabulary != decision.candidate_vocabulary:
            raise ValueError("membership scores and physical decision vocabulary differ")
        identity = dict(identity, candidate_vocabulary=decision.candidate_vocabulary,
                        evidence_level=decision.repetitions,
                        expected_event_counts=tuple(decision.membership.sum(axis=0, dtype=np.int64)),
                        observed_event_counts=tuple(scores.counts))
        if not scores.eligible:
            return cls(log_probability=None, failure_kind="incomplete",
                       failure_reason=scores.reason or "unidentifiable_candidates", **identity)
        values = scores.scores.detach().cpu().numpy() if isinstance(scores.scores, torch.Tensor) else np.asarray(scores.scores)
        return cls(log_probability=_log_normalize(np.asarray(values, dtype=np.float64)), **identity)

    def record(self) -> dict[str, object]:
        """JSON-safe persistence; null inside a log vector encodes true zero probability."""
        return {"schema": "n2p3_candidate_probability_evidence/1",
                "candidate_vocabulary": list(self.candidate_vocabulary),
                "log_probability": (None if self.log_probability is None else
                                    [None if np.isneginf(value) else float(value) for value in self.log_probability]),
                "participant_key": self.participant_key, "decision_id": self.decision_id,
                "physical_support_digest": self.physical_support_digest,
                "expert_identities": [identity.record() for identity in self.expert_identities],
                "evidence_level": self.evidence_level,
                "expected_event_counts": list(self.expected_event_counts),
                "observed_event_counts": list(self.observed_event_counts),
                "failure_reason": self.failure_reason, "failure_kind": self.failure_kind}

    @classmethod
    def from_record(cls, record: Mapping[str, object]):
        fields = {name for name, value in cls.__dataclass_fields__.items() if value.init}
        if set(record) != fields | {"schema"} or record["schema"] != "n2p3_candidate_probability_evidence/1":
            raise ValueError("candidate probability evidence record schema or fields differ")
        values = dict(record)
        values.pop("schema")
        values["expert_identities"] = tuple(CandidateExpertIdentity(**identity) for identity in values["expert_identities"])
        if values["log_probability"] is not None:
            values["log_probability"] = np.array([(-np.inf if value is None else value) for value in values["log_probability"]])
        return cls(**values)


def aggregate_membership_sums(sums, counts, *, aggregation="sum"):
    """Shared sum/legacy-mean arithmetic, including packed prefix reductions."""
    if aggregation == "sum":
        return sums
    if aggregation != "mean":
        raise ValueError("membership aggregation must be sum or explicit legacy mean")
    if isinstance(sums, torch.Tensor):
        denominator = torch.where(counts > 0, counts, torch.ones_like(counts))
        return torch.where(counts > 0, sums / denominator, sums.new_full((), -torch.inf))
    sums, counts = np.asarray(sums), np.asarray(counts)
    return np.where(counts > 0, sums / np.where(counts > 0, counts, 1), -np.inf)


def membership_evidence(evidence, membership, *, available=None, logit_offset=0., aggregation="sum"):
    """Compute A.T @ where(available, evidence-offset, 0), preserving autograd.

    This numeric kernel does not invent schedules. New decisions must also pass
    MembershipDecision's physical identity and coverage contract. ``mean`` is
    reserved for explicitly identified historical scorers.
    """
    if isinstance(evidence, torch.Tensor):
        if evidence.ndim != 1 or not evidence.is_floating_point():
            raise ValueError("membership evidence must be a floating vector")
        dtype = torch.float64 if evidence.dtype == torch.float64 else torch.float32
        values = evidence.to(dtype=dtype)
        if isinstance(membership, torch.Tensor) and membership.requires_grad:
            raise ValueError("physical membership cannot be learned")
        matrix = torch.as_tensor(np.array(membership, copy=True) if isinstance(membership, np.ndarray)
                                 else membership, device=values.device)
        observed = (torch.ones(len(values), dtype=torch.bool, device=values.device) if available is None
                    else torch.as_tensor(np.array(available, copy=True) if isinstance(available, np.ndarray)
                                         else available, device=values.device))
        if (matrix.ndim != 2 or matrix.shape[0] != len(values) or matrix.shape[1] < 2
                or not bool(((matrix == 0) | (matrix == 1)).all())):
            raise ValueError("membership must be an aligned binary matrix with a frozen vocabulary")
        if observed.dtype != torch.bool or observed.shape != values.shape:
            raise ValueError("availability must be an aligned boolean vector")
        offset = torch.as_tensor(logit_offset, device=values.device, dtype=dtype)
        if offset.requires_grad or offset.shape not in {torch.Size([]), values.shape} or not bool(torch.isfinite(offset).all()):
            raise ValueError("logit offset must be finite, frozen and scalar or aligned")
        if not bool(torch.isfinite(values[observed]).all()):
            raise ValueError("available membership evidence must be finite")
        with torch.autocast(device_type=values.device.type, enabled=False):
            matrix = matrix.to(dtype=dtype)
            selected = torch.where(observed, values - offset, torch.zeros_like(values))
            sums = matrix.T @ selected
            counts = matrix.T @ observed.to(dtype=dtype)
            return aggregate_membership_sums(sums, counts, aggregation=aggregation)
    values = np.asarray(evidence, dtype=float)
    matrix = np.asarray(membership)
    if values.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != len(values) or matrix.shape[1] < 2 or not (
            (matrix == 0) | (matrix == 1)).all():
        raise ValueError("membership must align a floating vector with a binary candidate matrix")
    observed = np.ones(len(values), dtype=bool) if available is None else np.asarray(available)
    if observed.dtype != bool or observed.shape != values.shape:
        raise ValueError("availability must be an aligned boolean vector")
    offset = np.asarray(logit_offset, dtype=float)
    if offset.shape not in {(), values.shape} or not np.isfinite(offset).all():
        raise ValueError("logit offset must be finite and scalar or aligned")
    if not np.isfinite(values[observed]).all():
        raise ValueError("available membership evidence must be finite")
    selected = np.where(observed, values - offset, 0.)
    return aggregate_membership_sums(matrix.T @ selected, matrix.T @ observed.astype(float),
                                     aggregation=aggregation)


@dataclass(frozen=True)
class MembershipScores:
    candidate_vocabulary: tuple[str, ...]
    scores: np.ndarray | torch.Tensor
    posterior: np.ndarray | torch.Tensor
    counts: np.ndarray
    stimulus_counts: np.ndarray
    complete: bool
    identifiable: bool
    predicted: str | None
    reason: str | None
    score_kind: str = MEMBERSHIP_SCORE_METHOD

    @property
    def eligible(self):
        return self.complete and self.identifiable

    def record(self):
        def values(array):
            return array.detach().cpu().tolist() if isinstance(array, torch.Tensor) else array.tolist()

        return {"schema": "n2p3_membership_scores/1", "score_kind": self.score_kind,
                "candidate_vocabulary": list(self.candidate_vocabulary),
                "scores": values(self.scores), "posterior": values(self.posterior),
                "probability_calibrated": False, "counts": self.counts.tolist(),
                "stimulus_counts": self.stimulus_counts.tolist(), "complete": self.complete,
                "identifiable": self.identifiable, "eligible": self.eligible,
                "prediction": self.predicted, "reason": self.reason}


def membership_decision_scores(evidence, decision: MembershipDecision, *, available=None,
                               temperature=1., prior=None, logit_offset=0.) -> MembershipScores:
    """Versioned sum scorer for direct, row/column and declared set membership.

    The normalized scores have no automatic calibration claim. Even a finite
    prior-only posterior cannot produce an operational EEG prediction.
    """
    if not isinstance(decision, MembershipDecision):
        raise ValueError("a physical MembershipDecision is required")
    if isinstance(temperature, bool) or not isinstance(temperature, Real) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite frozen scalar")
    observed = decision.evidence_indices >= 0
    if available is not None:
        supplied = available.detach().cpu().numpy() if isinstance(available, torch.Tensor) else np.asarray(available)
        if supplied.dtype != bool or supplied.shape != observed.shape:
            raise ValueError("availability must align with the physical event ledger")
        if np.any(supplied & ~observed):
            raise ValueError("availability cannot restore a physically unavailable event")
        observed = observed & supplied
    count = len(decision.candidate_vocabulary)
    if isinstance(prior, torch.Tensor):
        if prior.requires_grad:
            raise ValueError("candidate prior must be frozen, not a learned code bias")
        prior = prior.detach().cpu().numpy()
    probability = np.full(count, 1. / count) if prior is None else np.asarray(prior, dtype=float)
    if (probability.shape != (count,) or not np.isfinite(probability).all()
            or np.any(probability <= 0) or not np.isclose(probability.sum(), 1., rtol=0, atol=1e-8)):
        raise ValueError("candidate prior must be strictly positive and sum to one on the frozen vocabulary")
    total = membership_evidence(evidence, decision.membership, available=observed, logit_offset=logit_offset)
    if isinstance(total, torch.Tensor):
        scores = total / temperature + total.new_tensor(np.log(probability))
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("membership scores must remain finite")
        posterior = torch.softmax(scores, dim=0)
        detached = scores.detach().cpu().numpy()
    else:
        scores = total / temperature + np.log(probability)
        if not np.isfinite(scores).all():
            raise ValueError("membership scores must remain finite")
        unnormalized = np.exp(scores - scores.max())
        posterior = unnormalized / unnormalized.sum()
        detached = scores
    counts = decision.membership[observed].sum(axis=0, dtype=np.int64)
    stimulus_counts = np.bincount(decision.stimulus_indices[observed], minlength=len(decision.stimulus_vocabulary))
    complete = decision.schedule_complete and bool(observed.all())
    identifiable = len(np.unique(decision.membership[observed].T, axis=0)) == count
    reason = ("no_usable_evidence" if not observed.any() else
              "incomplete_candidate_coverage" if not complete else
              "unidentifiable_candidates" if not identifiable else None)
    predicted = None
    if reason is None:
        winners = np.flatnonzero(candidate_top_tie_mask(detached))
        if len(winners) == 1:
            predicted = decision.candidate_vocabulary[int(winners[0])]
        else:
            reason = "tie"
    return MembershipScores(decision.candidate_vocabulary, scores, posterior, counts,
                            stimulus_counts, complete, identifiable, predicted, reason)


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
        Precision aggregation uses reliability divided by predictive variance.
        The count-based ``trim0.2`` estimator does not support external weights
        and rejects that combination. Missing or zero-weight candidates always
        make the whole decision abstain.

    Returns
    -------
    CandidateScores
        Per-candidate scores (``-inf`` for empty candidates), observed trial
        counts, and the argmax winner or ``None`` within the fixed contrast
        tie floor or on incomplete evidence.
    """

    if aggregation not in EXTENDED_EVIDENCE_AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {sorted(EXTENDED_EVIDENCE_AGGREGATIONS)}.")
    if aggregation == "trim0.2" and trial_weights is not None:
        raise ValueError("trim0.2 does not support trial_weights; declare a weighted aggregation")
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
            candidate_variances = variances[selected][positive]
            if not np.isfinite(candidate_variances).all() or np.any(
                candidate_variances <= 0.0
            ):
                raise ValueError("predictive variances must be finite and positive.")
            inverse = selected_weights[positive] / candidate_variances
            scores[int(candidate)] = float(
                np.dot(inverse, selected_values[positive]) / inverse.sum()
            )

    finite_scores = np.asarray([scores[int(candidate)] for candidate in vocab.tolist()], dtype=float)
    if bool(np.isneginf(finite_scores).any()):
        predicted = None
    else:
        tied = np.flatnonzero(candidate_top_tie_mask(finite_scores))
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
        if not float(weights[mask].sum()) > 0.0:
            scores_by_group[str(group)] = {}
            continue
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
