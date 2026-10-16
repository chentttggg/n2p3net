"""Frozen arithmetic probability pools over one physical candidate decision."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from models.candidate_evidence import CandidateEvidence
from research.contracts import semantic_sha256
from research.execution import atomic_write_json
from transfer.beta_uniform_calibration import beta_uniform_log_probability
from transfer.noise_model import freeze_metadata
from transfer.pool_calibration import POOL_CALIBRATION_SCHEMA, uniform_contamination_log_probability

ENSEMBLE_PROGRAM_SCHEMA = "n2p3_candidate_ensemble_program/1"


def beta_uniform_evidence(evidence: CandidateEvidence, calibration):
    """Derive one calibrated consumer without rebinding the raw model/capture."""
    if not evidence.probability_available:
        return evidence
    digest = semantic_sha256(calibration)
    identities = tuple(replace(item, probability_calibrated=True,
        fit_digest=semantic_sha256({"expert_fit": item.fit_digest, "beta_probability": digest}),
        score_kind=item.score_kind + ":forward_beta_uniform_predictive/1")
        for item in evidence.expert_identities)
    return replace(evidence, log_probability=beta_uniform_log_probability(evidence.log_probability, calibration),
                   expert_identities=identities)


@dataclass(frozen=True)
class EnsembleProgram:
    """The declared expert roster and fixed weights, including failure semantics."""

    name: str
    expert_ids: tuple[str, ...]
    weights: tuple[float, ...]

    def __post_init__(self):
        experts = tuple(self.expert_ids)
        weights = np.asarray(self.weights, dtype=np.float64)
        if (not self.name or not experts or any(not name for name in experts)
                or len(set(experts)) != len(experts)):
            raise ValueError("ensemble needs a name and unique declared experts")
        if (weights.shape != (len(experts),) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
            raise ValueError("declared ensemble weights must be positive and sum to one")
        object.__setattr__(self, "expert_ids", experts)
        object.__setattr__(self, "weights", tuple(float(value) for value in weights))

    @classmethod
    def equal_weights(cls, name: str, expert_ids: Sequence[str]):
        experts = tuple(expert_ids)
        if not experts:
            raise ValueError("an equal pool needs at least one declared expert")
        return cls(name, experts, (1. / len(experts),) * len(experts))

    def record(self):
        return {"schema": ENSEMBLE_PROGRAM_SCHEMA, "name": self.name,
                "expert_ids": list(self.expert_ids), "weights": list(self.weights),
                "pooling": "arithmetic_probability", "failure_policy": "all_declared_experts_required"}

    def digest(self):
        return semantic_sha256(self.record())

    @classmethod
    def from_record(cls, record):
        if (record["schema"] != ENSEMBLE_PROGRAM_SCHEMA
                or record["pooling"] != "arithmetic_probability"
                or record["failure_policy"] != "all_declared_experts_required"):
            raise ValueError("unsupported frozen candidate ensemble program")
        return cls(record["name"], tuple(record["expert_ids"]), tuple(record["weights"]))


class EnsembleScorer:
    """Pool declared individual experts; fitting and truth stay outside scoring.

    An aggregate is an output, not a silently relabelled input expert. Hierarchical
    weights are declared directly over the same flat producer roster.
    """

    def __init__(self, program: EnsembleProgram, pool_calibration=None):
        self._program = program
        self._pool_calibration = None if pool_calibration is None else freeze_metadata(pool_calibration)
        if pool_calibration is not None:
            if (pool_calibration["schema"] != POOL_CALIBRATION_SCHEMA
                    or pool_calibration["program_sha256"] != program.digest()):
                raise ValueError("pool calibration must belong to this exact expert roster and weights")
            uniform_contamination_log_probability(np.full(2, -np.log(2.)), pool_calibration["epsilon"])

    @property
    def program(self):
        return self._program

    @property
    def pool_calibration(self):
        return self._pool_calibration

    def score(self, expert_evidence: Mapping[str, CandidateEvidence]) -> CandidateEvidence:
        if set(expert_evidence) != set(self.program.expert_ids):
            raise ValueError("evidence must cover exactly the declared expert roster")
        evidence = tuple(expert_evidence[name] for name in self.program.expert_ids)
        for name, item in zip(self.program.expert_ids, evidence, strict=True):
            if len(item.expert_identities) != 1 or item.expert_identities[0].expert_id != name:
                raise ValueError("expert evidence producer differs from its declared roster key")
        first = evidence[0]
        if self.pool_calibration is not None and self.pool_calibration["candidate_count"] != len(first.candidate_vocabulary):
            raise ValueError("pool calibration candidate count differs from the actual requested vocabulary")
        common = (first.participant_key, first.decision_id, first.physical_support_digest,
                  first.candidate_vocabulary, first.evidence_level, first.expected_event_counts)
        for item in evidence[1:]:
            actual = (item.participant_key, item.decision_id, item.physical_support_digest,
                      item.candidate_vocabulary, item.evidence_level, item.expected_event_counts)
            if actual != common:
                raise ValueError("experts must score the same physical decision, support and vocabulary")
        identities = tuple(identity for item in evidence for identity in item.expert_identities)
        observed = tuple(min(item.observed_event_counts[index] for item in evidence)
                         for index in range(len(first.candidate_vocabulary)))
        unavailable = [(name, item) for name, item in zip(self.program.expert_ids, evidence, strict=True)
                       if not item.probability_available]
        identity = dict(candidate_vocabulary=first.candidate_vocabulary,
                        participant_key=first.participant_key, decision_id=first.decision_id,
                        physical_support_digest=first.physical_support_digest,
                        expert_identities=identities, evidence_level=first.evidence_level,
                        expected_event_counts=first.expected_event_counts,
                        observed_event_counts=observed)
        if unavailable:
            reason = "; ".join(f"{name}: {item.failure_reason}" for name, item in unavailable)
            kinds = {item.failure_kind for _, item in unavailable}
            kind = ("fit_failure" if "fit_failure" in kinds else
                    "incomplete" if "incomplete" in kinds else "not_available")
            return CandidateEvidence(log_probability=None, failure_kind=kind,
                                     failure_reason=reason, **identity)
        log_probability = logsumexp(np.log(self.program.weights)[:, None]
                                    + np.stack([item.log_probability for item in evidence]), axis=0)
        if self.pool_calibration is not None:
            log_probability = uniform_contamination_log_probability(log_probability, self.pool_calibration["epsilon"])
            calibration_digest = semantic_sha256(self.pool_calibration)
            identity["expert_identities"] = tuple(replace(item, probability_calibrated=True,
                fit_digest=semantic_sha256({"expert_fit": item.fit_digest, "pool_probability": calibration_digest}),
                score_kind=item.score_kind + ":pool_uniform_contamination/1") for item in identities)
        return CandidateEvidence(log_probability=log_probability, **identity)

    def save(self, path: str | Path):
        record = self.program.record()
        payload = {"program": record, "program_sha256": semantic_sha256(record)}
        if self.pool_calibration is not None:
            payload.update(pool_calibration=self.pool_calibration,
                           pool_calibration_sha256=semantic_sha256(self.pool_calibration))
        return atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_bytes())
        if semantic_sha256(payload["program"]) != payload["program_sha256"]:
            raise ValueError("saved ensemble program changed")
        pool = payload.get("pool_calibration")
        if pool is not None and semantic_sha256(pool) != payload["pool_calibration_sha256"]:
            raise ValueError("saved pool probability calibration changed")
        return cls(EnsembleProgram.from_record(payload["program"]), pool)
