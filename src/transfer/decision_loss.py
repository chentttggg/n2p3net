"""Typed direct-membership decision loss and complete-decision batching.

The training contract in this module is deliberately narrower than the
post-hoc candidate evaluators.  It accepts only row-aligned metadata derived
from a :class:`data.events.ScheduledEventTimeline` whose binary labels obey
``y_i = 1[c_i == d*]``.  Row/column intersection tasks and binary-evidence
projections must use a different, protocol-aware objective.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.events import ScheduledEventTimeline

DIRECT_DECISION_METADATA_SCHEMA = "n2p3_direct_decision_metadata/1"
DECISION_LOSS_CONFIG_SCHEMA = "n2p3_decision_loss_config/1"
DECISION_LOSS_RESULT_SCHEMA = "n2p3_decision_loss_result/1"
DECISION_BATCH_PLAN_SCHEMA = "n2p3_complete_decision_batch_plan/1"
DECISION_TRAINING_LEDGER_SCHEMA = "n2p3_decision_training_ledger/1"

DIRECT_MEMBERSHIP_PROTOCOL = "scheduled_event_direct_membership"
PREFIX_WEIGHT_MODES = frozenset({"final", "uniform"})
DECISION_AGGREGATION = "count_neutral_mean"
DECISION_TEMPERATURE_SCHEDULE = "constant"
DECISION_BATCH_POLICY = "complete_decisions"
DIRECT_SOURCE_EVENT_CONTRACTS = frozenset(
    {"preserve", DIRECT_MEMBERSHIP_PROTOCOL}
)


def _strict_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\0" in value:
        raise ValueError(f"{name} must be an exact non-empty string without outer whitespace or NUL.")
    return value


def _vocabulary(value: Sequence[object], name: str = "candidate_vocabulary") -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not a string.")
    output_values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} entries must be strings; coercion is forbidden.")
        output_values.append(_strict_text(item, f"{name} entry"))
    output = tuple(output_values)
    if len(output) < 2 or len(set(output)) != len(output):
        raise ValueError(f"{name} must contain at least two unique entries.")
    return output


def _readonly_array(value: object, *, dtype: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if dtype == np.int64 and (
        not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must have an integer, non-boolean dtype; coercion is forbidden.")
    array = np.array(raw, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _strict_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    output = int(value)
    if output < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return output


@dataclass(frozen=True)
class DecisionLossConfig:
    """Frozen, JSON-serializable P2 objective and sampler contract."""

    candidate_vocabulary: tuple[str, ...]
    max_repetitions: int
    lambda_candidate: float = 0.0
    temperature: float = 1.0
    prefix_weight_mode: str = "final"
    trial_weight: float = 1.0
    aggregation: str = DECISION_AGGREGATION
    evidence_count_beta: float | None = None
    temperature_schedule: str = DECISION_TEMPERATURE_SCHEDULE
    min_decisions_per_batch: int = 2
    min_rows_per_batch: int = 128
    batch_policy: str = DECISION_BATCH_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_vocabulary", _vocabulary(self.candidate_vocabulary))
        object.__setattr__(
            self,
            "max_repetitions",
            _strict_integer(self.max_repetitions, "max_repetitions", minimum=1),
        )
        if isinstance(self.lambda_candidate, bool) or not isinstance(
            self.lambda_candidate, Real
        ):
            raise ValueError("lambda_candidate must be numeric.")
        weight = float(self.lambda_candidate)
        if not math.isfinite(weight) or not (weight == 0.0 or 0.1 <= weight <= 0.5):
            raise ValueError(
                "lambda_candidate must be 0 (P2 disabled) or in the preregistered [0.1,0.5] range."
            )
        object.__setattr__(self, "lambda_candidate", weight)
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, Real):
            raise ValueError("temperature must be numeric.")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and strictly positive.")
        object.__setattr__(self, "temperature", temperature)
        if self.prefix_weight_mode not in PREFIX_WEIGHT_MODES:
            raise ValueError(
                f"prefix_weight_mode must be one of {sorted(PREFIX_WEIGHT_MODES)}."
            )
        if isinstance(self.trial_weight, bool) or not isinstance(self.trial_weight, Real):
            raise ValueError("trial_weight must be numeric.")
        if float(self.trial_weight) != 1.0:
            raise ValueError("trial_weight is fixed at 1; zero or reweighted trial loss is forbidden.")
        object.__setattr__(self, "trial_weight", 1.0)
        if self.aggregation != DECISION_AGGREGATION:
            raise ValueError(
                f"aggregation must remain {DECISION_AGGREGATION!r} in the first P2 implementation."
            )
        if self.evidence_count_beta is not None:
            raise ValueError(
                "evidence_count_beta is not implemented; evidence-aware scoring must be a separate arm."
            )
        if self.temperature_schedule != DECISION_TEMPERATURE_SCHEDULE:
            raise ValueError(
                "Only a constant temperature is implemented; evidence-aware tau schedules are rejected."
            )
        object.__setattr__(
            self,
            "min_decisions_per_batch",
            _strict_integer(
                self.min_decisions_per_batch,
                "min_decisions_per_batch",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "min_rows_per_batch",
            _strict_integer(self.min_rows_per_batch, "min_rows_per_batch", minimum=128),
        )
        if self.batch_policy != DECISION_BATCH_POLICY:
            raise ValueError(f"batch_policy must be {DECISION_BATCH_POLICY!r}.")

    @property
    def active(self) -> bool:
        return self.lambda_candidate > 0.0

    @property
    def prefix_weights(self) -> tuple[float, ...]:
        if self.prefix_weight_mode == "final":
            return (0.0,) * (self.max_repetitions - 1) + (1.0,)
        weight = 1.0 / self.max_repetitions
        return (weight,) * self.max_repetitions

    def record(self) -> dict[str, object]:
        return {
            "schema": DECISION_LOSS_CONFIG_SCHEMA,
            "protocol": DIRECT_MEMBERSHIP_PROTOCOL,
            "candidate_vocabulary": list(self.candidate_vocabulary),
            "max_repetitions": self.max_repetitions,
            "lambda_candidate": self.lambda_candidate,
            "temperature": self.temperature,
            "prefix_weight_mode": self.prefix_weight_mode,
            "prefix_weights": list(self.prefix_weights),
            "trial_weight": self.trial_weight,
            "aggregation": self.aggregation,
            "evidence_count_beta": self.evidence_count_beta,
            "temperature_schedule": self.temperature_schedule,
            "min_decisions_per_batch": self.min_decisions_per_batch,
            "min_rows_per_batch": self.min_rows_per_batch,
            "batch_policy": self.batch_policy,
            "active": self.active,
        }

    @classmethod
    def from_record(cls, value: object) -> DecisionLossConfig:
        if not isinstance(value, Mapping):
            raise ValueError("decision loss config must be a JSON mapping.")
        expected = {
            "schema",
            "protocol",
            "candidate_vocabulary",
            "max_repetitions",
            "lambda_candidate",
            "temperature",
            "prefix_weight_mode",
            "prefix_weights",
            "trial_weight",
            "aggregation",
            "evidence_count_beta",
            "temperature_schedule",
            "min_decisions_per_batch",
            "min_rows_per_batch",
            "batch_policy",
            "active",
        }
        if set(value) != expected:
            raise ValueError(
                "decision loss config fields mismatch: "
                f"missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)}."
            )
        if value["schema"] != DECISION_LOSS_CONFIG_SCHEMA:
            raise ValueError("decision loss config schema is unsupported.")
        if value["protocol"] != DIRECT_MEMBERSHIP_PROTOCOL:
            raise ValueError("decision loss config protocol is not direct membership.")
        if not isinstance(value["candidate_vocabulary"], (list, tuple)):
            raise ValueError("serialized candidate_vocabulary must be a JSON array.")
        if not isinstance(value["prefix_weights"], (list, tuple)):
            raise ValueError("serialized prefix_weights must be a JSON array.")
        if not isinstance(value["active"], bool):
            raise ValueError("serialized active must be boolean.")
        config = cls(
            candidate_vocabulary=value["candidate_vocabulary"],
            max_repetitions=value["max_repetitions"],
            lambda_candidate=value["lambda_candidate"],
            temperature=value["temperature"],
            prefix_weight_mode=value["prefix_weight_mode"],
            trial_weight=value["trial_weight"],
            aggregation=value["aggregation"],
            evidence_count_beta=value["evidence_count_beta"],
            temperature_schedule=value["temperature_schedule"],
            min_decisions_per_batch=value["min_decisions_per_batch"],
            min_rows_per_batch=value["min_rows_per_batch"],
            batch_policy=value["batch_policy"],
        )
        if list(config.prefix_weights) != list(value["prefix_weights"]):
            raise ValueError("serialized prefix_weights disagree with prefix_weight_mode.")
        if value["active"] is not config.active:
            raise ValueError("serialized active flag disagrees with lambda_candidate.")
        return config


@dataclass(frozen=True)
class DirectDecisionMetadata:
    """P2 metadata aligned one-to-one with the matrix rows passed to ``fit``."""

    candidate_vocabulary: tuple[str, ...]
    requested_decision_ids: tuple[str, ...]
    truth_indices: tuple[int, ...]
    decision_indices: np.ndarray
    candidate_indices: np.ndarray
    target_indices: np.ndarray
    repetition_indices: np.ndarray
    source_evidence_indices: np.ndarray
    timeline_fingerprint: str
    source_event_contract: str
    timeline_complete: bool
    max_repetitions: int
    requested_row_counts: tuple[int, ...]
    retained_row_counts: tuple[int, ...]
    later_discarded_row_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        vocabulary = _vocabulary(self.candidate_vocabulary)
        decisions = tuple(_strict_text(value, "requested_decision_id") for value in self.requested_decision_ids)
        if not decisions or len(set(decisions)) != len(decisions):
            raise ValueError("requested_decision_ids must be non-empty and unique.")
        truths = tuple(
            _strict_integer(value, "truth index") for value in self.truth_indices
        )
        if len(truths) != len(decisions) or any(value >= len(vocabulary) for value in truths):
            raise ValueError("truth_indices must provide one in-vocabulary truth per requested decision.")
        max_repetitions = _strict_integer(
            self.max_repetitions, "max_repetitions", minimum=1
        )
        count_fields: dict[str, tuple[int, ...]] = {}
        for name in (
            "requested_row_counts",
            "retained_row_counts",
            "later_discarded_row_counts",
        ):
            raw = getattr(self, name)
            if not isinstance(raw, (tuple, list)) or len(raw) != len(decisions):
                raise ValueError(f"{name} must contain one count per requested decision.")
            counts = tuple(_strict_integer(value, f"{name} entry") for value in raw)
            count_fields[name] = counts
            object.__setattr__(self, name, counts)
        arrays = {
            "decision_indices": _readonly_array(
                self.decision_indices, dtype=np.int64, name="decision_indices"
            ),
            "candidate_indices": _readonly_array(
                self.candidate_indices, dtype=np.int64, name="candidate_indices"
            ),
            "target_indices": _readonly_array(
                self.target_indices, dtype=np.int64, name="target_indices"
            ),
            "repetition_indices": _readonly_array(
                self.repetition_indices, dtype=np.int64, name="repetition_indices"
            ),
            "source_evidence_indices": _readonly_array(
                self.source_evidence_indices,
                dtype=np.int64,
                name="source_evidence_indices",
            ),
        }
        if len({len(value) for value in arrays.values()}) != 1:
            raise ValueError("row-aligned direct decision fields must have equal length.")
        row_count = len(arrays["decision_indices"])
        if row_count:
            if np.any(arrays["decision_indices"] < 0) or np.any(
                arrays["decision_indices"] >= len(decisions)
            ):
                raise ValueError("decision_indices contain an out-of-vocabulary decision.")
            if np.any(arrays["candidate_indices"] < 0) or np.any(
                arrays["candidate_indices"] >= len(vocabulary)
            ):
                raise ValueError("candidate_indices contain an out-of-vocabulary candidate.")
            if np.any(arrays["target_indices"] < 0) or np.any(
                arrays["target_indices"] >= len(vocabulary)
            ):
                raise ValueError("target_indices contain an out-of-vocabulary truth.")
            if np.any(arrays["repetition_indices"] < 0):
                raise ValueError("repetition_indices must be non-negative.")
            if len(set(arrays["source_evidence_indices"].tolist())) != row_count:
                raise ValueError("source_evidence_indices must be unique.")
            if np.any(arrays["source_evidence_indices"] < 0):
                raise ValueError("source_evidence_indices must be non-negative.")
            if np.any(np.diff(arrays["source_evidence_indices"]) <= 0):
                raise ValueError("source_evidence_indices must be in strict source-row order.")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "candidate_vocabulary", vocabulary)
        object.__setattr__(self, "requested_decision_ids", decisions)
        object.__setattr__(self, "truth_indices", truths)
        object.__setattr__(
            self,
            "timeline_fingerprint",
            _strict_text(self.timeline_fingerprint, "timeline_fingerprint"),
        )
        object.__setattr__(
            self,
            "source_event_contract",
            _strict_text(self.source_event_contract, "source_event_contract"),
        )
        if self.source_event_contract not in DIRECT_SOURCE_EVENT_CONTRACTS:
            raise ValueError(
                "source_event_contract is not eligible for direct-membership decision loss."
            )
        if not isinstance(self.timeline_complete, (bool, np.bool_)):
            raise ValueError("timeline_complete must be boolean.")
        object.__setattr__(self, "timeline_complete", bool(self.timeline_complete))
        object.__setattr__(self, "max_repetitions", max_repetitions)

        expected_targets = np.asarray(truths, dtype=np.int64)[arrays["decision_indices"]]
        if row_count and not np.array_equal(expected_targets, arrays["target_indices"]):
            raise ValueError("row target_indices disagree with truth_indices authority.")
        observed_counts = tuple(
            int(np.count_nonzero(arrays["decision_indices"] == index))
            for index in range(len(decisions))
        )
        if observed_counts != count_fields["retained_row_counts"]:
            raise ValueError("retained_row_counts disagree with row-aligned decision_indices.")
        if any(
            retained > requested
            for retained, requested in zip(
                count_fields["retained_row_counts"],
                count_fields["requested_row_counts"],
                strict=True,
            )
        ):
            raise ValueError("retained rows cannot exceed requested pre-QC rows.")

    @property
    def n_rows(self) -> int:
        return int(len(self.decision_indices))

    @property
    def truth_by_decision(self) -> dict[str, str]:
        return {
            decision: self.candidate_vocabulary[self.truth_indices[index]]
            for index, decision in enumerate(self.requested_decision_ids)
        }

    @classmethod
    def from_timeline(
        cls,
        timeline: ScheduledEventTimeline,
        *,
        requested_row_mask: np.ndarray,
        retained_row_mask: np.ndarray,
        max_repetitions: int,
        source_event_contract: str,
        candidate_task_contract: object | None = None,
    ) -> DirectDecisionMetadata:
        """Map scheduled events through ``evidence_index`` after runner/QC row filtering."""

        if not isinstance(timeline, ScheduledEventTimeline):
            raise TypeError("direct decision metadata requires ScheduledEventTimeline.")
        event_contract = _strict_text(source_event_contract, "source_event_contract")
        if event_contract not in DIRECT_SOURCE_EVENT_CONTRACTS:
            raise DirectDecisionContractError(
                invalid_decision_ledger(
                    phase="metadata",
                    reason=(
                        f"source_event_contract_{event_contract}_is_ineligible_for_"
                        "direct_membership"
                    ),
                    metadata=None,
                    config=None,
                )
            )
        if candidate_task_contract is not None:
            if isinstance(candidate_task_contract, Mapping):
                task_record = dict(candidate_task_contract)
            elif hasattr(candidate_task_contract, "record"):
                task_record = dict(candidate_task_contract.record())
            else:
                raise TypeError("candidate_task_contract must expose a serializable record.")
            raise DirectDecisionContractError(
                invalid_decision_ledger(
                    phase="metadata",
                    reason=(
                        "candidate_task_contract_is_not_direct_membership:"
                        f"{task_record.get('membership_kind')}:{task_record.get('target_representation')}"
                    ),
                    metadata=None,
                    config=None,
                )
            )

        horizon = _strict_integer(max_repetitions, "max_repetitions", minimum=1)
        n_available = timeline.n_available
        timeline.validate(n_epochs=n_available)
        if not timeline.has_repetition_structure:
            raise DirectDecisionContractError(
                invalid_decision_ledger(
                    phase="metadata",
                    reason="scheduled_timeline_lacks_candidate_truth_or_repetition_metadata",
                    metadata=None,
                    config=None,
                )
            )
        if not timeline.supports_full_candidate_chain:
            raise DirectDecisionContractError(
                invalid_decision_ledger(
                    phase="metadata",
                    reason=(
                        "scheduled_timeline_lacks_one_frozen_complete_candidate_"
                        "vocabulary"
                    ),
                    metadata=None,
                    config=None,
                )
            )
        candidates = np.asarray(timeline.candidate_ids).astype(str)
        targets = np.asarray(timeline.target_candidate_ids).astype(str)
        groups = np.asarray(timeline.group_ids).astype(str)
        repetitions = np.asarray(timeline.repetition_indices, dtype=np.int64)
        evidence_indices = np.asarray(timeline.evidence_indices, dtype=np.int64)
        available_event_rows = np.flatnonzero(evidence_indices >= 0)
        available_event_rows = available_event_rows[
            np.argsort(evidence_indices[available_event_rows], kind="stable")
        ]
        requested_epoch_rows = np.asarray(requested_row_mask)
        retained_epoch_rows = np.asarray(retained_row_mask)
        for name, mask in (
            ("requested_row_mask", requested_epoch_rows),
            ("retained_row_mask", retained_epoch_rows),
        ):
            if mask.dtype != np.dtype(bool) or mask.shape != (n_available,):
                raise ValueError(
                    f"{name} must be a boolean vector aligned with available epochs."
                )
        if bool(np.any(retained_epoch_rows & ~requested_epoch_rows)):
            raise ValueError("retained_row_mask must be a subset of requested_row_mask.")
        available_repetitions = repetitions[available_event_rows]
        within_horizon = available_repetitions < horizon
        requested_budget_rows = requested_epoch_rows & within_horizon
        retained_budget_rows = retained_epoch_rows & within_horizon
        requested_event_rows = available_event_rows[requested_budget_rows]
        if len(requested_event_rows) == 0:
            raise DirectDecisionContractError(
                invalid_decision_ledger(
                    phase="metadata",
                    reason="requested_direct_decision_scope_is_empty",
                    metadata=None,
                    config=None,
                )
            )
        requested = tuple(sorted(np.unique(groups[requested_event_rows]).tolist()))
        decision_to_index = {decision: index for index, decision in enumerate(requested)}
        # Freeze the schedule-wide vocabulary before the source scope, QC, or
        # repetition horizon can remove a candidate from every retained row.
        vocabulary = tuple(sorted(np.unique(candidates).tolist()))
        candidate_to_index = {
            candidate: index for index, candidate in enumerate(vocabulary)
        }
        truth_indices: list[int] = []
        for decision in requested:
            rows = np.flatnonzero(groups == decision)
            decision_truths = np.unique(targets[rows])
            if len(decision_truths) != 1 or decision_truths[0] not in candidate_to_index:
                raise DirectDecisionContractError(
                    invalid_decision_ledger(
                        phase="metadata",
                        reason=f"decision_{decision}_does_not_have_one_in_vocabulary_truth",
                        metadata=None,
                        config=None,
                    )
                )
            truth_indices.append(candidate_to_index[str(decision_truths[0])])

        selected_event_rows = available_event_rows[
            retained_budget_rows
            & np.isin(groups[available_event_rows], requested)
        ]
        selected_source_indices = evidence_indices[selected_event_rows]
        if len(selected_source_indices) and not np.all(np.diff(selected_source_indices) > 0):
            raise RuntimeError("selected evidence indices are not in strict row order.")

        return cls(
            candidate_vocabulary=vocabulary,
            requested_decision_ids=requested,
            truth_indices=tuple(truth_indices),
            decision_indices=np.asarray(
                [decision_to_index[value] for value in groups[selected_event_rows]],
                dtype=np.int64,
            ),
            candidate_indices=np.asarray(
                [candidate_to_index[value] for value in candidates[selected_event_rows]],
                dtype=np.int64,
            ),
            target_indices=np.asarray(
                [candidate_to_index[value] for value in targets[selected_event_rows]],
                dtype=np.int64,
            ),
            repetition_indices=repetitions[selected_event_rows],
            source_evidence_indices=selected_source_indices,
            timeline_fingerprint=timeline.subset_groups(requested).fingerprint(),
            source_event_contract=event_contract,
            timeline_complete=timeline.complete,
            max_repetitions=horizon,
            requested_row_counts=tuple(
                int(np.count_nonzero(groups[requested_event_rows] == decision))
                for decision in requested
            ),
            retained_row_counts=tuple(
                int(np.count_nonzero(groups[selected_event_rows] == decision))
                for decision in requested
            ),
            later_discarded_row_counts=tuple(
                int(
                    np.count_nonzero(
                        requested_epoch_rows
                        & (groups[available_event_rows] == decision)
                        & ~within_horizon
                    )
                )
                for decision in requested
            ),
        )

    def subset_for_decisions(
        self,
        decision_ids: Sequence[str],
    ) -> DirectDecisionMetadata:
        requested = tuple(_strict_text(value, "decision_id") for value in decision_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("decision_ids must be non-empty and unique.")
        old_index = {value: index for index, value in enumerate(self.requested_decision_ids)}
        if any(value not in old_index for value in requested):
            raise ValueError("decision_ids contain values outside requested_decision_ids.")
        old_to_new = {old_index[value]: index for index, value in enumerate(requested)}
        mask = np.isin(self.decision_indices, tuple(old_to_new))
        old_rows = self.decision_indices[mask]
        return DirectDecisionMetadata(
            candidate_vocabulary=self.candidate_vocabulary,
            requested_decision_ids=requested,
            truth_indices=tuple(self.truth_indices[old_index[value]] for value in requested),
            decision_indices=np.asarray([old_to_new[int(value)] for value in old_rows]),
            candidate_indices=self.candidate_indices[mask],
            target_indices=self.target_indices[mask],
            repetition_indices=self.repetition_indices[mask],
            source_evidence_indices=self.source_evidence_indices[mask],
            timeline_fingerprint=self.timeline_fingerprint,
            source_event_contract=self.source_event_contract,
            timeline_complete=self.timeline_complete,
            max_repetitions=self.max_repetitions,
            requested_row_counts=tuple(
                self.requested_row_counts[old_index[value]] for value in requested
            ),
            retained_row_counts=tuple(
                self.retained_row_counts[old_index[value]] for value in requested
            ),
            later_discarded_row_counts=tuple(
                self.later_discarded_row_counts[old_index[value]] for value in requested
            ),
        )

    def record(self, *, include_rows: bool = True) -> dict[str, object]:
        output: dict[str, object] = {
            "schema": DIRECT_DECISION_METADATA_SCHEMA,
            "protocol": DIRECT_MEMBERSHIP_PROTOCOL,
            "candidate_vocabulary": list(self.candidate_vocabulary),
            "requested_decision_ids": list(self.requested_decision_ids),
            "truth_by_decision": self.truth_by_decision,
            "truth_indices": list(self.truth_indices),
            "timeline_fingerprint": self.timeline_fingerprint,
            "source_event_contract": self.source_event_contract,
            "timeline_complete": self.timeline_complete,
            "max_repetitions": self.max_repetitions,
            "requested_rows": int(sum(self.requested_row_counts)),
            "retained_rows": int(sum(self.retained_row_counts)),
            "later_discarded_rows": int(sum(self.later_discarded_row_counts)),
            "requested_row_counts": list(self.requested_row_counts),
            "retained_row_counts": list(self.retained_row_counts),
            "later_discarded_row_counts": list(self.later_discarded_row_counts),
            "row_count": self.n_rows,
        }
        if include_rows:
            output["rows"] = {
                "decision_indices": self.decision_indices.tolist(),
                "candidate_indices": self.candidate_indices.tolist(),
                "target_indices": self.target_indices.tolist(),
                "repetition_indices": self.repetition_indices.tolist(),
                "source_evidence_indices": self.source_evidence_indices.tolist(),
            }
        return output


@dataclass(frozen=True)
class DecisionLossResult:
    """One decision/prefix result retaining differentiable tensors when complete."""

    decision_id: str
    evidence_level: int
    status: str
    candidate_vocabulary: tuple[str, ...]
    target_candidate: str | None
    missing_candidates: tuple[str, ...]
    scores: torch.Tensor
    posterior: torch.Tensor | None
    loss: torch.Tensor | None
    argmax_candidate: str | None
    failure_reason: str | None = None

    def record(self) -> dict[str, object]:
        detached_scores = self.scores.detach().float().cpu().tolist()
        scores = ["-inf" if value == float("-inf") else float(value) for value in detached_scores]
        posterior = (
            None
            if self.posterior is None
            else [float(value) for value in self.posterior.detach().float().cpu().tolist()]
        )
        return {
            "schema": DECISION_LOSS_RESULT_SCHEMA,
            "decision_id": self.decision_id,
            "evidence_level": self.evidence_level,
            "status": self.status,
            "candidate_vocabulary": list(self.candidate_vocabulary),
            "target_candidate": self.target_candidate,
            "missing_candidates": list(self.missing_candidates),
            "scores": scores,
            "posterior": posterior,
            "loss": None if self.loss is None else float(self.loss.detach().float().item()),
            "argmax_candidate": self.argmax_candidate,
            "failure_reason": self.failure_reason,
        }


class DecisionLevelLoss(nn.Module):
    """Count-neutral candidate CE with explicit incomplete-decision state."""

    def __init__(
        self,
        *,
        candidate_vocabulary: Sequence[object] | None = None,
        num_candidates: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if (candidate_vocabulary is None) == (num_candidates is None):
            raise ValueError("provide exactly one of candidate_vocabulary or num_candidates.")
        if num_candidates is not None:
            count = _strict_integer(num_candidates, "num_candidates", minimum=2)
            candidate_vocabulary = tuple(str(index) for index in range(count))
        assert candidate_vocabulary is not None
        self.candidate_vocabulary = _vocabulary(candidate_vocabulary)
        if isinstance(temperature, bool) or not isinstance(temperature, Real):
            raise ValueError("temperature must be numeric.")
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and strictly positive.")

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_vocabulary)

    def forward(
        self,
        evidence: torch.Tensor,
        candidate_membership: torch.Tensor,
        target_candidate: torch.Tensor | int,
        *,
        decision_id: str = "decision",
        evidence_level: int = 1,
    ) -> DecisionLossResult:
        if evidence.ndim != 1 or not evidence.is_floating_point():
            raise ValueError("evidence must be a one-dimensional floating tensor.")
        if candidate_membership.ndim != 1 or candidate_membership.shape != evidence.shape:
            raise ValueError("candidate_membership must be a vector aligned with evidence.")
        if candidate_membership.dtype != torch.long:
            raise ValueError("candidate_membership must have dtype torch.long.")
        if candidate_membership.device != evidence.device:
            raise ValueError("candidate_membership and evidence must share a device.")
        level = _strict_integer(evidence_level, "evidence_level", minimum=1)
        decision_name = _strict_text(decision_id, "decision_id")
        if isinstance(target_candidate, torch.Tensor):
            if (
                target_candidate.numel() != 1
                or target_candidate.dtype == torch.bool
                or target_candidate.dtype
                not in {
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                }
            ):
                raise ValueError("target_candidate must be one scalar integer tensor.")
            target = int(target_candidate.item())
        else:
            if isinstance(target_candidate, bool) or not isinstance(target_candidate, Integral):
                raise ValueError("target_candidate must be an integer.")
            target = int(target_candidate)
        if not bool(torch.isfinite(evidence).all().item()):
            return DecisionLossResult(
                decision_id=decision_name,
                evidence_level=level,
                status="invalid",
                candidate_vocabulary=self.candidate_vocabulary,
                target_candidate=(
                    self.candidate_vocabulary[target]
                    if 0 <= target < self.num_candidates
                    else None
                ),
                missing_candidates=(),
                scores=evidence.new_full((self.num_candidates,), float("-inf")),
                posterior=None,
                loss=None,
                argmax_candidate=None,
                failure_reason="evidence_contains_nan_or_infinity",
            )
        scores_by_candidate: list[torch.Tensor] = []
        observed_candidates: list[bool] = []
        missing: list[str] = []
        outside = bool(
            ((candidate_membership < 0) | (candidate_membership >= self.num_candidates))
            .any()
            .item()
        )
        for index, candidate in enumerate(self.candidate_vocabulary):
            rows = candidate_membership == index
            if bool(rows.any().item()):
                scores_by_candidate.append(evidence[rows].mean())
                observed_candidates.append(True)
            else:
                scores_by_candidate.append(evidence.new_tensor(float("-inf")))
                observed_candidates.append(False)
                missing.append(candidate)
        scores = torch.stack(scores_by_candidate)
        observed_mask = torch.as_tensor(
            observed_candidates,
            dtype=torch.bool,
            device=evidence.device,
        )
        if not bool(torch.isfinite(scores[observed_mask]).all().item()):
            return DecisionLossResult(
                decision_id=decision_name,
                evidence_level=level,
                status="invalid",
                candidate_vocabulary=self.candidate_vocabulary,
                target_candidate=(
                    self.candidate_vocabulary[target]
                    if 0 <= target < self.num_candidates
                    else None
                ),
                missing_candidates=tuple(missing),
                scores=evidence.new_full((self.num_candidates,), float("-inf")),
                posterior=None,
                loss=None,
                argmax_candidate=None,
                failure_reason="candidate_mean_overflowed",
            )
        if outside or not 0 <= target < self.num_candidates:
            return DecisionLossResult(
                decision_id=decision_name,
                evidence_level=level,
                status="invalid",
                candidate_vocabulary=self.candidate_vocabulary,
                target_candidate=(
                    self.candidate_vocabulary[target] if 0 <= target < self.num_candidates else None
                ),
                missing_candidates=tuple(missing),
                scores=scores,
                posterior=None,
                loss=None,
                argmax_candidate=None,
                failure_reason="candidate_or_truth_outside_frozen_vocabulary",
            )
        if missing:
            return DecisionLossResult(
                decision_id=decision_name,
                evidence_level=level,
                status="incomplete",
                candidate_vocabulary=self.candidate_vocabulary,
                target_candidate=self.candidate_vocabulary[target],
                missing_candidates=tuple(missing),
                scores=scores,
                posterior=None,
                loss=None,
                argmax_candidate=None,
                failure_reason="candidate_coverage_incomplete",
            )
        scaled = scores / self.temperature
        posterior = torch.softmax(scaled, dim=0)
        loss = F.cross_entropy(
            scaled.unsqueeze(0),
            torch.tensor([target], device=evidence.device, dtype=torch.long),
        )
        maximum = scores.max()
        winners = torch.nonzero(scores == maximum, as_tuple=False).flatten()
        argmax_candidate = (
            self.candidate_vocabulary[int(winners[0].item())]
            if len(winners) == 1
            else None
        )
        return DecisionLossResult(
            decision_id=decision_name,
            evidence_level=level,
            status="complete" if argmax_candidate is not None else "tie",
            candidate_vocabulary=self.candidate_vocabulary,
            target_candidate=self.candidate_vocabulary[target],
            missing_candidates=(),
            scores=scores,
            posterior=posterior,
            loss=loss,
            argmax_candidate=argmax_candidate,
        )

    def packed_loss(
        self,
        evidence: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        max_repetitions: int,
        prefix_weights: Sequence[float],
        validate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return objective and per-prefix mean NLL for candidate-major packed rows."""

        repetitions = _strict_integer(max_repetitions, "max_repetitions", minimum=1)
        if evidence.ndim != 1 or not evidence.is_floating_point():
            raise ValueError("packed evidence must be a floating vector.")
        if target_indices.ndim != 1 or target_indices.dtype != torch.long:
            raise ValueError("target_indices must be a one-dimensional torch.long tensor.")
        if len(target_indices) == 0:
            raise ValueError("target_indices cannot be empty.")
        if target_indices.device != evidence.device:
            raise ValueError("target_indices and evidence must share a device.")
        expected_rows = len(target_indices) * self.num_candidates * repetitions
        if len(evidence) != expected_rows:
            raise ValueError(
                "packed evidence row count must equal decisions*candidates*repetitions."
            )
        weights = tuple(float(value) for value in prefix_weights)
        if (
            len(weights) != repetitions
            or any(not math.isfinite(value) or value < 0.0 for value in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError("prefix_weights must be finite, non-negative, and sum to one.")
        if validate and bool(
            ((target_indices < 0) | (target_indices >= self.num_candidates)).any().item()
        ):
            raise ValueError("target_indices contain an out-of-vocabulary truth.")
        if not bool(torch.isfinite(evidence).all().item()):
            raise ValueError("packed evidence contains NaN or infinity.")
        packed = evidence.reshape(len(target_indices), self.num_candidates, repetitions)
        divisor = torch.arange(
            1,
            repetitions + 1,
            device=evidence.device,
            dtype=evidence.dtype,
        )
        prefix_scores = packed.cumsum(dim=2) / divisor.view(1, 1, -1)
        if not bool(torch.isfinite(prefix_scores).all().item()):
            raise ValueError("packed candidate prefix means overflowed.")
        log_posterior = F.log_softmax(prefix_scores / self.temperature, dim=1)
        selected = target_indices.view(-1, 1, 1).expand(-1, 1, repetitions)
        per_decision_prefix = -log_posterior.gather(1, selected).squeeze(1)
        per_prefix = per_decision_prefix.mean(dim=0)
        weight_tensor = evidence.new_tensor(weights)
        return torch.sum(per_prefix * weight_tensor), per_prefix


@dataclass(frozen=True)
class DecisionBatch:
    decision_ids: tuple[str, ...]
    row_indices: np.ndarray
    target_indices: np.ndarray

    def __post_init__(self) -> None:
        rows = _readonly_array(self.row_indices, dtype=np.int64, name="row_indices")
        targets = _readonly_array(self.target_indices, dtype=np.int64, name="target_indices")
        if len(self.decision_ids) != len(targets) or not len(targets):
            raise ValueError("DecisionBatch needs one target per decision.")
        if len(set(rows.tolist())) != len(rows):
            raise ValueError("DecisionBatch row_indices must be unique.")
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "target_indices", targets)


@dataclass(frozen=True)
class DecisionBatchPlan:
    """Validated sampler plan; every epoch covers every row exactly once."""

    phase: str
    requested_max_rows: int
    rows_per_decision: int
    batch_decision_counts: tuple[int, ...]
    decision_ids: tuple[str, ...]
    truth_indices: tuple[int, ...]
    ordered_rows_by_decision: tuple[np.ndarray, ...]
    min_decisions_per_batch: int
    min_rows_per_batch: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _strict_text(self.phase, "phase"))
        object.__setattr__(
            self,
            "requested_max_rows",
            _strict_integer(self.requested_max_rows, "requested_max_rows", minimum=1),
        )
        object.__setattr__(
            self,
            "rows_per_decision",
            _strict_integer(self.rows_per_decision, "rows_per_decision", minimum=1),
        )
        object.__setattr__(
            self,
            "min_decisions_per_batch",
            _strict_integer(
                self.min_decisions_per_batch,
                "min_decisions_per_batch",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "min_rows_per_batch",
            _strict_integer(self.min_rows_per_batch, "min_rows_per_batch", minimum=128),
        )
        decision_ids = tuple(_strict_text(value, "decision_id") for value in self.decision_ids)
        if not decision_ids or len(set(decision_ids)) != len(decision_ids):
            raise ValueError("decision_ids must be non-empty and unique.")
        object.__setattr__(self, "decision_ids", decision_ids)
        truth_indices = tuple(
            _strict_integer(value, "truth index") for value in self.truth_indices
        )
        if len(truth_indices) != len(decision_ids):
            raise ValueError("truth_indices must contain one truth per decision.")
        object.__setattr__(self, "truth_indices", truth_indices)
        batch_counts = tuple(
            _strict_integer(value, "batch decision count", minimum=1)
            for value in self.batch_decision_counts
        )
        object.__setattr__(self, "batch_decision_counts", batch_counts)
        ordered_rows = tuple(
            _readonly_array(value, dtype=np.int64, name="ordered_rows_by_decision")
            for value in self.ordered_rows_by_decision
        )
        if len(ordered_rows) != len(decision_ids):
            raise ValueError("ordered_rows_by_decision must contain one array per decision.")
        object.__setattr__(self, "ordered_rows_by_decision", ordered_rows)
        if sum(batch_counts) != len(decision_ids):
            raise ValueError("batch_decision_counts must cover every decision exactly once.")
        if any(value < self.min_decisions_per_batch for value in batch_counts):
            raise ValueError("every grouped batch must satisfy min_decisions_per_batch.")
        if any(
            value * self.rows_per_decision < self.min_rows_per_batch
            for value in self.batch_decision_counts
        ):
            raise ValueError("every grouped batch must satisfy min_rows_per_batch.")
        if any(
            value * self.rows_per_decision > self.requested_max_rows
            for value in self.batch_decision_counts
        ):
            raise ValueError("a grouped batch exceeds requested_max_rows.")
        all_rows = np.concatenate(ordered_rows)
        if len(all_rows) != len(set(all_rows.tolist())):
            raise ValueError("decision groups overlap in the batch plan.")
        if sorted(all_rows.tolist()) != list(range(len(all_rows))):
            raise ValueError("batch plan must cover every row exactly once.")
        for rows in ordered_rows:
            if len(rows) != self.rows_per_decision:
                raise ValueError("every direct decision must have the same packed row count.")

    @property
    def n_rows(self) -> int:
        return self.rows_per_decision * len(self.decision_ids)

    @property
    def max_batch_rows(self) -> int:
        return max(self.batch_decision_counts) * self.rows_per_decision

    def batches(self, *, seed: int | None = None) -> tuple[DecisionBatch, ...]:
        order = np.arange(len(self.decision_ids), dtype=np.int64)
        if seed is not None:
            order = np.random.default_rng(int(seed)).permutation(order)
        batches: list[DecisionBatch] = []
        start = 0
        for decision_count in self.batch_decision_counts:
            selected = order[start : start + decision_count]
            start += decision_count
            batches.append(
                DecisionBatch(
                    decision_ids=tuple(self.decision_ids[int(index)] for index in selected),
                    row_indices=np.concatenate(
                        [self.ordered_rows_by_decision[int(index)] for index in selected]
                    ),
                    target_indices=np.asarray(
                        [self.truth_indices[int(index)] for index in selected], dtype=np.int64
                    ),
                )
            )
        flat = np.concatenate([batch.row_indices for batch in batches])
        if sorted(flat.tolist()) != list(range(self.n_rows)):
            raise RuntimeError("epoch decision batches do not cover every row exactly once.")
        return tuple(batches)

    def record(self) -> dict[str, object]:
        return {
            "schema": DECISION_BATCH_PLAN_SCHEMA,
            "policy": DECISION_BATCH_POLICY,
            "phase": self.phase,
            "requested_max_rows": self.requested_max_rows,
            "effective_max_rows": self.max_batch_rows,
            "rows_per_decision": self.rows_per_decision,
            "decision_count": len(self.decision_ids),
            "batch_count": len(self.batch_decision_counts),
            "batch_decision_counts": list(self.batch_decision_counts),
            "batch_row_counts": [
                value * self.rows_per_decision for value in self.batch_decision_counts
            ],
            "min_decisions_per_batch": self.min_decisions_per_batch,
            "min_rows_per_batch": self.min_rows_per_batch,
            "epoch_row_coverage": "exactly_once",
            "within_decision_order": "candidate_then_repetition",
        }


@dataclass(frozen=True)
class DecisionTrainingLedger:
    status: str
    valid: bool
    phase: str
    reason: str | None
    config: Mapping[str, object] | None
    metadata: Mapping[str, object] | None
    decisions: tuple[Mapping[str, object], ...] = ()
    batch_plan: Mapping[str, object] | None = None

    def record(self) -> dict[str, object]:
        return {
            "schema": DECISION_TRAINING_LEDGER_SCHEMA,
            "status": self.status,
            "valid": self.valid,
            "phase": self.phase,
            "reason": self.reason,
            "config": None if self.config is None else dict(self.config),
            "metadata": None if self.metadata is None else dict(self.metadata),
            "decisions": [dict(value) for value in self.decisions],
            "batch_plan": None if self.batch_plan is None else dict(self.batch_plan),
        }


class DirectDecisionContractError(RuntimeError):
    """Fail-closed P2 error carrying a durable, serializable invalid ledger."""

    def __init__(self, ledger: DecisionTrainingLedger) -> None:
        self.ledger = ledger
        self.record = ledger.record()
        super().__init__(f"Direct decision training is invalid: {ledger.reason or ledger.status}.")


def invalid_decision_ledger(
    *,
    phase: str,
    reason: str,
    metadata: DirectDecisionMetadata | None,
    config: DecisionLossConfig | None,
    decisions: Sequence[Mapping[str, object]] = (),
    batch_plan: Mapping[str, object] | None = None,
) -> DecisionTrainingLedger:
    return DecisionTrainingLedger(
        status="invalid",
        valid=False,
        phase=str(phase),
        reason=str(reason),
        config=None if config is None else config.record(),
        metadata=None if metadata is None else metadata.record(include_rows=False),
        decisions=tuple(dict(value) for value in decisions),
        batch_plan=batch_plan,
    )


def build_decision_batch_plan(
    metadata: DirectDecisionMetadata,
    labels: np.ndarray,
    config: DecisionLossConfig,
    *,
    max_batch_rows: int,
    phase: str,
) -> tuple[DecisionBatchPlan, DecisionTrainingLedger]:
    """Validate direct labels/coverage and pack complete decisions without splitting."""

    if not isinstance(metadata, DirectDecisionMetadata):
        raise TypeError("metadata must be DirectDecisionMetadata.")
    if not isinstance(config, DecisionLossConfig):
        raise TypeError("config must be DecisionLossConfig.")
    labels = np.asarray(labels)
    if labels.shape != (metadata.n_rows,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be an integer vector aligned with direct decision metadata.")
    max_rows = _strict_integer(max_batch_rows, "max_batch_rows", minimum=1)
    decision_records: list[dict[str, object]] = []
    invalid_reasons: list[str] = []
    if metadata.candidate_vocabulary != config.candidate_vocabulary:
        invalid_reasons.append("metadata_vocabulary_disagrees_with_frozen_config")
    if metadata.max_repetitions != config.max_repetitions:
        invalid_reasons.append("metadata_horizon_disagrees_with_frozen_config")
    if not metadata.timeline_complete:
        invalid_reasons.append("scheduled_timeline_is_not_complete")

    expected_slots = {
        (candidate, repetition)
        for candidate in range(len(config.candidate_vocabulary))
        for repetition in range(config.max_repetitions)
    }
    ordered_rows_by_decision: list[np.ndarray] = []
    for decision_index, decision_id in enumerate(metadata.requested_decision_ids):
        rows = np.flatnonzero(metadata.decision_indices == decision_index)
        slot_counter = Counter(
            zip(
                metadata.candidate_indices[rows].tolist(),
                metadata.repetition_indices[rows].tolist(),
                strict=True,
            )
        )
        observed_slots = set(slot_counter)
        missing_slots = sorted(expected_slots - observed_slots)
        unexpected_slots = sorted(observed_slots - expected_slots)
        duplicate_slots = sorted(slot for slot, count in slot_counter.items() if count != 1)
        missing_candidates = sorted(
            candidate
            for candidate in range(len(config.candidate_vocabulary))
            if not bool(np.any(metadata.candidate_indices[rows] == candidate))
        )
        target = metadata.truth_indices[decision_index]
        target_mismatch = bool(
            len(rows) and np.any(metadata.target_indices[rows] != target)
        )
        relation_violations = (
            []
            if not len(rows)
            else rows[
                labels[rows].astype(np.int64)
                != (metadata.candidate_indices[rows] == target).astype(np.int64)
            ].tolist()
        )
        decision_valid = not (
            missing_slots
            or unexpected_slots
            or duplicate_slots
            or target_mismatch
            or relation_violations
        )
        truth_name = (
            metadata.candidate_vocabulary[target]
            if 0 <= target < len(metadata.candidate_vocabulary)
            else None
        )
        decision_valid = decision_valid and (
            metadata.candidate_vocabulary == config.candidate_vocabulary
            and truth_name is not None
        )
        record = {
            "decision_id": decision_id,
            "truth_candidate": truth_name,
            "status": "complete" if decision_valid else "incomplete",
            "selected_rows": int(len(rows)),
            "requested_rows": metadata.requested_row_counts[decision_index],
            "retained_rows": metadata.retained_row_counts[decision_index],
            "qc_removed_rows": (
                metadata.requested_row_counts[decision_index]
                - metadata.retained_row_counts[decision_index]
            ),
            "later_discarded_rows": metadata.later_discarded_row_counts[decision_index],
            "expected_rows": len(expected_slots),
            "missing_candidates": [config.candidate_vocabulary[value] for value in missing_candidates],
            "missing_slots": [
                {
                    "candidate": config.candidate_vocabulary[candidate],
                    "repetition": repetition,
                }
                for candidate, repetition in missing_slots
            ],
            "unexpected_slots": [list(value) for value in unexpected_slots],
            "duplicate_slots": [list(value) for value in duplicate_slots],
            "target_mismatch": target_mismatch,
            "label_relation": "y_i=1[c_i=d*]",
            "label_relation_violation_rows": relation_violations,
        }
        decision_records.append(record)
        if not decision_valid:
            invalid_reasons.append(f"decision_{decision_id}_is_incomplete_or_mislabeled")
            continue
        row_by_slot = {
            (int(metadata.candidate_indices[row]), int(metadata.repetition_indices[row])): int(row)
            for row in rows
        }
        ordered_rows_by_decision.append(
            np.asarray(
                [
                    row_by_slot[(candidate, repetition)]
                    for candidate in range(len(config.candidate_vocabulary))
                    for repetition in range(config.max_repetitions)
                ],
                dtype=np.int64,
            )
        )

    if invalid_reasons:
        ledger = invalid_decision_ledger(
            phase=phase,
            reason=";".join(dict.fromkeys(invalid_reasons)),
            metadata=metadata,
            config=config,
            decisions=decision_records,
        )
        raise DirectDecisionContractError(ledger)

    rows_per_decision = len(expected_slots)
    minimum_decisions = max(
        config.min_decisions_per_batch,
        math.ceil(config.min_rows_per_batch / rows_per_decision),
    )
    decision_count = len(metadata.requested_decision_ids)
    maximum_decisions = max_rows // rows_per_decision
    if decision_count < minimum_decisions:
        reason = (
            f"{phase}_has_{decision_count}_decisions_but_requires_at_least_{minimum_decisions}"
        )
        raise DirectDecisionContractError(
            invalid_decision_ledger(
                phase=phase,
                reason=reason,
                metadata=metadata,
                config=config,
                decisions=decision_records,
            )
        )
    if maximum_decisions < minimum_decisions:
        reason = (
            f"effective_batch_budget_{max_rows}_cannot_hold_{minimum_decisions}_complete_decisions"
        )
        raise DirectDecisionContractError(
            invalid_decision_ledger(
                phase=phase,
                reason=reason,
                metadata=metadata,
                config=config,
                decisions=decision_records,
            )
        )

    batch_count = math.ceil(decision_count / maximum_decisions)
    if batch_count > decision_count // minimum_decisions:
        raise DirectDecisionContractError(
            invalid_decision_ledger(
                phase=phase,
                reason="decision_count_cannot_be_partitioned_into_contract_valid_batches",
                metadata=metadata,
                config=config,
                decisions=decision_records,
            )
        )
    # A valid partition need not consist of equal-sized batches plus one tail:
    # e.g. 7 decisions with min/max 2/3 requires (3, 2, 2).  Start every
    # batch at the lower bound and distribute the remaining complete groups.
    remaining_decisions = decision_count - batch_count * minimum_decisions
    batch_capacity = maximum_decisions - minimum_decisions
    counts_list: list[int] = []
    for _ in range(batch_count):
        added = min(batch_capacity, remaining_decisions)
        counts_list.append(minimum_decisions + added)
        remaining_decisions -= added
    if remaining_decisions != 0:
        raise RuntimeError("decision batch partition left unassigned decisions.")
    counts = tuple(counts_list)
    plan = DecisionBatchPlan(
        phase=str(phase),
        requested_max_rows=max_rows,
        rows_per_decision=rows_per_decision,
        batch_decision_counts=counts,
        decision_ids=metadata.requested_decision_ids,
        truth_indices=metadata.truth_indices,
        ordered_rows_by_decision=tuple(ordered_rows_by_decision),
        min_decisions_per_batch=config.min_decisions_per_batch,
        min_rows_per_batch=config.min_rows_per_batch,
    )
    ledger = DecisionTrainingLedger(
        status="valid",
        valid=True,
        phase=str(phase),
        reason=None,
        config=config.record(),
        metadata=metadata.record(include_rows=False),
        decisions=tuple(decision_records),
        batch_plan=plan.record(),
    )
    return plan, ledger


__all__ = [
    "DECISION_AGGREGATION",
    "DECISION_BATCH_PLAN_SCHEMA",
    "DECISION_BATCH_POLICY",
    "DECISION_LOSS_CONFIG_SCHEMA",
    "DECISION_LOSS_RESULT_SCHEMA",
    "DECISION_TRAINING_LEDGER_SCHEMA",
    "DIRECT_DECISION_METADATA_SCHEMA",
    "DIRECT_MEMBERSHIP_PROTOCOL",
    "DIRECT_SOURCE_EVENT_CONTRACTS",
    "DecisionBatch",
    "DecisionBatchPlan",
    "DecisionLevelLoss",
    "DecisionLossConfig",
    "DecisionLossResult",
    "DecisionTrainingLedger",
    "DirectDecisionContractError",
    "DirectDecisionMetadata",
    "build_decision_batch_plan",
    "invalid_decision_ledger",
]
