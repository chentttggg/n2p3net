"""Versioned semantic contracts for training, evaluation, and inference.

File names and arm labels are presentation details. These contracts bind the
resolved scientific choices that determine what a checkpoint or result means.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

TRAINING_RUN_CONTRACT_SCHEMA = "n2p3_training_run_contract/2"
EVALUATION_RUN_CONTRACT_SCHEMA = "n2p3_evaluation_run_contract/2"
STATISTICAL_DESIGN_SCHEMA = "n2p3_statistical_design_contract/1"
DECISION_PLAN_SCHEMA = "n2p3_decision_plan_contract/1"
ARM_CONTRACT_SCHEMA = "n2p3_arm_contract/1"
MIN_PROMOTION_HIT_AT_R = 8

_SHA256_LENGTH = 64


def _validated_sha256(value: str, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic contracts cannot contain NaN or infinity.")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("semantic contract mapping keys must be non-empty strings.")
            output[key] = _canonical_value(item)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    if hasattr(value, "item"):
        return _canonical_value(value.item())
    raise TypeError(f"unsupported semantic contract value {type(value).__name__}.")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON for a JSON-like semantic record."""

    canonical = _canonical_value(value)
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def assert_promotion_evidence_gate(
    test_repetitions: int,
    *,
    primary_evidence_level: int | None = None,
    evidence_levels: Sequence[int] | None = None,
    name: str = "promotion",
) -> None:
    """Enforce the predeclared minimum evidence horizon for promotion claims."""

    if isinstance(test_repetitions, bool) or not isinstance(test_repetitions, int):
        raise ValueError(f"{name} test_repetitions must be an integer.")
    if test_repetitions < MIN_PROMOTION_HIT_AT_R:
        raise ValueError(f"{name} requires test_repetitions >= {MIN_PROMOTION_HIT_AT_R}.")
    if primary_evidence_level is not None:
        if isinstance(primary_evidence_level, bool) or not isinstance(primary_evidence_level, int):
            raise ValueError(f"{name} primary evidence level must be an integer.")
        if not MIN_PROMOTION_HIT_AT_R <= primary_evidence_level <= test_repetitions:
            raise ValueError(
                f"{name} primary hit@R must use {MIN_PROMOTION_HIT_AT_R} <= R <= test_repetitions."
            )
    if evidence_levels is not None:
        normalized = tuple(int(value) for value in evidence_levels)
        if MIN_PROMOTION_HIT_AT_R not in normalized:
            raise ValueError(f"{name} evidence is missing hit@{MIN_PROMOTION_HIT_AT_R}.")
        if primary_evidence_level is not None and primary_evidence_level not in normalized:
            raise ValueError(f"{name} evidence is missing its primary hit@R level.")


def _string_tuple(
    values: Sequence[str], name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    output = tuple(str(value).strip() for value in values)
    if (not allow_empty and not output) or any(not value for value in output):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(output)) != len(output):
        raise ValueError(f"{name} must not contain duplicates.")
    return output


@dataclass(frozen=True)
class TrainingRunContract:
    source_cache_sha256: str
    source_identity_digest: str
    source_snapshot_sha256: str
    architecture: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    validation: Mapping[str, Any]
    objective: Mapping[str, Any]
    seed: int
    training_participant_keys: tuple[str, ...]
    holdout_participant_keys: tuple[str, ...]
    schema: str = TRAINING_RUN_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRAINING_RUN_CONTRACT_SCHEMA:
            raise ValueError(f"training contract schema must be {TRAINING_RUN_CONTRACT_SCHEMA!r}.")
        object.__setattr__(
            self,
            "source_cache_sha256",
            _validated_sha256(self.source_cache_sha256, "source_cache_sha256"),
        )
        object.__setattr__(
            self,
            "source_identity_digest",
            _validated_sha256(self.source_identity_digest, "source_identity_digest"),
        )
        object.__setattr__(
            self,
            "source_snapshot_sha256",
            _validated_sha256(self.source_snapshot_sha256, "source_snapshot_sha256"),
        )
        training = _string_tuple(self.training_participant_keys, "training_participant_keys")
        holdout = _string_tuple(
            self.holdout_participant_keys, "holdout_participant_keys", allow_empty=True
        )
        if set(training) & set(holdout):
            raise ValueError("training and holdout participant keys must be disjoint.")
        object.__setattr__(self, "training_participant_keys", tuple(sorted(training)))
        object.__setattr__(self, "holdout_participant_keys", tuple(sorted(holdout)))
        for name in ("architecture", "preprocessing", "optimizer", "validation", "objective"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a non-empty mapping.")
            _canonical_value(value)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")

    def record(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    def digest(self) -> str:
        return semantic_sha256(self.record())


@dataclass(frozen=True)
class EvaluationRunContract:
    arm_name: str
    model_origin: Mapping[str, Any]
    target_cache_sha256: str
    target_identity_digest: str
    source_snapshot_sha256: str
    target_protocol: Mapping[str, Any]
    adaptation: Mapping[str, Any]
    decision: Mapping[str, Any]
    requested_participant_keys: tuple[str, ...]
    evidence_scope: Mapping[str, Any]
    schema: str = EVALUATION_RUN_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVALUATION_RUN_CONTRACT_SCHEMA:
            raise ValueError(
                f"evaluation contract schema must be {EVALUATION_RUN_CONTRACT_SCHEMA!r}."
            )
        for name in (
            "target_cache_sha256",
            "target_identity_digest",
            "source_snapshot_sha256",
        ):
            object.__setattr__(self, name, _validated_sha256(getattr(self, name), name))
        arm_name = self.arm_name.strip()
        if not arm_name:
            raise ValueError("arm_name must be non-empty.")
        object.__setattr__(self, "arm_name", arm_name)
        if not isinstance(self.model_origin, Mapping):
            raise ValueError("model_origin must be a mapping.")
        origin_kind = self.model_origin.get("kind")
        if origin_kind == "checkpoint":
            _validated_sha256(
                str(self.model_origin.get("checkpoint_sha256", "")),
                "model_origin.checkpoint_sha256",
            )
            _validated_sha256(
                str(self.model_origin.get("training_contract_digest", "")),
                "model_origin.training_contract_digest",
            )
        elif origin_kind == "scratch":
            _validated_sha256(
                str(self.model_origin.get("initialization_contract_digest", "")),
                "model_origin.initialization_contract_digest",
            )
        else:
            raise ValueError("model_origin.kind must be checkpoint or scratch.")
        _canonical_value(self.model_origin)
        participants = _string_tuple(self.requested_participant_keys, "requested_participant_keys")
        object.__setattr__(self, "requested_participant_keys", tuple(sorted(participants)))
        for name in ("target_protocol", "adaptation", "decision", "evidence_scope"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a non-empty mapping.")
            _canonical_value(value)

    def record(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    def digest(self) -> str:
        return semantic_sha256(self.record())


@dataclass(frozen=True, order=True)
class DecisionPlanEntry:
    participant_key: str
    decision_id: str
    target_candidate: str

    def __post_init__(self) -> None:
        for name in ("participant_key", "decision_id", "target_candidate"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"decision plan {name} must be non-empty.")
            object.__setattr__(self, name, value)

    def record(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, order=True)
class ParticipantSelectionFailure:
    participant_key: str
    stage: Literal["decision_selection"]
    reason: str

    def __post_init__(self) -> None:
        participant = self.participant_key.strip()
        reason = self.reason.strip()
        if not participant or not reason:
            raise ValueError("participant selection failure fields must be non-empty.")
        if self.stage != "decision_selection":
            raise ValueError("participant selection failure stage must be decision_selection.")
        object.__setattr__(self, "participant_key", participant)
        object.__setattr__(self, "reason", reason)

    def record(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionPlanContract:
    target_cache_sha256: str
    target_identity_digest: str
    requested_participant_keys: tuple[str, ...]
    entries: tuple[DecisionPlanEntry, ...]
    participant_selection_failures: tuple[ParticipantSelectionFailure, ...] = ()
    schema: str = DECISION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DECISION_PLAN_SCHEMA:
            raise ValueError(f"decision plan schema must be {DECISION_PLAN_SCHEMA!r}.")
        for name in ("target_cache_sha256", "target_identity_digest"):
            object.__setattr__(self, name, _validated_sha256(getattr(self, name), name))
        requested = tuple(
            sorted(_string_tuple(self.requested_participant_keys, "requested_participant_keys"))
        )
        entries = tuple(sorted(self.entries))
        failures = tuple(sorted(self.participant_selection_failures))
        keys = [(entry.participant_key, entry.decision_id) for entry in entries]
        if len(keys) != len(set(keys)):
            raise ValueError("decision plan participant/decision keys must be unique.")
        failure_keys = [failure.participant_key for failure in failures]
        if len(failure_keys) != len(set(failure_keys)):
            raise ValueError("participant selection failures must be unique.")
        planned_participants = {entry.participant_key for entry in entries}
        failed_participants = set(failure_keys)
        if planned_participants & failed_participants:
            raise ValueError("planned and selection-failed participants must be disjoint.")
        if planned_participants | failed_participants != set(requested):
            raise ValueError(
                "every requested participant must have planned decisions or one selection failure."
            )
        object.__setattr__(self, "requested_participant_keys", requested)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "participant_selection_failures", failures)

    @property
    def participant_keys(self) -> tuple[str, ...]:
        return tuple(sorted({entry.participant_key for entry in self.entries}))

    def record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target_cache_sha256": self.target_cache_sha256,
            "target_identity_digest": self.target_identity_digest,
            "requested_participant_keys": list(self.requested_participant_keys),
            "entries": [entry.record() for entry in self.entries],
            "participant_selection_failures": [
                failure.record() for failure in self.participant_selection_failures
            ],
        }

    def digest(self) -> str:
        return semantic_sha256(self.record())

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> DecisionPlanContract:
        entries = value.get("entries")
        failures = value.get("participant_selection_failures")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
            raise ValueError("decision plan entries must be a sequence.")
        if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes, bytearray)):
            raise ValueError("participant selection failures must be a sequence.")
        return cls(
            target_cache_sha256=str(value.get("target_cache_sha256", "")),
            target_identity_digest=str(value.get("target_identity_digest", "")),
            requested_participant_keys=tuple(value.get("requested_participant_keys", ())),
            entries=tuple(
                DecisionPlanEntry(**dict(entry)) for entry in entries if isinstance(entry, Mapping)
            ),
            participant_selection_failures=tuple(
                ParticipantSelectionFailure(**dict(failure))
                for failure in failures
                if isinstance(failure, Mapping)
            ),
            schema=str(value.get("schema", "")),
        )


def training_procedure_record(contract: TrainingRunContract) -> dict[str, Any]:
    """Project a training run onto fields that define the source procedure.

    Seed, fitted participant identities, cache identity, and source snapshot are
    declared experiment axes/provenance. Everything that controls architecture,
    preprocessing, fitting, validation, and objective remains invariant per arm.
    """

    return _canonical_value(
        {
            "architecture": contract.architecture,
            "preprocessing": contract.preprocessing,
            "optimizer": contract.optimizer,
            "validation": contract.validation,
            "objective": contract.objective,
        }
    )


def scratch_procedure_record(initialization: Mapping[str, Any]) -> dict[str, Any]:
    """Remove replicate randomness from a scratch initialization procedure."""

    procedure = {
        key: value for key, value in initialization.items() if key not in {"seed", "random_seed"}
    }
    if not procedure:
        raise ValueError("scratch initialization lacks a seed-independent procedure.")
    return _canonical_value(procedure)


@dataclass(frozen=True)
class ArmContract:
    arm_name: str
    model_origin_kind: Literal["checkpoint", "scratch"]
    source_training_procedure: Mapping[str, Any]
    adaptation_procedure: Mapping[str, Any]
    allowed_variation_axes: tuple[Literal["training_replicate", "participant_partition"], ...]
    schema: str = ARM_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ARM_CONTRACT_SCHEMA:
            raise ValueError(f"arm contract schema must be {ARM_CONTRACT_SCHEMA!r}.")
        arm_name = self.arm_name.strip()
        if not arm_name:
            raise ValueError("arm_name must be non-empty.")
        object.__setattr__(self, "arm_name", arm_name)
        if self.model_origin_kind not in {"checkpoint", "scratch"}:
            raise ValueError("model_origin_kind must be checkpoint or scratch.")
        for name in ("source_training_procedure", "adaptation_procedure"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a non-empty mapping.")
            object.__setattr__(self, name, _canonical_value(value))
        axes = _string_tuple(self.allowed_variation_axes, "allowed_variation_axes")
        allowed = {"training_replicate", "participant_partition"}
        if set(axes) != allowed:
            raise ValueError(
                "BI arm contracts must declare exactly training_replicate and "
                "participant_partition as variation axes."
            )
        object.__setattr__(self, "allowed_variation_axes", axes)

    def record(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    def digest(self) -> str:
        return semantic_sha256(self.record())

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> ArmContract:
        return cls(**dict(value))


@dataclass(frozen=True)
class StatisticalDesignContract:
    inference_scope: Literal["conditional_frozen_models", "training_procedure"]
    participant_cluster_key: str
    training_replicate_keys: tuple[str, ...]
    partition_keys: tuple[str, ...]
    planned_contrasts: tuple[tuple[str, str], ...]
    multiplicity_method: Literal["holm"] = "holm"
    power_plan_digest: str | None = None
    schema: str = STATISTICAL_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATISTICAL_DESIGN_SCHEMA:
            raise ValueError(f"statistical design schema must be {STATISTICAL_DESIGN_SCHEMA!r}.")
        if not self.participant_cluster_key.strip():
            raise ValueError("participant_cluster_key must be non-empty.")
        object.__setattr__(
            self,
            "training_replicate_keys",
            _string_tuple(self.training_replicate_keys, "training_replicate_keys"),
        )
        object.__setattr__(
            self, "partition_keys", _string_tuple(self.partition_keys, "partition_keys")
        )
        contrasts = tuple(
            (str(left).strip(), str(right).strip()) for left, right in self.planned_contrasts
        )
        if not contrasts or any(
            not left or not right or left == right for left, right in contrasts
        ):
            raise ValueError("planned_contrasts must contain distinct non-empty arm pairs.")
        if len(set(contrasts)) != len(contrasts):
            raise ValueError("planned_contrasts must not contain duplicates.")
        object.__setattr__(self, "planned_contrasts", contrasts)
        if self.inference_scope == "training_procedure":
            if self.power_plan_digest is None:
                raise ValueError(
                    "training_procedure inference requires a preregistered power plan."
                )
            object.__setattr__(
                self,
                "power_plan_digest",
                _validated_sha256(self.power_plan_digest, "power_plan_digest"),
            )
        elif self.power_plan_digest is not None:
            object.__setattr__(
                self,
                "power_plan_digest",
                _validated_sha256(self.power_plan_digest, "power_plan_digest"),
            )

    @property
    def allows_training_procedure_claims(self) -> bool:
        return self.inference_scope == "training_procedure"

    def record(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    def digest(self) -> str:
        return semantic_sha256(self.record())


def assert_contract_digest(record: Mapping[str, Any], expected_digest: str, *, name: str) -> None:
    expected = _validated_sha256(expected_digest, f"{name}_digest")
    actual = semantic_sha256(record)
    if actual != expected:
        raise ValueError(f"{name} semantic digest mismatch: expected {expected}, got {actual}.")


__all__ = [
    "ARM_CONTRACT_SCHEMA",
    "DECISION_PLAN_SCHEMA",
    "EVALUATION_RUN_CONTRACT_SCHEMA",
    "STATISTICAL_DESIGN_SCHEMA",
    "TRAINING_RUN_CONTRACT_SCHEMA",
    "ArmContract",
    "DecisionPlanContract",
    "DecisionPlanEntry",
    "EvaluationRunContract",
    "ParticipantSelectionFailure",
    "StatisticalDesignContract",
    "TrainingRunContract",
    "assert_contract_digest",
    "canonical_json_bytes",
    "semantic_sha256",
    "scratch_procedure_record",
    "training_procedure_record",
]
