"""Execute a strict, resumable candidate-promotion experiment matrix.

The versioned JSON plan is the authority for the scientific matrix.  This
orchestrator only composes the existing supervised pretrainer, candidate
cross-decision runner, generic manifest builder, and generic analyzer.  It
does not reinterpret dataset-specific identities or silently reuse artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from data.domain import source_domain_ids  # noqa: E402
from data.epochs import (  # noqa: E402
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
)
from data.identity import DatasetIdentityTable  # noqa: E402
from experiments.analyze_candidate_cross_decision import ANALYSIS_SCHEMA  # noqa: E402
from experiments.build_candidate_cross_decision_manifest import (  # noqa: E402
    MANIFEST_SCHEMA,
    RESULT_SCHEMA,
)
from research.contracts import (  # noqa: E402
    MIN_PROMOTION_HIT_AT_R,
    EvaluationRunContract,
    StatisticalDesignContract,
    assert_contract_digest,
    assert_promotion_evidence_gate,
    semantic_sha256,
)
from research.evaluation import (  # noqa: E402
    source_snapshot_sha256_from_archive_manifest,
)
from train.runtime import cpu_thread_budget, resolve_cpu_worker_plan  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    checkpoint_training_contract,
    load_checkpoint_payload,
)

PLAN_SCHEMA = "n2p3_candidate_promotion_plan/2"
JOURNAL_SCHEMA = "n2p3_candidate_promotion_journal/1"
DRY_RUN_SCHEMA = "n2p3_candidate_promotion_dag/1"
ORCHESTRATOR_LOCK_SCHEMA = "n2p3_candidate_promotion_orchestrator_lock/1"

_PRETRAIN_POOLING_MODES = frozenset(
    {"ms_flatten", "full_unfold", "mlp_full_unfold", "quadratic_full_unfold"}
)
_TRAINING_PRECISIONS = frozenset({"auto", "bf16", "fp32"})
_EVALUATION_HEADS = frozenset(
    {"auto", "zero_shot", "linear", "mlp16", "classifier_fine", "full_fine"}
)
_NORMALIZATION_MODES = frozenset({"source", "target_prefix", "shrinkage"})
_EPOCH_SELECTION_MODES = frozenset({"fixed_budget", "target_time_split"})
_IDENTITY_POLICIES = frozenset({"source", "source_or_global", "global"})
_INFERENCE_SCOPES = frozenset({"conditional_frozen_models", "training_procedure"})
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

TaskKind = Literal["checkpoint", "result", "manifest", "analysis"]


@dataclass(frozen=True)
class TargetSubjectSpec:
    target_subject: str
    source_holdout_subject: str
    partition_key: str


@dataclass(frozen=True)
class SourceArmSpec:
    name: str
    cache: Path
    source_domain_mass: tuple[tuple[str, float], ...]
    risk_within_domain_unit: str | None
    selection_domain: str | None


@dataclass(frozen=True)
class TrainingReplicateSpec:
    key: str
    seed: int


@dataclass(frozen=True)
class EvaluationArmSpec:
    name: str
    head: str
    normalization: str
    epoch_selection: str
    epochs: int | None
    batch_size: int | None
    lr: float | None
    target_stat_weight: float
    fold_local_qc: bool
    adapt_batchnorm: bool


@dataclass(frozen=True)
class TrainingConfig:
    pooling_mode: str
    temporal_kernel_size: int
    epochs: int
    batch_size: int
    precision: str


@dataclass(frozen=True)
class StatisticalDesignSpec:
    inference_scope: str
    planned_contrasts: tuple[tuple[str, str], ...]
    evidence_level: int
    bootstrap_iterations: int
    bootstrap_seed: int
    confidence_level: float
    power_plan: Path | None


@dataclass(frozen=True)
class PromotionPlan:
    path: Path
    resolution_root: Path
    raw_record: Mapping[str, Any]
    digest: str
    source_snapshot_manifest: Path
    target_cache: Path
    target_subjects: tuple[TargetSubjectSpec, ...]
    source_arms: tuple[SourceArmSpec, ...]
    training_replicates: tuple[TrainingReplicateSpec, ...]
    evaluation_arms: tuple[EvaluationArmSpec, ...]
    training: TrainingConfig
    calibration_selections: int
    test_reps: int
    identity_exclusion_policy: str
    output_root: Path
    statistical_design: StatisticalDesignSpec


@dataclass(frozen=True)
class MatrixTask:
    task_id: str
    kind: TaskKind
    argv: tuple[str, ...]
    output: Path
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class PromotionDag:
    plan: PromotionPlan
    device: str
    tasks: tuple[MatrixTask, ...]
    subject_scope_paths: Mapping[str, Path]

    @property
    def task_by_id(self) -> dict[str, MatrixTask]:
        return {task.task_id: task for task in self.tasks}


@dataclass(frozen=True)
class _AttestedCacheInput:
    identity_table: DatasetIdentityTable
    channel_names: tuple[str, ...]
    channel_positions_m: tuple[tuple[float, float, float], ...]
    channel_mask: tuple[bool, ...]
    preprocessing: Mapping[str, Any]
    source_reference: str
    source_domains: tuple[str, ...]
    resolved_record: Mapping[str, Any]


@dataclass(frozen=True)
class _SubprocessInvocation:
    task: MatrixTask
    dependency_outputs: Mapping[str, str]
    attempt_index: int
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class _SubprocessOutcome:
    error: BaseException | None


def evaluation_arm_name(source_arm: str, evaluation_arm: str) -> str:
    """Return the stable arm label consumed by the generic analyzer."""

    return f"{source_arm}__{evaluation_arm}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings.")
    return dict(value)


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a JSON sequence.")
    return list(value)


def _keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    name: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{name} is missing required fields {missing}.")
    if unknown:
        raise ValueError(f"{name} has unsupported fields {unknown}.")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    result = value.strip()
    if "\x00" in result or "\r" in result or "\n" in result:
        raise ValueError(f"{name} contains a forbidden control character.")
    return result


def _component(value: object, name: str) -> str:
    result = _text(value, name)
    if _SAFE_COMPONENT.fullmatch(result) is None:
        raise ValueError(
            f"{name} must be a portable path component matching {_SAFE_COMPONENT.pattern!r}."
        )
    return result


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")
    return result


def _optional_positive_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be null or numeric.")
    result = float(value)
    if not isfinite(result) or not result > 0.0:
        raise ValueError(f"{name} must be positive when provided.")
    return result


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _resolve_plan_path(value: object, name: str, root: Path) -> Path:
    raw = _text(value, name)
    path = Path(raw)
    return (path if path.is_absolute() else root / path).resolve()


def _assert_unique(values: Sequence[object], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")


def _parse_targets(value: object) -> tuple[TargetSubjectSpec, ...]:
    records = _sequence(value, "target_subjects")
    if not records:
        raise ValueError("target_subjects must not be empty.")
    targets: list[TargetSubjectSpec] = []
    for index, raw_record in enumerate(records):
        name = f"target_subjects[{index}]"
        record = _mapping(raw_record, name)
        _keys(
            record,
            required={"target_subject", "source_holdout_subject", "partition_key"},
            name=name,
        )
        target = _text(record["target_subject"], f"{name}.target_subject")
        holdout = _text(record["source_holdout_subject"], f"{name}.source_holdout_subject")
        if "," in holdout:
            raise ValueError(f"{name}.source_holdout_subject must not contain a comma.")
        targets.append(
            TargetSubjectSpec(
                target_subject=target,
                source_holdout_subject=holdout,
                partition_key=_component(record["partition_key"], f"{name}.partition_key"),
            )
        )
    _assert_unique([item.target_subject for item in targets], "target subject mappings")
    _assert_unique([item.source_holdout_subject for item in targets], "source holdout mappings")
    _assert_unique([item.partition_key for item in targets], "partition keys")
    return tuple(targets)


def _parse_source_arms(value: object, root: Path) -> tuple[SourceArmSpec, ...]:
    records = _sequence(value, "source_arms")
    if not records:
        raise ValueError("source_arms must not be empty.")
    arms: list[SourceArmSpec] = []
    for index, raw_record in enumerate(records):
        name = f"source_arms[{index}]"
        record = _mapping(raw_record, name)
        _keys(
            record,
            required={
                "name",
                "cache",
                "source_domain_mass",
                "risk_within_domain_unit",
                "selection_domain",
            },
            name=name,
        )
        raw_masses = record["source_domain_mass"]
        parsed_masses: list[tuple[str, float]] = []
        if raw_masses is not None:
            masses = _mapping(raw_masses, f"{name}.source_domain_mass")
            for raw_domain, raw_mass in masses.items():
                domain = _text(raw_domain, f"{name}.source_domain_mass domain")
                if "=" in domain:
                    raise ValueError(
                        f"{name}.source_domain_mass domain {domain!r} cannot be encoded by the CLI."
                    )
                mass = _number(raw_mass, f"{name}.source_domain_mass[{domain!r}]")
                if mass <= 0.0:
                    raise ValueError(f"{name}.source domain masses must be positive.")
                parsed_masses.append((domain, mass))
            if len(parsed_masses) < 2 or not np.isclose(
                sum(mass for _, mass in parsed_masses), 1.0, rtol=0.0, atol=1e-9
            ):
                raise ValueError(
                    f"{name}.source_domain_mass must contain at least two domains summing to one."
                )
        risk_unit = record["risk_within_domain_unit"]
        selection_domain = record["selection_domain"]
        if parsed_masses:
            risk_unit = _text(risk_unit, f"{name}.risk_within_domain_unit")
            if risk_unit not in {"epoch", "participant"}:
                raise ValueError(f"{name}.risk_within_domain_unit must be epoch or participant.")
            selection_domain = _text(selection_domain, f"{name}.selection_domain")
            if selection_domain not in {domain for domain, _ in parsed_masses}:
                raise ValueError(f"{name}.selection_domain must name one weighted domain.")
        else:
            if risk_unit is not None:
                raise ValueError(f"{name} cannot set risk unit without source domain mass.")
            if selection_domain is not None:
                selection_domain = _text(selection_domain, f"{name}.selection_domain")
        arms.append(
            SourceArmSpec(
                name=_component(record["name"], f"{name}.name"),
                cache=_resolve_plan_path(record["cache"], f"{name}.cache", root),
                source_domain_mass=tuple(sorted(parsed_masses)),
                risk_within_domain_unit=risk_unit,
                selection_domain=selection_domain,
            )
        )
    _assert_unique([item.name for item in arms], "source arm names")
    return tuple(arms)


def _parse_replicates(value: object) -> tuple[TrainingReplicateSpec, ...]:
    records = _sequence(value, "training_replicates")
    if not records:
        raise ValueError("training_replicates must not be empty.")
    replicates: list[TrainingReplicateSpec] = []
    for index, raw_record in enumerate(records):
        name = f"training_replicates[{index}]"
        record = _mapping(raw_record, name)
        _keys(record, required={"key", "seed"}, name=name)
        replicates.append(
            TrainingReplicateSpec(
                key=_component(record["key"], f"{name}.key"),
                seed=_integer(record["seed"], f"{name}.seed", minimum=0),
            )
        )
    _assert_unique([item.key for item in replicates], "training replicate keys")
    _assert_unique([item.seed for item in replicates], "training replicate seeds")
    return tuple(replicates)


def _parse_evaluation_arms(value: object) -> tuple[EvaluationArmSpec, ...]:
    records = _sequence(value, "evaluation_arms")
    if not records:
        raise ValueError("evaluation_arms must not be empty.")
    arms: list[EvaluationArmSpec] = []
    for index, raw_record in enumerate(records):
        name = f"evaluation_arms[{index}]"
        record = _mapping(raw_record, name)
        _keys(
            record,
            required={
                "name",
                "head",
                "normalization",
                "epoch_selection",
                "epochs",
                "batch_size",
                "lr",
                "target_stat_weight",
                "fold_local_qc",
            },
            optional={"adapt_batchnorm"},
            name=name,
        )
        head = _text(record["head"], f"{name}.head")
        if head not in _EVALUATION_HEADS:
            supported = sorted(_EVALUATION_HEADS)
            raise ValueError(
                f"{name}.head must be a current candidate runner head from {supported}."
            )
        normalization = _text(record["normalization"], f"{name}.normalization")
        if normalization not in _NORMALIZATION_MODES:
            raise ValueError(f"{name}.normalization must be one of {sorted(_NORMALIZATION_MODES)}.")
        epoch_selection = _text(record["epoch_selection"], f"{name}.epoch_selection")
        if epoch_selection not in _EPOCH_SELECTION_MODES:
            raise ValueError(
                f"{name}.epoch_selection must be one of {sorted(_EPOCH_SELECTION_MODES)}."
            )
        epochs = (
            None
            if record["epochs"] is None
            else _integer(record["epochs"], f"{name}.epochs", minimum=1)
        )
        batch_size = (
            None
            if record["batch_size"] is None
            else _integer(record["batch_size"], f"{name}.batch_size", minimum=1)
        )
        lr = _optional_positive_number(record["lr"], f"{name}.lr")
        target_stat_weight = _unit_interval(
            record["target_stat_weight"], f"{name}.target_stat_weight"
        )
        fold_local_qc = _boolean(record["fold_local_qc"], f"{name}.fold_local_qc")
        adapt_batchnorm = _boolean(record.get("adapt_batchnorm", False), f"{name}.adapt_batchnorm")
        zero_fit = head in {"auto", "zero_shot"}
        if zero_fit:
            invalid: list[str] = []
            if normalization != "source":
                invalid.append("normalization must be source")
            if epoch_selection != "fixed_budget":
                invalid.append("epoch_selection must be fixed_budget")
            if epochs is not None:
                invalid.append("epochs must be null")
            if batch_size is not None:
                invalid.append("batch_size must be null")
            if lr is not None:
                invalid.append("lr must be null")
            if target_stat_weight != 0.0:
                invalid.append("target_stat_weight must be 0")
            if fold_local_qc:
                invalid.append("fold_local_qc must be false")
            if adapt_batchnorm:
                invalid.append("adapt_batchnorm must be false")
            if invalid:
                raise ValueError(f"{name} checkpoint zero-shot combination is invalid: {invalid}.")
        else:
            missing_fit = [
                field
                for field, field_value in (
                    ("epochs", epochs),
                    ("batch_size", batch_size),
                    ("lr", lr),
                )
                if field_value is None
            ]
            if missing_fit:
                raise ValueError(f"{name} fitted head requires explicit values for {missing_fit}.")
            if normalization != "shrinkage" and target_stat_weight != 0.0:
                raise ValueError(
                    f"{name}.target_stat_weight must be 0 unless normalization is shrinkage."
                )
            if head != "full_fine" and adapt_batchnorm:
                raise ValueError(f"{name}.adapt_batchnorm is only meaningful for full_fine.")
        arms.append(
            EvaluationArmSpec(
                name=_component(record["name"], f"{name}.name"),
                head=head,
                normalization=normalization,
                epoch_selection=epoch_selection,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                target_stat_weight=target_stat_weight,
                fold_local_qc=fold_local_qc,
                adapt_batchnorm=adapt_batchnorm,
            )
        )
    _assert_unique([arm.name for arm in arms], "evaluation arm names")
    return tuple(arms)


def _parse_training(value: object) -> TrainingConfig:
    record = _mapping(value, "training")
    _keys(
        record,
        required={"pooling_mode", "temporal_kernel_size", "epochs", "batch_size", "precision"},
        name="training",
    )
    pooling = _text(record["pooling_mode"], "training.pooling_mode")
    if pooling not in _PRETRAIN_POOLING_MODES:
        raise ValueError(
            "training.pooling_mode must be one of the supervised pretrain runner modes "
            f"{sorted(_PRETRAIN_POOLING_MODES)}."
        )
    kernel = _integer(record["temporal_kernel_size"], "training.temporal_kernel_size", minimum=3)
    if kernel % 2 == 0:
        raise ValueError("training.temporal_kernel_size must be odd.")
    precision = _text(record["precision"], "training.precision")
    if precision not in _TRAINING_PRECISIONS:
        raise ValueError(f"training.precision must be one of {sorted(_TRAINING_PRECISIONS)}.")
    return TrainingConfig(
        pooling_mode=pooling,
        temporal_kernel_size=kernel,
        epochs=_integer(record["epochs"], "training.epochs", minimum=1),
        batch_size=_integer(record["batch_size"], "training.batch_size", minimum=1),
        precision=precision,
    )


def _parse_statistical_design(
    value: object,
    *,
    root: Path,
    available_arms: frozenset[str],
    test_reps: int,
) -> StatisticalDesignSpec:
    record = _mapping(value, "statistical_design")
    _keys(
        record,
        required={
            "inference_scope",
            "planned_contrasts",
            "evidence_level",
            "bootstrap_iterations",
            "bootstrap_seed",
            "confidence_level",
        },
        optional={"power_plan"},
        name="statistical_design",
    )
    inference_scope = _text(record["inference_scope"], "statistical_design.inference_scope")
    if inference_scope not in _INFERENCE_SCOPES:
        raise ValueError(
            f"statistical_design.inference_scope must be one of {sorted(_INFERENCE_SCOPES)}."
        )
    contrasts_raw = _sequence(record["planned_contrasts"], "statistical_design.planned_contrasts")
    contrasts: list[tuple[str, str]] = []
    for index, raw_pair in enumerate(contrasts_raw):
        pair = _sequence(raw_pair, f"statistical_design.planned_contrasts[{index}]")
        if len(pair) != 2:
            raise ValueError("each planned contrast must contain exactly two arm names.")
        left = _text(pair[0], f"planned contrast {index} left arm")
        right = _text(pair[1], f"planned contrast {index} right arm")
        if left == right:
            raise ValueError("planned contrast arms must be distinct.")
        unknown = sorted({left, right} - available_arms)
        if unknown:
            raise ValueError(f"planned contrast {index} names unknown arms {unknown}.")
        contrasts.append((left, right))
    if not contrasts:
        raise ValueError("statistical_design.planned_contrasts must not be empty.")
    canonical_pairs = [frozenset(pair) for pair in contrasts]
    if len(set(canonical_pairs)) != len(canonical_pairs):
        raise ValueError("planned contrasts must not contain duplicate or reversed pairs.")
    evidence_level = _integer(
        record["evidence_level"],
        "statistical_design.evidence_level",
        minimum=MIN_PROMOTION_HIT_AT_R,
    )
    assert_promotion_evidence_gate(
        test_reps,
        primary_evidence_level=evidence_level,
        name="candidate promotion plan",
    )
    power_plan = (
        _resolve_plan_path(record["power_plan"], "statistical_design.power_plan", root)
        if record.get("power_plan") is not None
        else None
    )
    if inference_scope == "training_procedure" and power_plan is None:
        raise ValueError("training_procedure inference requires statistical_design.power_plan.")
    return StatisticalDesignSpec(
        inference_scope=inference_scope,
        planned_contrasts=tuple(contrasts),
        evidence_level=evidence_level,
        bootstrap_iterations=_integer(
            record["bootstrap_iterations"],
            "statistical_design.bootstrap_iterations",
            minimum=1,
        ),
        bootstrap_seed=_integer(
            record["bootstrap_seed"], "statistical_design.bootstrap_seed", minimum=0
        ),
        confidence_level=_number(record["confidence_level"], "statistical_design.confidence_level"),
        power_plan=power_plan,
    )


def load_plan(path: str | Path, *, root: str | Path | None = None) -> PromotionPlan:
    """Load and structurally validate a versioned promotion plan."""

    plan_path = Path(path).resolve()
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    record = _mapping(raw, "plan")
    _keys(
        record,
        required={
            "schema",
            "stage",
            "source_snapshot_manifest",
            "target_cache",
            "target_subjects",
            "source_arms",
            "training_replicates",
            "evaluation_arms",
            "training",
            "calibration_selections",
            "test_reps",
            "identity_exclusion_policy",
            "output_root",
            "statistical_design",
        },
        name="plan",
    )
    if record["schema"] != PLAN_SCHEMA:
        raise ValueError(f"plan.schema must be {PLAN_SCHEMA!r}.")
    if record["stage"] != "development":
        raise ValueError("candidate promotion plans must declare stage='development'.")
    resolution_root = Path(root).resolve() if root is not None else plan_path.parent.resolve()
    targets = _parse_targets(record["target_subjects"])
    source_arms = _parse_source_arms(record["source_arms"], resolution_root)
    replicates = _parse_replicates(record["training_replicates"])
    evaluation_arms = _parse_evaluation_arms(record["evaluation_arms"])
    expanded_arm_names = [
        evaluation_arm_name(source_arm.name, evaluation_arm.name)
        for source_arm in source_arms
        for evaluation_arm in evaluation_arms
    ]
    _assert_unique(expanded_arm_names, "expanded source/evaluation arm names")
    available_arms = frozenset(expanded_arm_names)
    test_reps = _integer(record["test_reps"], "test_reps", minimum=MIN_PROMOTION_HIT_AT_R)
    assert_promotion_evidence_gate(test_reps, name="candidate promotion plan")
    identity_policy = _text(record["identity_exclusion_policy"], "identity_exclusion_policy")
    if identity_policy not in _IDENTITY_POLICIES:
        raise ValueError(f"identity_exclusion_policy must be one of {sorted(_IDENTITY_POLICIES)}.")
    output_root = _resolve_plan_path(record["output_root"], "output_root", resolution_root)
    if root is not None:
        try:
            output_root.relative_to(resolution_root)
        except ValueError as error:
            raise ValueError("output_root escapes the explicit --root boundary.") from error
    return PromotionPlan(
        path=plan_path,
        resolution_root=resolution_root,
        raw_record=record,
        digest=semantic_sha256(record),
        source_snapshot_manifest=_resolve_plan_path(
            record["source_snapshot_manifest"],
            "source_snapshot_manifest",
            resolution_root,
        ),
        target_cache=_resolve_plan_path(record["target_cache"], "target_cache", resolution_root),
        target_subjects=targets,
        source_arms=source_arms,
        training_replicates=replicates,
        evaluation_arms=evaluation_arms,
        training=_parse_training(record["training"]),
        calibration_selections=_integer(
            record["calibration_selections"], "calibration_selections", minimum=1
        ),
        test_reps=test_reps,
        identity_exclusion_policy=identity_policy,
        output_root=output_root,
        statistical_design=_parse_statistical_design(
            record["statistical_design"],
            root=resolution_root,
            available_arms=available_arms,
            test_reps=test_reps,
        ),
    )


def _within_output(path: Path, output_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(output_root.resolve())
    except ValueError as error:
        raise ValueError(f"derived output escapes output_root: {resolved}") from error
    return resolved


def build_dag(plan: PromotionPlan, *, device: str = "cuda") -> PromotionDag:
    """Materialize the complete sequential subprocess DAG without writing files."""

    device_name = _text(device, "device")
    output_root = plan.output_root.resolve()
    scope_paths = {
        target.partition_key: _within_output(
            output_root / "subject_scopes" / f"{target.partition_key}.json", output_root
        )
        for target in plan.target_subjects
    }
    tasks: list[MatrixTask] = []
    result_tasks: list[MatrixTask] = []
    for arm in plan.source_arms:
        for replicate in plan.training_replicates:
            for target in plan.target_subjects:
                checkpoint_id = f"checkpoint/{arm.name}/{replicate.key}/{target.partition_key}"
                checkpoint_path = _within_output(
                    output_root
                    / "checkpoints"
                    / arm.name
                    / replicate.key
                    / f"{target.partition_key}.pt",
                    output_root,
                )
                checkpoint_argv = [
                    sys.executable,
                    str((ROOT / "experiments" / "run_pretrain_supervised.py").resolve()),
                    "--source-cache",
                    str(arm.cache),
                    "--source-snapshot-manifest",
                    str(plan.source_snapshot_manifest),
                    "--holdout-subjects",
                    target.source_holdout_subject,
                    "--pooling-mode",
                    plan.training.pooling_mode,
                    "--temporal-kernel-size",
                    str(plan.training.temporal_kernel_size),
                    "--epochs",
                    str(plan.training.epochs),
                    "--batch-size",
                    str(plan.training.batch_size),
                    "--precision",
                    plan.training.precision,
                    "--seed",
                    str(replicate.seed),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--device",
                    device_name,
                ]
                for domain, mass in arm.source_domain_mass:
                    checkpoint_argv.extend(
                        ["--source-domain-mass", f"{domain}={mass!r}"]
                    )
                if arm.source_domain_mass:
                    assert arm.risk_within_domain_unit is not None
                    checkpoint_argv.extend(
                        [
                            "--risk-within-domain-unit",
                            arm.risk_within_domain_unit,
                        ]
                    )
                if arm.selection_domain is not None:
                    checkpoint_argv.extend(
                        ["--selection-domain", arm.selection_domain]
                    )
                checkpoint_task = MatrixTask(
                    task_id=checkpoint_id,
                    kind="checkpoint",
                    argv=tuple(checkpoint_argv),
                    output=checkpoint_path,
                    dependencies=(),
                )
                tasks.append(checkpoint_task)
                for evaluation_arm in plan.evaluation_arms:
                    arm_name = evaluation_arm_name(arm.name, evaluation_arm.name)
                    result_id = f"result/{arm_name}/{replicate.key}/{target.partition_key}"
                    result_path = _within_output(
                        output_root
                        / "results"
                        / arm_name
                        / replicate.key
                        / f"{target.partition_key}.json",
                        output_root,
                    )
                    result_argv = [
                        sys.executable,
                        str((ROOT / "experiments" / "run_candidate_cross_decision.py").resolve()),
                        "--dataset-cache",
                        str(plan.target_cache),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--arm-name",
                        arm_name,
                        "--training-replicate-key",
                        replicate.key,
                        "--partition-key",
                        target.partition_key,
                        "--source-snapshot-manifest",
                        str(plan.source_snapshot_manifest),
                        "--identity-exclusion-policy",
                        plan.identity_exclusion_policy,
                        "--calibration-selections",
                        str(plan.calibration_selections),
                        "--test-reps",
                        str(plan.test_reps),
                        "--head",
                        evaluation_arm.head,
                        "--normalization",
                        evaluation_arm.normalization,
                    ]
                    if evaluation_arm.head not in {"auto", "zero_shot"}:
                        assert evaluation_arm.epochs is not None
                        assert evaluation_arm.batch_size is not None
                        assert evaluation_arm.lr is not None
                        result_argv.extend(
                            [
                                "--epoch-selection",
                                evaluation_arm.epoch_selection,
                                "--target-stat-weight",
                                str(evaluation_arm.target_stat_weight),
                                "--epochs",
                                str(evaluation_arm.epochs),
                                "--batch-size",
                                str(evaluation_arm.batch_size),
                                "--lr",
                                str(evaluation_arm.lr),
                                (
                                    "--fold-local-qc"
                                    if evaluation_arm.fold_local_qc
                                    else "--no-fold-local-qc"
                                ),
                            ]
                        )
                        if evaluation_arm.adapt_batchnorm:
                            result_argv.append("--adapt-batchnorm")
                    result_argv.extend(
                        [
                            "--seed",
                            str(replicate.seed),
                            "--target-subjects-file",
                            str(scope_paths[target.partition_key]),
                            "--device",
                            device_name,
                            "--output",
                            str(result_path),
                        ]
                    )
                    result_task = MatrixTask(
                        task_id=result_id,
                        kind="result",
                        argv=tuple(result_argv),
                        output=result_path,
                        dependencies=(checkpoint_id,),
                    )
                    tasks.append(result_task)
                    result_tasks.append(result_task)

    manifest_path = _within_output(output_root / "manifest.json", output_root)
    manifest_argv = [
        sys.executable,
        str((ROOT / "experiments" / "build_candidate_cross_decision_manifest.py").resolve()),
    ]
    for result_task in result_tasks:
        manifest_argv.extend(["--result", str(result_task.output)])
    manifest_argv.extend(
        [
            "--source-snapshot-manifest",
            str(plan.source_snapshot_manifest),
            "--inference-scope",
            plan.statistical_design.inference_scope,
        ]
    )
    for left, right in plan.statistical_design.planned_contrasts:
        manifest_argv.extend(["--planned-contrast", left, right])
    manifest_argv.extend(
        [
            "--evidence-level",
            str(plan.statistical_design.evidence_level),
            "--bootstrap-iterations",
            str(plan.statistical_design.bootstrap_iterations),
            "--bootstrap-seed",
            str(plan.statistical_design.bootstrap_seed),
            "--confidence-level",
            str(plan.statistical_design.confidence_level),
        ]
    )
    if plan.statistical_design.power_plan is not None:
        manifest_argv.extend(["--power-plan", str(plan.statistical_design.power_plan)])
    manifest_argv.extend(["--output", str(manifest_path)])
    manifest_task = MatrixTask(
        task_id="manifest",
        kind="manifest",
        argv=tuple(manifest_argv),
        output=manifest_path,
        dependencies=tuple(task.task_id for task in result_tasks),
    )
    tasks.append(manifest_task)

    analysis_path = _within_output(output_root / "analysis.json", output_root)
    tasks.append(
        MatrixTask(
            task_id="analysis",
            kind="analysis",
            argv=(
                sys.executable,
                str((ROOT / "experiments" / "analyze_candidate_cross_decision.py").resolve()),
                "--manifest",
                str(manifest_path),
                "--output",
                str(analysis_path),
            ),
            output=analysis_path,
            dependencies=(manifest_task.task_id,),
        )
    )
    task_ids = [task.task_id for task in tasks]
    _assert_unique(task_ids, "DAG task ids")
    output_paths = [task.output for task in tasks]
    _assert_unique(output_paths, "DAG task outputs")
    known = set(task_ids)
    completed: set[str] = set()
    for task in tasks:
        if not set(task.dependencies) <= completed:
            missing = sorted(set(task.dependencies) - completed)
            raise ValueError(f"task {task.task_id!r} has non-topological dependencies {missing}.")
        completed.add(task.task_id)
    if completed != known:
        raise ValueError("DAG construction lost one or more tasks.")
    return PromotionDag(
        plan=plan,
        device=device_name,
        tasks=tuple(tasks),
        subject_scope_paths=scope_paths,
    )


def _portable_path(path: Path, dag: PromotionDag) -> str:
    resolved = path.resolve()
    roots = (
        ("$OUTPUT_ROOT", dag.plan.output_root),
        ("$PLAN_ROOT", dag.plan.resolution_root),
        ("$REPO_ROOT", ROOT),
    )
    for label, root in roots:
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        return label if not relative else f"{label}/{relative}"
    return f"$EXTERNAL/{resolved.name}"


def _journal_argv(argv: Sequence[str], dag: PromotionDag) -> list[str]:
    sensitive_flags = {"--password", "--secret", "--token", "--api-key"}
    result: list[str] = []
    redact_next = False
    for index, raw in enumerate(argv):
        if index == 0:
            result.append("$PYTHON")
            continue
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if raw in sensitive_flags:
            result.append(raw)
            redact_next = True
            continue
        path = Path(raw)
        if path.is_absolute():
            result.append(_portable_path(path, dag))
        elif re.search(r"(?i)(password|secret|token|api[_-]?key)=", raw):
            result.append("<redacted>")
        else:
            result.append(raw)
    return result


def dag_record(
    dag: PromotionDag, *, resolved_inputs: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    counts = {
        kind: sum(task.kind == kind for task in dag.tasks)
        for kind in ("checkpoint", "result", "manifest", "analysis")
    }
    record = {
        "schema": DRY_RUN_SCHEMA,
        "plan": _portable_path(dag.plan.path, dag),
        "plan_digest": dag.plan.digest,
        "stage": "development",
        "device": dag.device,
        "output_root": "$OUTPUT_ROOT",
        "task_count": len(dag.tasks),
        "task_counts": counts,
        "tasks": [
            {
                "task_id": task.task_id,
                "kind": task.kind,
                "dependencies": list(task.dependencies),
                "argv": _journal_argv(task.argv, dag),
                "output": _portable_path(task.output, dag),
            }
            for task in dag.tasks
        ],
    }
    if resolved_inputs is not None:
        record["resolved_inputs"] = dict(resolved_inputs)
    return record


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01 * (attempt + 1))
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _orchestrator_lock_path(dag: PromotionDag) -> Path:
    return _within_output(
        dag.plan.output_root / ".promotion.orchestrator.lock.json",
        dag.plan.output_root,
    )


def _acquire_orchestrator_lock(dag: PromotionDag, dag_digest: str) -> dict[str, Any]:
    lock_path = _orchestrator_lock_path(dag)
    token = uuid4().hex
    record = {
        "schema": ORCHESTRATOR_LOCK_SCHEMA,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "dag_digest": _validate_sha(dag_digest, "orchestrator lock DAG digest"),
        "created_at": _now(),
        "token": token,
    }
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        try:
            owner = _read_json_mapping(lock_path, "promotion orchestrator lock")
        except (OSError, ValueError, json.JSONDecodeError) as read_error:
            owner = {"unreadable": f"{type(read_error).__name__}: {read_error}"}
        raise FileExistsError(
            "promotion output root is locked by another or stale orchestrator; "
            "locks are never stolen automatically and require operator inspection: "
            f"{owner}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, ensure_ascii=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    return record


def _release_orchestrator_lock(dag: PromotionDag, owner: Mapping[str, Any]) -> None:
    lock_path = _orchestrator_lock_path(dag)
    current = _read_json_mapping(lock_path, "promotion orchestrator lock")
    identity_fields = ("schema", "pid", "host", "dag_digest", "token")
    if any(current.get(field) != owner.get(field) for field in identity_fields):
        raise RuntimeError("promotion orchestrator lock ownership changed during execution.")
    lock_path.unlink()


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, name)


def _validate_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _embedded_artifact_record(task: MatrixTask) -> dict[str, Any]:
    """Validate the minimum embedded schema and semantic contract linkage."""

    if task.kind == "checkpoint":
        payload = load_checkpoint_payload(task.output)
        training = checkpoint_training_contract(payload)
        return {
            "schema": CHECKPOINT_SCHEMA,
            "training_contract_digest": training.digest(),
        }
    record = _read_json_mapping(task.output, f"{task.kind} output")
    if task.kind == "result":
        if record.get("schema") != RESULT_SCHEMA or record.get("run_status") not in {
            "completed",
            "completed_with_selection_failures",
        }:
            raise ValueError("result output is not a completed current-schema artifact.")
        evaluation_record = _mapping(
            record.get("evaluation_contract"), "result evaluation_contract"
        )
        evaluation_digest = _validate_sha(
            record.get("evaluation_contract_digest"), "evaluation_contract_digest"
        )
        assert_contract_digest(
            evaluation_record, evaluation_digest, name="result evaluation_contract"
        )
        evaluation = EvaluationRunContract(**evaluation_record)
        if evaluation.digest() != evaluation_digest:
            raise ValueError("result evaluation contract digest disagrees after parsing.")
        decision_record = _mapping(record.get("decision_plan"), "result decision_plan")
        decision_digest = _validate_sha(record.get("decision_plan_digest"), "decision_plan_digest")
        assert_contract_digest(decision_record, decision_digest, name="result decision plan")
        return {
            "schema": RESULT_SCHEMA,
            "evaluation_contract_digest": evaluation_digest,
            "decision_plan_digest": decision_digest,
        }
    if task.kind == "manifest":
        if record.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("manifest output has an unsupported schema.")
        design_record = _mapping(record.get("statistical_design"), "manifest statistical_design")
        design_digest = _validate_sha(
            record.get("statistical_design_digest"), "statistical_design_digest"
        )
        assert_contract_digest(design_record, design_digest, name="manifest statistical design")
        design = StatisticalDesignContract(**design_record)
        if design.digest() != design_digest:
            raise ValueError("manifest statistical design digest disagrees after parsing.")
        return {"schema": MANIFEST_SCHEMA, "statistical_design_digest": design_digest}
    if record.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("analysis output has an unsupported schema.")
    design_digest = _validate_sha(
        record.get("statistical_design_digest"), "analysis statistical_design_digest"
    )
    manifest_digest = _validate_sha(record.get("manifest_sha256"), "analysis manifest_sha256")
    return {
        "schema": ANALYSIS_SCHEMA,
        "statistical_design_digest": design_digest,
        "manifest_sha256": manifest_digest,
    }


def _artifact_attestation(task: MatrixTask) -> dict[str, Any]:
    if not task.output.is_file():
        raise ValueError(f"task {task.task_id!r} did not produce {task.output}.")
    embedded = _embedded_artifact_record(task)
    return {
        "path": task.output.name,
        "sha256": _sha256_file(task.output),
        "byte_size": task.output.stat().st_size,
        "embedded": embedded,
    }


def _load_attested_cache_input(
    cache_path: Path, *, name: str, dag: PromotionDag
) -> _AttestedCacheInput:
    dataset = load_epoch_dataset(
        cache_path,
        require_labels=True,
        validation="attested",
    )
    attestation = loaded_epoch_cache_attestation(dataset)
    cache_sha256 = _validate_sha(attestation.get("sha256"), f"{name} cache sha256")
    decoded_dataset_sha256 = _validate_sha(
        attestation.get("decoded_dataset_sha256"),
        f"{name} decoded dataset sha256",
    )
    if dataset.identity_table is None:
        raise ValueError(f"{name} lacks a participant identity table.")
    source_reference = dataset.provenance.get("source_reference")
    if not isinstance(source_reference, str) or not source_reference.strip():
        raise ValueError(f"{name} lacks a non-empty source_reference contract.")
    positions = np.asarray(dataset.channel_positions_m, dtype=np.float64)
    domains = (
        tuple(sorted(set(source_domain_ids(dataset).tolist())))
        if dataset.provenance.get("source_domain_axis") is not None
        else ()
    )
    resolved_record = {
        "path": _portable_path(cache_path, dag),
        "cache_sha256": cache_sha256,
        "decoded_dataset_sha256": decoded_dataset_sha256,
        "verified_cache_attestation": dict(attestation),
        "identity_digest": dataset.identity_table.digest(),
        "channel_contract_digest": semantic_sha256(
            {
                "channel_names": list(dataset.channel_names),
                "channel_positions_m": positions.tolist(),
                "channel_mask": np.asarray(dataset.channel_mask, dtype=bool).tolist(),
            }
        ),
        "preprocessing_digest": semantic_sha256(asdict(dataset.preprocessing)),
        "source_reference": source_reference.strip(),
        "source_domains": list(domains),
    }
    return _AttestedCacheInput(
        identity_table=dataset.identity_table,
        channel_names=tuple(dataset.channel_names),
        channel_positions_m=tuple(tuple(float(value) for value in row) for row in positions),
        channel_mask=tuple(bool(value) for value in dataset.channel_mask),
        preprocessing=asdict(dataset.preprocessing),
        source_reference=source_reference.strip(),
        source_domains=domains,
        resolved_record=resolved_record,
    )


def _assert_cache_compatibility(
    target: _AttestedCacheInput,
    source: _AttestedCacheInput,
    *,
    source_arm: str,
) -> None:
    mismatched: list[str] = []
    if source.channel_names != target.channel_names:
        mismatched.append("channel_names")
    if source.channel_positions_m != target.channel_positions_m:
        mismatched.append("channel_positions_m")
    if source.channel_mask != target.channel_mask:
        mismatched.append("channel_mask")
    if source.preprocessing != target.preprocessing:
        mismatched.append("preprocessing")
    if source.source_reference.casefold() != target.source_reference.casefold():
        mismatched.append("source_reference")
    if mismatched:
        raise ValueError(
            f"source arm {source_arm!r} cache contract disagrees with target cache: {mismatched}."
        )


def _verify_inputs(dag: PromotionDag) -> dict[str, Any]:
    plan = dag.plan
    inputs = {
        "source_snapshot_manifest": plan.source_snapshot_manifest,
        "target_cache": plan.target_cache,
        **{f"source_arm[{arm.name}].cache": arm.cache for arm in plan.source_arms},
    }
    if plan.statistical_design.power_plan is not None:
        inputs["statistical_design.power_plan"] = plan.statistical_design.power_plan
    missing = [f"{name}: {path}" for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("promotion plan inputs do not exist:\n" + "\n".join(missing))
    source_snapshot_sha256 = source_snapshot_sha256_from_archive_manifest(
        plan.source_snapshot_manifest
    )
    unique_cache_paths: dict[Path, str] = {
        plan.target_cache.resolve(): "target cache",
    }
    for arm in plan.source_arms:
        unique_cache_paths.setdefault(
            arm.cache.resolve(),
            f"source cache for arm {arm.name!r}",
        )
    cache_tasks = tuple(unique_cache_paths.items())
    worker_plan = resolve_cpu_worker_plan(len(cache_tasks))

    def load_cache(task: tuple[Path, str]) -> tuple[Path, _AttestedCacheInput]:
        path, name = task
        return path, _load_attested_cache_input(path, name=name, dag=dag)

    with cpu_thread_budget(worker_plan.threads_per_worker):
        if worker_plan.effective_workers == 1:
            loaded = tuple(load_cache(task) for task in cache_tasks)
        else:
            with ThreadPoolExecutor(
                max_workers=worker_plan.effective_workers,
                thread_name_prefix="promotion-preflight",
            ) as executor:
                loaded = tuple(executor.map(load_cache, cache_tasks))
    cache_by_path = dict(loaded)
    target = cache_by_path[plan.target_cache.resolve()]
    source_by_arm: dict[str, _AttestedCacheInput] = {}
    for arm in plan.source_arms:
        source = cache_by_path[arm.cache.resolve()]
        _assert_cache_compatibility(target, source, source_arm=arm.name)
        planned_domains = {domain for domain, _ in arm.source_domain_mass}
        observed_domains = set(source.source_domains)
        if planned_domains and planned_domains != observed_domains:
            raise ValueError(
                f"source arm {arm.name!r} domain risk disagrees with cache: "
                f"planned={sorted(planned_domains)}, observed={sorted(observed_domains)}."
            )
        if len(observed_domains) > 1 and arm.selection_domain is None:
            raise ValueError(
                f"source arm {arm.name!r} multi-domain cache requires selection_domain."
            )
        if arm.selection_domain is not None and arm.selection_domain not in observed_domains:
            raise ValueError(
                f"source arm {arm.name!r} selection_domain is absent from cache."
            )
        source_by_arm[arm.name] = source

    for target_spec in plan.target_subjects:
        target_key = target.identity_table.record_for(target_spec.target_subject).authority_key(
            plan.identity_exclusion_policy
        )
        for arm in plan.source_arms:
            source_key = (
                source_by_arm[arm.name]
                .identity_table.record_for(target_spec.source_holdout_subject)
                .authority_key(plan.identity_exclusion_policy)
            )
            if source_key != target_key:
                raise ValueError(
                    f"target {target_spec.target_subject!r} and source holdout "
                    f"{target_spec.source_holdout_subject!r} have different "
                    f"{plan.identity_exclusion_policy!r} authority keys in source arm "
                    f"{arm.name!r}."
                )

    resolved: dict[str, Any] = {
        "source_snapshot": {
            "manifest_path": _portable_path(plan.source_snapshot_manifest, dag),
            "manifest_sha256": _sha256_file(plan.source_snapshot_manifest),
            "archive_sha256": source_snapshot_sha256,
        },
        "target_cache": dict(target.resolved_record),
        "source_caches": {
            arm.name: dict(source_by_arm[arm.name].resolved_record) for arm in plan.source_arms
        },
        "preflight_loading": worker_plan.record(),
        "identity_exclusion_policy": plan.identity_exclusion_policy,
        "target_holdout_authority_keys": {
            target_spec.partition_key: target.identity_table.record_for(
                target_spec.target_subject
            ).authority_key(plan.identity_exclusion_policy)
            for target_spec in plan.target_subjects
        },
        "power_plan": (
            None
            if plan.statistical_design.power_plan is None
            else {
                "path": _portable_path(plan.statistical_design.power_plan, dag),
                "sha256": _sha256_file(plan.statistical_design.power_plan),
            }
        ),
    }
    resolved["digest"] = semantic_sha256(resolved)
    return resolved


def _journal_path(dag: PromotionDag) -> Path:
    return _within_output(dag.plan.output_root / "promotion.journal.json", dag.plan.output_root)


def _dag_digest(dag: PromotionDag, resolved_inputs: Mapping[str, Any]) -> str:
    record = dag_record(dag, resolved_inputs=resolved_inputs)
    record.pop("schema", None)
    return semantic_sha256(record)


def _initial_state(dag: PromotionDag, resolved_inputs: Mapping[str, Any]) -> dict[str, Any]:
    timestamp = _now()
    return {
        "schema": JOURNAL_SCHEMA,
        "plan": {
            "path": _portable_path(dag.plan.path, dag),
            "sha256": dag.plan.digest,
            "stage": "development",
        },
        "dag_digest": _dag_digest(dag, resolved_inputs),
        "resolved_inputs": dict(resolved_inputs),
        "status": "running",
        "created_at": timestamp,
        "updated_at": timestamp,
        "finished_at": None,
        "derived_inputs": {},
        "dispatch_sequence": [],
        "tasks": {
            task.task_id: {
                "kind": task.kind,
                "dependencies": list(task.dependencies),
                "argv": _journal_argv(task.argv, dag),
                "output_path": _portable_path(task.output, dag),
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "attempts": [],
            }
            for task in dag.tasks
        },
    }


def _load_or_create_state(
    dag: PromotionDag,
    *,
    resume: bool,
    resolved_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    journal = _journal_path(dag)
    known_outputs = [task.output for task in dag.tasks]
    known_outputs.extend(dag.subject_scope_paths.values())
    known_outputs.append(dag.plan.output_root / "logs")
    if journal.exists():
        if not resume:
            raise FileExistsError(
                f"journal already exists; use --resume after inspecting it: {journal}"
            )
        state = _read_json_mapping(journal, "promotion journal")
        if state.get("schema") != JOURNAL_SCHEMA:
            raise ValueError("promotion journal has an unsupported schema.")
        plan_record = _mapping(state.get("plan"), "promotion journal plan")
        if plan_record.get("sha256") != dag.plan.digest:
            raise ValueError("promotion journal belongs to a different plan digest.")
        if state.get("resolved_inputs") != resolved_inputs:
            raise ValueError(
                "promotion journal inputs changed; use a new output root rather than "
                "resuming artifacts from different inputs."
            )
        if state.get("dag_digest") != _dag_digest(dag, resolved_inputs):
            raise ValueError("promotion journal belongs to a different resolved DAG.")
        state_tasks = _mapping(state.get("tasks"), "promotion journal tasks")
        if set(state_tasks) != {task.task_id for task in dag.tasks}:
            raise ValueError("promotion journal task set disagrees with the current DAG.")
        for task in dag.tasks:
            task_record = _mapping(
                state_tasks[task.task_id], f"promotion journal task {task.task_id}"
            )
            expected_static = {
                "kind": task.kind,
                "dependencies": list(task.dependencies),
                "argv": _journal_argv(task.argv, dag),
                "output_path": _portable_path(task.output, dag),
            }
            mismatched = [
                key for key, expected in expected_static.items() if task_record.get(key) != expected
            ]
            if mismatched:
                raise ValueError(
                    f"promotion journal task {task.task_id!r} changed static fields {mismatched}."
                )
            attempts = task_record.setdefault("attempts", [])
            if not isinstance(attempts, list) or not all(
                isinstance(attempt, Mapping) for attempt in attempts
            ):
                raise ValueError(f"promotion journal task {task.task_id!r} has invalid attempts.")
        dispatch_sequence = state.setdefault("dispatch_sequence", [])
        if not isinstance(dispatch_sequence, list) or not all(
            isinstance(task_id, str) and task_id in state_tasks for task_id in dispatch_sequence
        ):
            raise ValueError("promotion journal dispatch_sequence is invalid.")
        state["status"] = "running"
        state["finished_at"] = None
        state["updated_at"] = _now()
        _atomic_write_json(journal, state)
        return state
    collisions = [path for path in known_outputs if path.exists()]
    if collisions:
        rendered = "\n".join(str(path) for path in collisions)
        raise FileExistsError(
            "unattested promotion outputs already exist; mere existence is not resume "
            f"evidence:\n{rendered}"
        )
    if resume:
        # Starting a clean plan with --resume is harmless, but never adopts orphan files.
        pass
    state = _initial_state(dag, resolved_inputs)
    _atomic_write_json(journal, state)
    return state


def _materialize_subject_scopes(dag: PromotionDag, state: dict[str, Any]) -> None:
    records: dict[str, Any] = {}
    targets = {target.partition_key: target for target in dag.plan.target_subjects}
    for partition_key, path in dag.subject_scope_paths.items():
        _within_output(path, dag.plan.output_root)
        _atomic_write_json(path, [targets[partition_key].target_subject])
        records[partition_key] = {
            "path": _portable_path(path, dag),
            "sha256": _sha256_file(path),
            "byte_size": path.stat().st_size,
            "target_subject": targets[partition_key].target_subject,
        }
    state["derived_inputs"] = {"target_subject_scopes": records}
    state["updated_at"] = _now()
    _atomic_write_json(_journal_path(dag), state)


def _dependency_digests(task: MatrixTask, state_tasks: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for dependency in task.dependencies:
        record = _mapping(state_tasks.get(dependency), f"dependency {dependency}")
        output = _mapping(record.get("output"), f"dependency {dependency} output")
        result[dependency] = _validate_sha(output.get("sha256"), f"dependency {dependency} sha256")
    return result


def _resume_validation(
    dag: PromotionDag,
    task: MatrixTask,
    record: Mapping[str, Any],
    state_tasks: Mapping[str, Any],
) -> tuple[bool, str | None]:
    if record.get("status") != "completed":
        return False, "journal task is not completed"
    output = record.get("output")
    if not isinstance(output, Mapping):
        return False, "journal task has no output attestation"
    if not task.output.is_file():
        return False, "output file is missing"
    try:
        byte_size = _integer(output.get("byte_size"), "journal output byte_size", minimum=1)
        digest = _validate_sha(output.get("sha256"), "journal output sha256")
        if task.output.stat().st_size != byte_size:
            return False, "output byte_size changed"
        if _sha256_file(task.output) != digest:
            return False, "output SHA-256 changed"
        embedded = _embedded_artifact_record(task)
        if embedded != output.get("embedded"):
            return False, "embedded schema or contract digest changed"
        attempts = record.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            return False, "completed task has no subprocess attempt provenance"
        attempt = attempts[-1]
        if not isinstance(attempt, Mapping) or attempt.get("status") != "completed":
            return False, "latest subprocess attempt is not completed"
        attempt_number = _integer(
            attempt.get("attempt"),
            "journal attempt number",
            minimum=1,
        )
        stdout_path, stderr_path = _task_log_paths(
            dag,
            task,
            attempt_number=attempt_number,
        )
        logs = _mapping(attempt.get("logs"), "journal attempt logs")
        for stream_name, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            log = _mapping(logs.get(stream_name), f"journal {stream_name} log")
            if log.get("path") != _portable_path(path, dag):
                return False, f"{stream_name} log path changed"
            byte_size = _integer(
                log.get("byte_size"),
                f"journal {stream_name} log byte_size",
                minimum=0,
            )
            digest = _validate_sha(
                log.get("sha256"),
                f"journal {stream_name} log sha256",
            )
            if not path.is_file() or path.stat().st_size != byte_size:
                return False, f"{stream_name} log is missing or changed size"
            if _sha256_file(path) != digest:
                return False, f"{stream_name} log SHA-256 changed"
        current_dependencies = _dependency_digests(task, state_tasks)
        if current_dependencies != record.get("dependency_outputs", {}):
            return False, "dependency output digest changed"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"resume validation failed: {type(error).__name__}: {error}"
    return True, None


def _error_record(error: BaseException) -> dict[str, Any]:
    message = (
        f"subprocess exited with return code {error.returncode}"
        if isinstance(error, subprocess.CalledProcessError)
        else str(error)
    )
    record: dict[str, Any] = {
        "type": type(error).__name__,
        "message": message,
    }
    if isinstance(error, subprocess.CalledProcessError):
        record["returncode"] = error.returncode
    return record


def _validate_max_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_workers must be a positive integer.")
    return value


def _task_log_paths(
    dag: PromotionDag,
    task: MatrixTask,
    *,
    attempt_number: int,
) -> tuple[Path, Path]:
    task_index = next(
        index for index, candidate in enumerate(dag.tasks) if candidate.task_id == task.task_id
    )
    task_digest = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:12]
    stem = f"{task_index:04d}-{task.kind}-{task_digest}.attempt-{attempt_number:03d}"
    log_root = _within_output(dag.plan.output_root / "logs", dag.plan.output_root)
    return (
        _within_output(log_root / f"{stem}.stdout.log", dag.plan.output_root),
        _within_output(log_root / f"{stem}.stderr.log", dag.plan.output_root),
    )


def _run_subprocess_invocation(
    invocation: _SubprocessInvocation,
    runner: Callable[..., Any],
) -> _SubprocessOutcome:
    """Worker boundary: invoke one subprocess and write only its private logs."""

    try:
        with (
            invocation.stdout_path.open("xb") as stdout,
            invocation.stderr_path.open("xb") as stderr,
        ):
            runner(
                list(invocation.task.argv),
                check=True,
                shell=False,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
            )
    except BaseException as error:
        return _SubprocessOutcome(error=error)
    return _SubprocessOutcome(error=None)


def _log_attestation(path: Path, dag: PromotionDag) -> dict[str, Any]:
    _within_output(path, dag.plan.output_root)
    if not path.is_file():
        raise FileNotFoundError(f"subprocess log is missing: {path.name}")
    return {
        "path": _portable_path(path, dag),
        "sha256": _sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _prepare_execution_records(
    dag: PromotionDag,
    state: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, dict[str, Any]]:
    raw_tasks = state.get("tasks")
    if not isinstance(raw_tasks, dict):
        raise ValueError("promotion journal tasks must be a mutable JSON mapping.")
    state_tasks: dict[str, dict[str, Any]] = {}
    timestamp = _now()
    for task in dag.tasks:
        raw_record = raw_tasks.get(task.task_id)
        if not isinstance(raw_record, dict):
            raise ValueError(f"journal task {task.task_id} must be a mutable JSON mapping.")
        record = raw_record
        attempts = record.setdefault("attempts", [])
        if not isinstance(attempts, list):
            raise ValueError(f"journal task {task.task_id} attempts must be a list.")
        orphan_reason: str | None = None
        if resume and record.get("status") == "running":
            orphan_reason = "previous orchestrator ended while task was running"
            if (
                attempts
                and isinstance(attempts[-1], dict)
                and attempts[-1].get("status") == "running"
            ):
                attempts[-1]["status"] = "orphaned"
                attempts[-1]["finished_at"] = timestamp
                attempts[-1]["orphaned_reason"] = orphan_reason
            else:
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "status": "orphaned",
                        "started_at": record.get("started_at"),
                        "finished_at": timestamp,
                        "orphaned_reason": orphan_reason,
                    }
                )
            record["status"] = "pending"
        can_skip, reason = (
            _resume_validation(dag, task, record, raw_tasks)
            if resume and orphan_reason is None
            else (False, orphan_reason)
        )
        if can_skip:
            record["resume_validation"] = {
                "status": "skipped_verified",
                "checked_at": timestamp,
            }
        else:
            if resume:
                record["resume_validation"] = {
                    "status": "rerun",
                    "reason": reason,
                    "checked_at": timestamp,
                }
            else:
                record.pop("resume_validation", None)
            record["status"] = "pending"
            record["started_at"] = None
            record["finished_at"] = None
            record.pop("error", None)
            record.pop("output", None)
            record.pop("dependency_outputs", None)
        state_tasks[task.task_id] = record
    state["updated_at"] = timestamp
    _atomic_write_json(_journal_path(dag), state)
    return state_tasks


def _dispatch_invocation(
    dag: PromotionDag,
    task: MatrixTask,
    record: dict[str, Any],
    state_tasks: Mapping[str, Any],
) -> _SubprocessInvocation:
    attempts = record.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError(f"journal task {task.task_id} attempts must be a list.")
    attempt_number = len(attempts) + 1
    stdout_path, stderr_path = _task_log_paths(
        dag,
        task,
        attempt_number=attempt_number,
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(f"subprocess attempt logs already exist for task {task.task_id!r}.")
    task.output.parent.mkdir(parents=True, exist_ok=True)
    dependency_outputs = _dependency_digests(task, state_tasks)
    started_at = _now()
    attempt = {
        "attempt": attempt_number,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "stdout_path": _portable_path(stdout_path, dag),
        "stderr_path": _portable_path(stderr_path, dag),
    }
    attempts.append(attempt)
    record["status"] = "running"
    record["started_at"] = started_at
    record["finished_at"] = None
    record.pop("error", None)
    record.pop("output", None)
    record.pop("dependency_outputs", None)
    return _SubprocessInvocation(
        task=task,
        dependency_outputs=dependency_outputs,
        attempt_index=len(attempts) - 1,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _invocation_attempt(
    record: dict[str, Any], invocation: _SubprocessInvocation
) -> dict[str, Any]:
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or invocation.attempt_index >= len(attempts):
        raise ValueError(f"journal task {invocation.task.task_id} lost its running attempt.")
    attempt = attempts[invocation.attempt_index]
    if not isinstance(attempt, dict):
        raise ValueError(f"journal task {invocation.task.task_id} attempt is invalid.")
    return attempt


def _complete_invocation(
    dag: PromotionDag,
    invocation: _SubprocessInvocation,
    outcome: _SubprocessOutcome,
    state_tasks: Mapping[str, Any],
) -> BaseException | None:
    record_value = state_tasks.get(invocation.task.task_id)
    if not isinstance(record_value, dict):
        raise ValueError(f"journal task {invocation.task.task_id} is invalid.")
    record = record_value
    attempt = _invocation_attempt(record, invocation)
    finished_at = _now()
    error = outcome.error
    try:
        logs = {
            "stdout": _log_attestation(invocation.stdout_path, dag),
            "stderr": _log_attestation(invocation.stderr_path, dag),
        }
        if error is None:
            current_dependencies = _dependency_digests(invocation.task, state_tasks)
            if current_dependencies != dict(invocation.dependency_outputs):
                raise RuntimeError("dependency output digests changed while a task was running.")
            output = _artifact_attestation(invocation.task)
    except BaseException as finalize_error:
        if error is None:
            error = finalize_error
        else:
            error.add_note(
                f"Task finalization also failed: {type(finalize_error).__name__}: {finalize_error}"
            )
        logs = {
            "stdout": (
                _log_attestation(invocation.stdout_path, dag)
                if invocation.stdout_path.is_file()
                else None
            ),
            "stderr": (
                _log_attestation(invocation.stderr_path, dag)
                if invocation.stderr_path.is_file()
                else None
            ),
        }
    attempt["finished_at"] = finished_at
    attempt["logs"] = logs
    record["finished_at"] = finished_at
    if error is not None:
        error_record = _error_record(error)
        attempt["status"] = "failed"
        attempt["error"] = error_record
        record["status"] = "failed"
        record["error"] = error_record
        return error
    attempt["status"] = "completed"
    record["status"] = "completed"
    record["output"] = output
    record["dependency_outputs"] = dict(invocation.dependency_outputs)
    return None


def _cancel_invocation_before_start(
    invocation: _SubprocessInvocation,
    state_tasks: Mapping[str, Any],
) -> None:
    record_value = state_tasks.get(invocation.task.task_id)
    if not isinstance(record_value, dict):
        raise ValueError(f"journal task {invocation.task.task_id} is invalid.")
    record = record_value
    attempt = _invocation_attempt(record, invocation)
    finished_at = _now()
    attempt["status"] = "cancelled_before_start"
    attempt["finished_at"] = finished_at
    attempt["logs"] = {"stdout": None, "stderr": None}
    record["status"] = "pending"
    record["started_at"] = None
    record["finished_at"] = None


def _orphan_invocation_after_interrupt(
    dag: PromotionDag,
    invocation: _SubprocessInvocation,
    state_tasks: Mapping[str, Any],
) -> None:
    record_value = state_tasks.get(invocation.task.task_id)
    if not isinstance(record_value, dict):
        raise ValueError(f"journal task {invocation.task.task_id} is invalid.")
    record = record_value
    attempt = _invocation_attempt(record, invocation)
    finished_at = _now()
    attempt["status"] = "orphaned_after_interrupt"
    attempt["finished_at"] = finished_at
    attempt["logs"] = {
        "stdout": (
            _log_attestation(invocation.stdout_path, dag)
            if invocation.stdout_path.is_file()
            else None
        ),
        "stderr": (
            _log_attestation(invocation.stderr_path, dag)
            if invocation.stderr_path.is_file()
            else None
        ),
    }
    record["status"] = "pending"
    record["started_at"] = None
    record["finished_at"] = None
    record.pop("output", None)
    record.pop("dependency_outputs", None)


def _clear_running_records_after_abort(
    state_tasks: Mapping[str, Any],
    *,
    reason: str,
) -> None:
    timestamp = _now()
    for record_value in state_tasks.values():
        if not isinstance(record_value, dict) or record_value.get("status") != "running":
            continue
        attempts = record_value.get("attempts")
        if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
            if attempts[-1].get("status") == "running":
                attempts[-1]["status"] = "orphaned"
                attempts[-1]["finished_at"] = timestamp
                attempts[-1]["orphaned_reason"] = reason
        record_value["status"] = "pending"
        record_value["started_at"] = None
        record_value["finished_at"] = None
        record_value.pop("output", None)
        record_value.pop("dependency_outputs", None)


def execute_dag(
    dag: PromotionDag,
    *,
    resume: bool = False,
    subprocess_runner: Callable[..., Any] | None = None,
    verify_inputs: bool = True,
    max_workers: int = 1,
) -> Path:
    """Execute one dependency-aware DAG with one main-thread state writer."""

    workers = _validate_max_workers(max_workers)
    resolved_inputs = (
        _verify_inputs(dag)
        if verify_inputs
        else {
            "verification": "disabled_for_orchestrator_test",
            "digest": semantic_sha256({"verification": "disabled_for_orchestrator_test"}),
        }
    )
    runner = subprocess.run if subprocess_runner is None else subprocess_runner
    dag.plan.output_root.mkdir(parents=True, exist_ok=True)
    resolved_dag_digest = _dag_digest(dag, resolved_inputs)
    lock_owner = _acquire_orchestrator_lock(dag, resolved_dag_digest)

    def run_locked() -> Path:
        state = _load_or_create_state(
            dag,
            resume=resume,
            resolved_inputs=resolved_inputs,
        )
        state["execution"] = {
            "max_workers": workers,
            "scheduler": "dependency_aware_threadpool_subprocess_only",
            "state_writer": "orchestrator_main_thread_only",
            "orchestrator_lock": {
                "schema": lock_owner["schema"],
                "pid": lock_owner["pid"],
                "host": lock_owner["host"],
                "dag_digest": lock_owner["dag_digest"],
            },
            "keyboard_interrupt_policy": (
                "stop_dispatch;cancel_not_started;wait_active;mark_unattested_outputs_orphaned"
            ),
        }
        _materialize_subject_scopes(dag, state)
        state_tasks = _prepare_execution_records(dag, state, resume=resume)
        journal = _journal_path(dag)
        task_order = {task.task_id: index for index, task in enumerate(dag.tasks)}
        pending = {
            task.task_id
            for task in dag.tasks
            if state_tasks[task.task_id].get("status") == "pending"
        }
        dispatch_sequence = state.get("dispatch_sequence")
        if not isinstance(dispatch_sequence, list):
            raise ValueError("promotion journal dispatch_sequence must be a list.")
        in_flight: dict[Future[_SubprocessOutcome], _SubprocessInvocation] = {}
        primary_error: BaseException | None = None
        executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="promotion-subprocess"
        )

        def write_state() -> None:
            state["updated_at"] = _now()
            _atomic_write_json(journal, state)

        def mark_dispatch_failure(
            invocation: _SubprocessInvocation,
            error: BaseException,
        ) -> None:
            record = state_tasks[invocation.task.task_id]
            attempt = _invocation_attempt(record, invocation)
            finished_at = _now()
            error_record = _error_record(error)
            attempt["status"] = "failed"
            attempt["finished_at"] = finished_at
            attempt["logs"] = {"stdout": None, "stderr": None}
            attempt["error"] = error_record
            record["status"] = "failed"
            record["finished_at"] = finished_at
            record["error"] = error_record

        try:
            while pending or in_flight:
                while primary_error is None and len(in_flight) < workers:
                    ready = next(
                        (
                            task
                            for task in dag.tasks
                            if task.task_id in pending
                            and all(
                                state_tasks[dependency].get("status") == "completed"
                                for dependency in task.dependencies
                            )
                        ),
                        None,
                    )
                    if ready is None:
                        break
                    pending.remove(ready.task_id)
                    invocation = _dispatch_invocation(
                        dag,
                        ready,
                        state_tasks[ready.task_id],
                        state_tasks,
                    )
                    dispatch_sequence.append(ready.task_id)
                    write_state()
                    try:
                        future = executor.submit(_run_subprocess_invocation, invocation, runner)
                    except BaseException as error:
                        mark_dispatch_failure(invocation, error)
                        primary_error = error
                        state["status"] = "failed"
                        write_state()
                        break
                    in_flight[future] = invocation

                if not in_flight:
                    if primary_error is not None:
                        break
                    if pending:
                        blocked = [task.task_id for task in dag.tasks if task.task_id in pending]
                        raise RuntimeError(
                            "promotion DAG has pending tasks with no satisfied dependency path: "
                            f"{blocked}."
                        )
                    break

                completed_futures, _ = wait(
                    tuple(in_flight),
                    return_when=FIRST_COMPLETED,
                )
                for future in sorted(
                    completed_futures,
                    key=lambda value: task_order[in_flight[value].task.task_id],
                ):
                    invocation = in_flight.pop(future)
                    try:
                        outcome = future.result()
                        if not isinstance(outcome, _SubprocessOutcome):
                            raise TypeError("subprocess worker returned an invalid outcome.")
                    except BaseException as error:
                        outcome = _SubprocessOutcome(error=error)
                    error = _complete_invocation(
                        dag,
                        invocation,
                        outcome,
                        state_tasks,
                    )
                    if error is not None and primary_error is None:
                        primary_error = error
                        state["status"] = "failed"
                    write_state()

                if primary_error is not None:
                    for future, invocation in list(in_flight.items()):
                        if future.cancel():
                            _cancel_invocation_before_start(invocation, state_tasks)
                            in_flight.pop(future)
                    state["status"] = "failed"
                    write_state()
        except KeyboardInterrupt as error:
            state["status"] = "interrupted_draining"
            state["last_error"] = _error_record(error)
            for future, invocation in list(in_flight.items()):
                if future.cancel():
                    _cancel_invocation_before_start(invocation, state_tasks)
                    in_flight.pop(future)
            write_state()
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as shutdown_error:
                error.add_note(
                    "Could not drain active subprocess workers; the orchestrator lock was retained: "
                    f"{type(shutdown_error).__name__}: {shutdown_error}"
                )
                error._retain_orchestrator_lock = True  # type: ignore[attr-defined]
                write_state()
                raise error from shutdown_error
            for invocation in in_flight.values():
                try:
                    _orphan_invocation_after_interrupt(dag, invocation, state_tasks)
                except BaseException as orphan_error:
                    error.add_note(
                        "Could not fully attest interrupted task logs: "
                        f"{type(orphan_error).__name__}: {orphan_error}"
                    )
                    record = state_tasks[invocation.task.task_id]
                    record["status"] = "pending"
                    record["started_at"] = None
                    record["finished_at"] = None
            _clear_running_records_after_abort(
                state_tasks,
                reason="orchestrator interrupted while draining subprocesses",
            )
            state["status"] = "interrupted"
            state["finished_at"] = _now()
            write_state()
            raise
        except BaseException as error:
            state["status"] = "failed_draining"
            state["last_error"] = _error_record(error)
            for future, invocation in list(in_flight.items()):
                if future.cancel():
                    _cancel_invocation_before_start(invocation, state_tasks)
                    in_flight.pop(future)
            write_state()
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as shutdown_error:
                error.add_note(
                    "Could not drain active subprocess workers; the orchestrator lock was retained: "
                    f"{type(shutdown_error).__name__}: {shutdown_error}"
                )
                error._retain_orchestrator_lock = True  # type: ignore[attr-defined]
                write_state()
                raise error from shutdown_error
            for invocation in in_flight.values():
                try:
                    _orphan_invocation_after_interrupt(dag, invocation, state_tasks)
                except BaseException as orphan_error:
                    error.add_note(
                        "Could not fully attest failed task logs: "
                        f"{type(orphan_error).__name__}: {orphan_error}"
                    )
            _clear_running_records_after_abort(
                state_tasks,
                reason="orchestrator failed while draining subprocesses",
            )
            state["status"] = "failed"
            state["finished_at"] = _now()
            write_state()
            raise
        else:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as error:
                state["status"] = "failed_draining"
                state["last_error"] = _error_record(error)
                write_state()
                error._retain_orchestrator_lock = True  # type: ignore[attr-defined]
                raise

        if primary_error is not None:
            state["status"] = "failed"
            state["finished_at"] = _now()
            state["updated_at"] = state["finished_at"]
            _atomic_write_json(journal, state)
            raise primary_error
        state["status"] = "completed"
        state["finished_at"] = _now()
        state["updated_at"] = state["finished_at"]
        _atomic_write_json(journal, state)
        return dag.tasks[-1].output

    try:
        result = run_locked()
    except BaseException as error:
        if getattr(error, "_retain_orchestrator_lock", False):
            error.add_note(
                "Promotion orchestrator lock retained because worker drain was not confirmed."
            )
            raise
        try:
            _release_orchestrator_lock(dag, lock_owner)
        except BaseException as release_error:
            error.add_note(
                "Could not release promotion orchestrator lock: "
                f"{type(release_error).__name__}: {release_error}"
            )
        raise
    _release_orchestrator_lock(dag, lock_owner)
    return result


def _positive_int_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Resolve relative plan paths from this explicit root instead of the plan directory.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workers", type=_positive_int_argument, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = load_plan(args.plan, root=args.root)
    dag = build_dag(plan, device=args.device)
    if args.dry_run:
        resolved_inputs = _verify_inputs(dag)
        print(
            json.dumps(
                dag_record(dag, resolved_inputs=resolved_inputs),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    output = execute_dag(dag, resume=args.resume, max_workers=args.max_workers)
    print(
        json.dumps(
            {
                "status": "completed",
                "analysis": str(output),
                "analysis_sha256": _sha256_file(output),
                "journal": str(_journal_path(dag)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
