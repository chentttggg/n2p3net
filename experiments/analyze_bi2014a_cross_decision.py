"""Analyze manifest-bound BI2014a cross-decision result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.bi2014a_schedule import BI2014A_FLASH_SCHEDULE  # noqa: E402
from research.contracts import (  # noqa: E402
    ArmContract,
    DecisionPlanContract,
    EvaluationRunContract,
    StatisticalDesignContract,
    TrainingRunContract,
    assert_contract_digest,
    canonical_json_bytes,
    scratch_procedure_record,
    semantic_sha256,
    training_procedure_record,
)
from research.evaluation import source_snapshot_sha256_from_archive_manifest  # noqa: E402
from stats.hierarchical import paired_interval  # noqa: E402
from transfer.bi_decision import bi2014a_expected_candidate_counts  # noqa: E402
from transfer.outcomes import (  # noqa: E402
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionStatus,
    build_decision_outcome_accounting,
)

MANIFEST_SCHEMA = "n2p3_bi2014a_cross_decision_experiment/2"
RESULT_SCHEMA = "n2p3_bi2014a_cross_decision_result/2"
ANALYSIS_SCHEMA = "n2p3_bi2014a_cross_decision_analysis/2"
POWER_PLAN_SCHEMA = "n2p3_power_plan_contract/1"
SUPPORTED_METRIC = "requested_participant_operational_hit_rate"


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str


@dataclass(frozen=True)
class CheckpointSpec:
    artifact: Artifact
    training: TrainingRunContract


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    result: Artifact
    replicate: str
    partition: str
    checkpoint_id: str | None
    evaluation: EvaluationRunContract
    evaluation_digest: str
    plan: DecisionPlanContract
    arm_contract: ArmContract

    @property
    def arm(self) -> str:
        return self.evaluation.arm_name

    @property
    def participants(self) -> tuple[str, ...]:
        return self.evaluation.requested_participant_keys


@dataclass(frozen=True)
class ManifestSpec:
    path: Path
    source_snapshot: Artifact
    target_cache_sha256: str
    target_identity_digest: str
    participants: tuple[str, ...]
    evidence_level: int
    replicate_aggregation: str
    iterations: int
    seed: int
    confidence_level: float
    checkpoints: Mapping[str, CheckpointSpec]
    arm_contracts: Mapping[str, ArmContract]
    design: StatisticalDesignContract
    design_digest: str
    runs: tuple[RunSpec, ...]


@dataclass(frozen=True)
class ValidatedRun:
    spec: RunSpec
    participant_rates: Mapping[str, float]
    accounting: Mapping[str, Any]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence.")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _sha(value: object, name: str) -> str:
    digest = _text(value, name).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} {path}: {error}") from error
    return dict(_mapping(value, name))


def _artifact(
    record: Mapping[str, Any],
    *,
    base: Path,
    path_field: str,
    sha_field: str,
    name: str,
) -> Artifact:
    path = Path(_text(record.get(path_field), f"{name}.{path_field}"))
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    expected = _sha(record.get(sha_field), f"{name}.{sha_field}")
    try:
        actual = _file_sha(path)
    except OSError as error:
        raise ValueError(f"cannot hash {name} {path}: {error}") from error
    if actual != expected:
        raise ValueError(f"{name} SHA mismatch.")
    return Artifact(path, expected)


def _contract(
    contract_type: type[TrainingRunContract]
    | type[EvaluationRunContract]
    | type[StatisticalDesignContract],
    record: object,
    digest: object,
    name: str,
) -> TrainingRunContract | EvaluationRunContract | StatisticalDesignContract:
    values = dict(_mapping(record, name))
    expected = _sha(digest, f"{name}_digest")
    assert_contract_digest(values, expected, name=name)
    try:
        return contract_type(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {error}") from error


def _character(value: object, name: str) -> str:
    raw = _text(value, name)
    decoded: dict[str, int] = {}
    for part in raw.split("|"):
        axis, separator, index_text = part.partition(":")
        if not separator or axis in decoded:
            raise ValueError(f"{name} must contain one BI row and one column.")
        try:
            index = int(index_text)
        except ValueError as error:
            raise ValueError(f"{name} has a non-integer index.") from error
        if str(index) != index_text or not 0 <= index < BI2014A_FLASH_SCHEDULE.grid_size:
            raise ValueError(f"{name} index is outside the BI grid.")
        decoded[axis] = index
    if set(decoded) != {"row", "column"}:
        raise ValueError(f"{name} must contain one BI row and one column.")
    return f"row:{decoded['row']}|column:{decoded['column']}"


def _validate_protocol(evaluation: EvaluationRunContract) -> None:
    protocol = evaluation.target_protocol
    required_protocol = {
        "estimand": "known_early_decisions_to_later_unknown_decisions",
        "split_axis": "decision_time",
        "calibration_truth_access": "adapter_visible",
        "test_truth_access": "scorer_only",
    }
    if any(protocol.get(key) != value for key, value in required_protocol.items()):
        raise ValueError("target protocol does not implement the BI early-to-later estimand.")
    _integer(protocol.get("calibration_selections"), "calibration_selections", 1)
    repetitions = _integer(protocol.get("test_repetitions"), "test_repetitions", 1)
    required_decision = {
        "task": "BI2014a_6x6_row_column",
        "schedule": BI2014A_FLASH_SCHEDULE.record(),
        "schedule_digest": BI2014A_FLASH_SCHEDULE.digest(),
        "aggregation": "cumulative_row_column_llr",
        "tie_policy": "abstain",
        "test_repetitions": repetitions,
    }
    if any(evaluation.decision.get(key) != value for key, value in required_decision.items()):
        raise ValueError("decision contract disagrees with BI2014a row/column semantics.")


def _parse_checkpoints(value: object, base: Path) -> dict[str, CheckpointSpec]:
    output: dict[str, CheckpointSpec] = {}
    for raw_id, raw in _mapping(value, "checkpoint_registry").items():
        checkpoint_id = _text(raw_id, "checkpoint_id")
        record = _mapping(raw, f"checkpoint {checkpoint_id}")
        artifact = _artifact(
            record,
            base=base,
            path_field="checkpoint_path",
            sha_field="checkpoint_sha256",
            name=f"checkpoint {checkpoint_id}",
        )
        training = _contract(
            TrainingRunContract,
            record.get("training_contract"),
            record.get("training_contract_digest"),
            f"checkpoint {checkpoint_id} training_contract",
        )
        assert isinstance(training, TrainingRunContract)
        output[checkpoint_id] = CheckpointSpec(artifact, training)
    return output


def _parse_arm_contracts(value: object) -> dict[str, ArmContract]:
    output: dict[str, ArmContract] = {}
    for raw_name, raw in _mapping(value, "arm_contracts").items():
        arm_name = _text(raw_name, "arm contract name")
        wrapper = _mapping(raw, f"arm contract {arm_name}")
        record = _mapping(wrapper.get("contract"), f"arm contract {arm_name}.contract")
        digest = _sha(wrapper.get("digest"), f"arm contract {arm_name}.digest")
        assert_contract_digest(record, digest, name=f"arm contract {arm_name}")
        contract = ArmContract.from_record(record)
        if contract.arm_name != arm_name or contract.digest() != digest:
            raise ValueError(f"arm contract {arm_name!r} identity mismatch.")
        output[arm_name] = contract
    if not output:
        raise ValueError("arm_contracts cannot be empty.")
    return output


def _parse_plan(
    value: object,
    digest: object,
    participants: tuple[str, ...],
    name: str,
) -> DecisionPlanContract:
    record = _mapping(value, f"{name}.decision_plan")
    expected_digest = _sha(digest, f"{name}.decision_plan_digest")
    assert_contract_digest(record, expected_digest, name=f"{name}.decision_plan")
    plan = DecisionPlanContract.from_record(record)
    if plan.digest() != expected_digest:
        raise ValueError(f"{name} decision plan digest disagrees with its typed contract.")
    if plan.requested_participant_keys != participants:
        raise ValueError(f"{name} decision plan requested cohort mismatch.")
    for entry in plan.entries:
        _character(entry.target_candidate, "target_candidate")
    return plan


def _validate_origin(
    run_id: str,
    evaluation: EvaluationRunContract,
    checkpoint_id: object,
    checkpoints: Mapping[str, CheckpointSpec],
) -> str | None:
    origin = evaluation.model_origin
    if origin.get("kind") == "scratch":
        if checkpoint_id is not None or set(origin) != {
            "kind",
            "initialization_contract_digest",
            "initialization",
        }:
            raise ValueError(f"scratch run {run_id!r} has an invalid model origin.")
        initialization = _mapping(origin.get("initialization"), "scratch initialization")
        if not initialization:
            raise ValueError("scratch initialization cannot be empty.")
        assert_contract_digest(
            initialization,
            str(origin.get("initialization_contract_digest")),
            name="scratch initialization",
        )
        return None
    if set(origin) != {"kind", "checkpoint_sha256", "training_contract_digest"}:
        raise ValueError(f"checkpoint run {run_id!r} has an invalid model origin.")
    resolved = _text(checkpoint_id, f"run {run_id} checkpoint_id")
    if resolved not in checkpoints:
        raise ValueError(f"run {run_id!r} references an unknown checkpoint.")
    checkpoint = checkpoints[resolved]
    expected = {
        "kind": "checkpoint",
        "checkpoint_sha256": checkpoint.artifact.sha256,
        "training_contract_digest": checkpoint.training.digest(),
    }
    if dict(origin) != expected:
        raise ValueError(f"run {run_id!r} model origin disagrees with checkpoint registry.")
    if evaluation.source_snapshot_sha256 != checkpoint.training.source_snapshot_sha256:
        raise ValueError(f"run {run_id!r} source snapshot disagrees with training contract.")
    if set(evaluation.requested_participant_keys) != set(
        checkpoint.training.holdout_participant_keys
    ):
        raise ValueError(f"run {run_id!r} participants do not equal checkpoint holdout.")
    return resolved


def _parse_run(
    value: object,
    *,
    base: Path,
    design: StatisticalDesignContract,
    checkpoints: Mapping[str, CheckpointSpec],
    arm_contracts: Mapping[str, ArmContract],
    target_cache_sha256: str,
    target_identity_digest: str,
    source_snapshot_sha256: str,
    evidence_level: int,
) -> RunSpec:
    record = _mapping(value, "run")
    run_id = _text(record.get("run_id"), "run_id")
    replicate = _text(record.get("training_replicate_key"), "training_replicate_key")
    partition = _text(record.get("partition_key"), "partition_key")
    if replicate not in design.training_replicate_keys or partition not in design.partition_keys:
        raise ValueError(f"run {run_id!r} uses an undeclared replicate or partition.")
    evaluation = _contract(
        EvaluationRunContract,
        record.get("evaluation_contract"),
        record.get("evaluation_contract_digest"),
        f"run {run_id} evaluation_contract",
    )
    assert isinstance(evaluation, EvaluationRunContract)
    if (
        evaluation.target_cache_sha256 != target_cache_sha256
        or evaluation.target_identity_digest != target_identity_digest
        or evaluation.source_snapshot_sha256 != source_snapshot_sha256
    ):
        raise ValueError(f"run {run_id!r} dataset/source identity disagrees with manifest.")
    _validate_protocol(evaluation)
    repetitions = _integer(
        evaluation.target_protocol.get("test_repetitions"), "test_repetitions", 1
    )
    if evidence_level > repetitions:
        raise ValueError(f"run {run_id!r} does not reach the requested evidence level.")
    checkpoint = _validate_origin(run_id, evaluation, record.get("checkpoint_id"), checkpoints)
    if evaluation.arm_name not in arm_contracts:
        raise ValueError(f"run {run_id!r} references an undeclared arm contract.")
    arm_contract = arm_contracts[evaluation.arm_name]
    if evaluation.model_origin.get("kind") != arm_contract.model_origin_kind:
        raise ValueError(f"run {run_id!r} model origin kind disagrees with its arm contract.")
    if evaluation.adaptation.get("procedure") != arm_contract.adaptation_procedure:
        raise ValueError(f"run {run_id!r} adaptation disagrees with its arm contract.")
    source_procedure = (
        scratch_procedure_record(evaluation.model_origin["initialization"])
        if checkpoint is None
        else training_procedure_record(checkpoints[checkpoint].training)
    )
    if source_procedure != arm_contract.source_training_procedure:
        raise ValueError(f"run {run_id!r} source training procedure changed within its arm.")
    result = _artifact(
        record,
        base=base,
        path_field="result_path",
        sha_field="result_sha256",
        name=f"run {run_id} result",
    )
    return RunSpec(
        run_id=run_id,
        result=result,
        replicate=replicate,
        partition=partition,
        checkpoint_id=checkpoint,
        evaluation=evaluation,
        evaluation_digest=evaluation.digest(),
        plan=_parse_plan(
            record.get("decision_plan"),
            record.get("decision_plan_digest"),
            evaluation.requested_participant_keys,
            run_id,
        ),
        arm_contract=arm_contract,
    )


def _validate_adaptation(arm: str, runs: tuple[RunSpec, ...]) -> None:
    procedures: set[bytes] = set()
    by_replicate: dict[str, set[bytes]] = {}
    random_seeds: dict[str, set[int]] = {}
    for run in runs:
        adaptation = run.evaluation.adaptation
        if set(adaptation) != {"procedure", "replicate_parameters"}:
            raise ValueError(f"arm {arm!r} adaptation contract is not structured.")
        procedure = _mapping(adaptation["procedure"], "adaptation procedure")
        replicate = _mapping(adaptation["replicate_parameters"], "replicate parameters")
        if not procedure or replicate.get("training_replicate_key") != run.replicate:
            raise ValueError(f"arm {arm!r} adaptation contract is inconsistent.")
        random_seed = _integer(replicate.get("random_seed"), "adaptation random_seed")
        procedures.add(canonical_json_bytes(procedure))
        by_replicate.setdefault(run.replicate, set()).add(canonical_json_bytes(replicate))
        random_seeds.setdefault(run.replicate, set()).add(random_seed)
    if len(procedures) != 1 or any(len(values) != 1 for values in by_replicate.values()):
        raise ValueError(f"arm {arm!r} adaptation changes outside the replicate axis.")
    if any(len(values) != 1 for values in random_seeds.values()):
        raise ValueError(f"arm {arm!r} changes adaptation seed within one replicate.")
    target_seeds = [next(iter(values)) for values in random_seeds.values()]
    if len(set(target_seeds)) != len(target_seeds):
        raise ValueError(f"arm {arm!r} reuses adaptation randomness across replicates.")


def _validate_checkpoint_replicate_seeds(
    arm: str,
    runs: tuple[RunSpec, ...],
    checkpoints: Mapping[str, CheckpointSpec],
) -> None:
    seeds_by_replicate: dict[str, set[int]] = {}
    for run in runs:
        if run.checkpoint_id is None:
            continue
        seeds_by_replicate.setdefault(run.replicate, set()).add(
            checkpoints[run.checkpoint_id].training.seed
        )
    if any(len(seeds) != 1 for seeds in seeds_by_replicate.values()):
        raise ValueError(f"arm {arm!r} changes training seed across one replicate.")
    seeds = [next(iter(values)) for values in seeds_by_replicate.values()]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"arm {arm!r} reuses source-training seed across replicates.")


def _validate_partitions(
    runs: tuple[RunSpec, ...],
    design: StatisticalDesignContract,
    participants: tuple[str, ...],
) -> None:
    partition_sets: list[set[str]] = []
    for partition in design.partition_keys:
        cells = tuple(run for run in runs if run.partition == partition)
        if len({run.participants for run in cells}) != 1 or len({run.plan for run in cells}) != 1:
            raise ValueError("participant or decision plan changes across matched cells.")
        partition_sets.append(set(cells[0].participants))
    overlaps = any(
        left & right
        for index, left in enumerate(partition_sets)
        for right in partition_sets[index + 1 :]
    )
    if overlaps or set().union(*partition_sets) != set(participants):
        raise ValueError("participant partitions must be disjoint and complete.")


def _validate_origin_reuse(runs: tuple[RunSpec, ...]) -> None:
    checkpoint_cells: dict[str, set[tuple[str, str]]] = {}
    scratch_replicates: dict[str, set[str]] = {}
    for run in runs:
        digest = semantic_sha256(run.evaluation.model_origin)
        if run.checkpoint_id is None:
            scratch_replicates.setdefault(digest, set()).add(run.replicate)
        else:
            checkpoint_cells.setdefault(digest, set()).add((run.replicate, run.partition))
    if any(len(cells) > 1 for cells in checkpoint_cells.values()) or any(
        len(values) > 1 for values in scratch_replicates.values()
    ):
        raise ValueError("model origin is reused across declared independent units.")


def _validate_run_grid(
    runs: tuple[RunSpec, ...],
    design: StatisticalDesignContract,
    participants: tuple[str, ...],
    checkpoints: Mapping[str, CheckpointSpec],
) -> None:
    if len({run.run_id for run in runs}) != len(runs) or len(
        {run.result.path for run in runs}
    ) != len(runs):
        raise ValueError("run IDs and result artifacts must be unique.")
    arms = tuple(dict.fromkeys(run.arm for run in runs))
    if {arm for contrast in design.planned_contrasts for arm in contrast} - set(arms):
        raise ValueError("planned contrast references an undeclared arm.")
    expected = {
        (arm, replicate, partition)
        for arm in arms
        for replicate in design.training_replicate_keys
        for partition in design.partition_keys
    }
    observed = [(run.arm, run.replicate, run.partition) for run in runs]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("manifest run grid is not a complete matched design.")
    _validate_partitions(runs, design, participants)
    if set(checkpoints) != {run.checkpoint_id for run in runs if run.checkpoint_id is not None}:
        raise ValueError("checkpoint registry has unused entries.")
    protocol_digests = {
        semantic_sha256(
            {
                "target_protocol": run.evaluation.target_protocol,
                "decision": run.evaluation.decision,
                "evidence_scope": run.evaluation.evidence_scope,
            }
        )
        for run in runs
    }
    if len(protocol_digests) != 1:
        raise ValueError("matched runs use different target/decision evidence contracts.")
    for arm in arms:
        arm_runs = tuple(run for run in runs if run.arm == arm)
        _validate_adaptation(arm, arm_runs)
        _validate_checkpoint_replicate_seeds(arm, arm_runs, checkpoints)
    _validate_origin_reuse(runs)


def _validate_power_plan(
    value: object,
    design: StatisticalDesignContract,
    metric: Mapping[str, Any],
) -> None:
    record = _mapping(value, "power_plan_contract")
    if (
        record.get("schema") != POWER_PLAN_SCHEMA
        or record.get("inference_scope") != "training_procedure"
        or tuple(record.get("training_replicate_keys", ())) != design.training_replicate_keys
        or tuple(tuple(pair) for pair in record.get("planned_contrasts", ()))
        != design.planned_contrasts
        or canonical_json_bytes(record.get("primary_metric")) != canonical_json_bytes(metric)
        or record.get("planned_independent_replicates") != len(design.training_replicate_keys)
    ):
        raise ValueError("power plan disagrees with the statistical design.")
    parameters = _mapping(record.get("planning_parameters"), "power planning_parameters")
    _text(parameters.get("method"), "power method")
    _text(parameters.get("effect_definition"), "power effect_definition")
    for field in ("target_power", "alpha"):
        probability = float(parameters.get(field, math.nan))
        if not 0.0 < probability < 1.0:
            raise ValueError(f"power {field} must be in (0,1).")
    if design.power_plan_digest is None:
        raise ValueError("training procedure design lacks a power plan digest.")
    assert_contract_digest(record, design.power_plan_digest, name="power_plan_contract")


def _parse_bootstrap(value: object) -> tuple[int, int, float]:
    record = _mapping(value, "bootstrap")
    confidence = float(record.get("confidence_level", math.nan))
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence_level must be in (0,1).")
    return (
        _integer(record.get("iterations"), "bootstrap iterations", 1000),
        _integer(record.get("seed"), "bootstrap seed"),
        confidence,
    )


def _validate_inference_plan(
    manifest: Mapping[str, Any],
    design: StatisticalDesignContract,
    metric: Mapping[str, Any],
) -> None:
    power_plan = manifest.get("power_plan_contract")
    if design.inference_scope == "training_procedure":
        if len(design.training_replicate_keys) < 2:
            raise ValueError("training procedure needs independent retraining replicates.")
        _validate_power_plan(power_plan, design, metric)
    elif power_plan is not None:
        raise ValueError("conditional analysis must not carry a training-procedure power plan.")


def _parse_manifest(path: str | Path) -> ManifestSpec:
    manifest_path = Path(path).resolve()
    raw = _read_json(manifest_path, "experiment manifest")
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA!r}; v1 is unsupported.")
    base = manifest_path.parent
    source_manifest = _artifact(
        _mapping(raw.get("source_snapshot"), "source_snapshot"),
        base=base,
        path_field="manifest_path",
        sha_field="manifest_sha256",
        name="source snapshot manifest",
    )
    source_sha256 = source_snapshot_sha256_from_archive_manifest(source_manifest.path)
    source_payload = _read_json(source_manifest.path, "source snapshot manifest")
    source_archive = (source_manifest.path.parent / str(source_payload["archive"])).resolve()
    source = Artifact(source_archive, source_sha256)
    target = _mapping(raw.get("target_dataset"), "target_dataset")
    participants = tuple(
        sorted(
            _text(item, "participant_key")
            for item in _sequence(target.get("requested_participant_keys"), "participants")
        )
    )
    if not participants or len(participants) != len(set(participants)):
        raise ValueError("target participants must be non-empty and unique.")
    metric = _mapping(raw.get("metric"), "metric")
    if metric.get("name") != SUPPORTED_METRIC:
        raise ValueError(f"metric.name must be {SUPPORTED_METRIC!r}.")
    evidence_level = _integer(metric.get("evidence_level"), "metric evidence_level", 1)
    design = _contract(
        StatisticalDesignContract,
        raw.get("statistical_design"),
        raw.get("statistical_design_digest"),
        "statistical_design",
    )
    assert isinstance(design, StatisticalDesignContract)
    expected_aggregation = (
        "mean_across_declared_frozen_models"
        if design.inference_scope == "conditional_frozen_models"
        else "independent_training_replicates"
    )
    if metric.get("replicate_aggregation") != expected_aggregation:
        raise ValueError("metric replicate aggregation disagrees with inference scope.")
    iterations, seed, confidence = _parse_bootstrap(raw.get("bootstrap"))
    checkpoints = _parse_checkpoints(raw.get("checkpoint_registry"), base)
    arm_contracts = _parse_arm_contracts(raw.get("arm_contracts"))
    if any(item.training.source_snapshot_sha256 != source.sha256 for item in checkpoints.values()):
        raise ValueError("checkpoint source snapshot disagrees with physical archive.")
    target_cache = _sha(target.get("target_cache_sha256"), "target_cache_sha256")
    target_identity = _sha(target.get("target_identity_digest"), "target_identity_digest")
    runs = tuple(
        _parse_run(
            item,
            base=base,
            design=design,
            checkpoints=checkpoints,
            arm_contracts=arm_contracts,
            target_cache_sha256=target_cache,
            target_identity_digest=target_identity,
            source_snapshot_sha256=source.sha256,
            evidence_level=evidence_level,
        )
        for item in _sequence(raw.get("runs"), "runs")
    )
    if not runs:
        raise ValueError("manifest runs cannot be empty.")
    _validate_run_grid(runs, design, participants, checkpoints)
    _validate_inference_plan(raw, design, metric)
    return ManifestSpec(
        manifest_path,
        source,
        target_cache,
        target_identity,
        participants,
        evidence_level,
        expected_aggregation,
        iterations,
        seed,
        confidence,
        checkpoints,
        arm_contracts,
        design,
        design.digest(),
        runs,
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value, "optional text")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("decision timing must be finite.")
    return number


def _parse_outcome(raw: object, participant_field: str) -> DecisionOutcome:
    record = _mapping(raw, "decision outcome")
    coverage_record = _mapping(record.get("candidate_coverage"), "candidate_coverage")
    coverage = CandidateCoverage.from_mappings(
        _mapping(coverage_record.get("expected_event_counts"), "expected counts"),
        _mapping(coverage_record.get("observed_event_counts"), "observed counts"),
    )
    if dict(coverage_record) != coverage.to_record():
        raise ValueError("candidate coverage summary is inconsistent.")
    timing = _mapping(record.get("timing"), "decision timing")
    try:
        outcome = DecisionOutcome(
            key=DecisionKey(
                _text(record.get(participant_field), participant_field),
                _text(record.get("decision_id"), "decision_id"),
            ),
            evidence_level=_integer(record.get("evidence_level"), "evidence_level", 1),
            status=DecisionStatus(record.get("status")),
            coverage=coverage,
            target_candidate=_optional_text(record.get("target_candidate")),
            predicted_candidate=_optional_text(record.get("predicted_candidate")),
            failure_reason=_optional_text(record.get("failure_reason")),
            onset_start_s=_optional_float(timing.get("onset_start_s")),
            onset_end_s=_optional_float(timing.get("onset_end_s")),
            evidence_available_s=_optional_float(timing.get("evidence_available_s")),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid decision outcome: {error}") from error
    if timing != outcome.to_record()["timing"]:
        raise ValueError("decision timing summary is inconsistent.")
    return outcome


def _validate_progression(series: Sequence[DecisionOutcome]) -> None:
    for previous, current in zip(series, series[1:], strict=False):
        previous_counts = dict(previous.coverage.observed_event_counts)
        current_counts = dict(current.coverage.observed_event_counts)
        if any(current_counts.get(key, 0) < count for key, count in previous_counts.items()):
            raise ValueError("candidate coverage decreases with more evidence.")
        for earlier, later in (
            (previous.onset_end_s, current.onset_end_s),
            (previous.evidence_available_s, current.evidence_available_s),
        ):
            if earlier is not None and later is not None and later < earlier:
                raise ValueError("decision time decreases with more evidence.")


def _validate_bi_outcomes(
    outcomes: tuple[DecisionOutcome, ...],
    plan: DecisionPlanContract,
    levels: tuple[int, ...],
) -> None:
    plan_by_key = {
        DecisionKey(item.participant_key, item.decision_id): item for item in plan.entries
    }
    for outcome in outcomes:
        if dict(outcome.coverage.expected_event_counts) != bi2014a_expected_candidate_counts(
            outcome.evidence_level
        ):
            raise ValueError("outcome coverage disagrees with the BI schedule.")
        planned = plan_by_key.get(outcome.key)
        if planned is None or outcome.target_candidate != planned.target_candidate:
            raise ValueError("outcome target disagrees with the frozen decision plan.")
        if outcome.predicted_candidate is not None:
            _character(outcome.predicted_candidate, "predicted_candidate")
    for key in plan_by_key:
        series = sorted(
            (outcome for outcome in outcomes if outcome.key == key),
            key=lambda outcome: outcome.evidence_level,
        )
        if tuple(outcome.evidence_level for outcome in series) != levels:
            raise ValueError("outcomes do not equal the manifest-frozen decision plan.")
        _validate_progression(series)


def _parse_failures(
    value: object,
    outcomes: tuple[DecisionOutcome, ...],
) -> dict[DecisionKey, dict[str, str]]:
    failures: dict[DecisionKey, dict[str, str]] = {}
    for raw in _sequence(value, "decision_failures"):
        record = _mapping(raw, "decision failure")
        key = DecisionKey(
            _text(record.get("participant_key"), "participant_key"),
            _text(record.get("decision_id"), "decision_id"),
        )
        if key in failures:
            raise ValueError("decision_failures contains duplicate keys.")
        status = DecisionStatus(record.get("status"))
        if status not in {DecisionStatus.INCOMPLETE, DecisionStatus.FIT_FAILURE}:
            raise ValueError("decision failure status must be incomplete or fit_failure.")
        failure = {
            "status": status.value,
            "stage": _text(record.get("stage"), "failure stage"),
            "reason": _text(record.get("reason"), "failure reason"),
        }
        matching = tuple(outcome for outcome in outcomes if outcome.key == key)
        if not matching or any(
            outcome.status != status or outcome.failure_reason != failure["reason"]
            for outcome in matching
        ):
            raise ValueError("decision failure disagrees with primary outcomes.")
        failures[key] = failure
    failed_outcome_keys = {
        outcome.key
        for outcome in outcomes
        if outcome.status in {DecisionStatus.INCOMPLETE, DecisionStatus.FIT_FAILURE}
    }
    if failed_outcome_keys != set(failures):
        raise ValueError("decision_failures does not cover every failed outcome.")
    return failures


def _validate_result(manifest: ManifestSpec, spec: RunSpec) -> ValidatedRun:
    result = _read_json(spec.result.path, f"run {spec.run_id} result")
    if result.get("schema") != RESULT_SCHEMA or result.get("run_status") != "completed":
        raise ValueError(f"run {spec.run_id!r} is not a completed v2 result.")
    if "subject_decision_ledger" in result:
        raise ValueError("subject_decision_ledger is obsolete; primary outcomes are authoritative.")
    evaluation = _contract(
        EvaluationRunContract,
        result.get("evaluation_contract"),
        result.get("evaluation_contract_digest"),
        f"run {spec.run_id} result evaluation_contract",
    )
    assert isinstance(evaluation, EvaluationRunContract)
    if canonical_json_bytes(evaluation.record()) != canonical_json_bytes(spec.evaluation.record()):
        raise ValueError("result evaluation contract disagrees with manifest.")
    result_plan = _parse_plan(
        result.get("decision_plan"),
        result.get("decision_plan_digest"),
        spec.participants,
        f"run {spec.run_id} result",
    )
    if result_plan.digest() != spec.plan.digest():
        raise ValueError("result decision plan disagrees with manifest.")
    participant_accounting = {
        "requested": len(spec.plan.requested_participant_keys),
        "decision_planned": len(spec.plan.participant_keys),
        "selection_failed": len(spec.plan.participant_selection_failures),
    }
    if result.get("participant_accounting") != participant_accounting:
        raise ValueError("participant_accounting disagrees with the decision plan.")
    participants = tuple(
        sorted(
            _text(item, "participant_key")
            for item in _sequence(
                result.get("requested_participant_keys"), "requested participants"
            )
        )
    )
    expected_checkpoint = (
        None
        if spec.checkpoint_id is None
        else manifest.checkpoints[spec.checkpoint_id].artifact.sha256
    )
    if (
        participants != spec.participants
        or result.get("target_cache_sha256") != manifest.target_cache_sha256
        or result.get("checkpoint_sha256") != expected_checkpoint
    ):
        raise ValueError("result participant/cache/checkpoint provenance mismatch.")
    levels = tuple(range(1, int(evaluation.target_protocol["test_repetitions"]) + 1))
    outcomes = tuple(
        _parse_outcome(raw, manifest.design.participant_cluster_key)
        for raw in _sequence(result.get("decision_outcomes"), "decision_outcomes")
    )
    _validate_bi_outcomes(outcomes, spec.plan, levels)
    failures = _parse_failures(result.get("decision_failures"), outcomes)
    plan_keys = tuple(
        DecisionKey(item.participant_key, item.decision_id) for item in spec.plan.entries
    )
    accounting = build_decision_outcome_accounting(
        outcomes,
        requested_decisions=plan_keys,
        evidence_levels=levels,
    ).to_record()
    data_ineligible = sum(
        failure["stage"] == "decision_eligibility" for failure in failures.values()
    )
    evaluation_failed = len(failures) - data_ineligible
    accounting["data_eligible"] = len(plan_keys) - data_ineligible
    accounting["data_ineligible"] = data_ineligible
    accounting["evaluation_successful"] = len(plan_keys) - data_ineligible - evaluation_failed
    accounting["evaluation_failed"] = evaluation_failed
    if canonical_json_bytes(result.get("decision_accounting")) != canonical_json_bytes(accounting):
        raise ValueError("decision_accounting disagrees with primary outcomes and failure ledger.")
    rates = {}
    selection_failed = {
        failure.participant_key for failure in spec.plan.participant_selection_failures
    }
    for participant in spec.participants:
        requested = sum(key.subject_id == participant for key in plan_keys)
        correct = sum(
            outcome.key.subject_id == participant
            and outcome.evidence_level == manifest.evidence_level
            and outcome.status == DecisionStatus.CORRECT
            for outcome in outcomes
        )
        if participant in selection_failed:
            if requested:
                raise AssertionError("selection-failed participant has planned decisions")
            rates[participant] = 0.0
        else:
            if requested < 1:
                raise ValueError("planned participant has no decision denominator.")
            rates[participant] = correct / requested
    return ValidatedRun(spec, rates, accounting)


def _cell_values(
    manifest: ManifestSpec,
    runs: tuple[ValidatedRun, ...],
) -> tuple[tuple[str, ...], dict[tuple[str, str, str], float], dict[str, dict[str, float]]]:
    indexed = {(run.spec.arm, run.spec.replicate, run.spec.partition): run for run in runs}
    arms = tuple(dict.fromkeys(run.spec.arm for run in runs))
    values: dict[tuple[str, str, str], float] = {}
    replicate_means: dict[str, dict[str, float]] = {}
    for arm in arms:
        replicate_means[arm] = {}
        for replicate in manifest.design.training_replicate_keys:
            participant_values = {
                participant: value
                for partition in manifest.design.partition_keys
                for participant, value in indexed[
                    (arm, replicate, partition)
                ].participant_rates.items()
            }
            values.update(
                {
                    (arm, replicate, participant): value
                    for participant, value in participant_values.items()
                }
            )
            replicate_means[arm][replicate] = float(np.mean(list(participant_values.values())))
    return arms, values, replicate_means


def _analysis_records(
    manifest: ManifestSpec,
    arms: tuple[str, ...],
    values: Mapping[tuple[str, str, str], float],
    replicate_means: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    conditional = manifest.design.inference_scope == "conditional_frozen_models"
    participant_means = {
        (arm, participant): float(
            np.mean(
                [
                    values[(arm, replicate, participant)]
                    for replicate in manifest.design.training_replicate_keys
                ]
            )
        )
        for arm in arms
        for participant in manifest.participants
    }
    metrics = {
        arm: {
            (
                "conditional_frozen_model_requested_participant_operational_hit_rate"
                if conditional
                else "training_randomness_requested_participant_operational_hit_rate"
            ): float(
                np.mean(
                    [participant_means[(arm, participant)] for participant in manifest.participants]
                )
            ),
            "per_training_replicate_requested_participant_operational_hit_rate": dict(
                replicate_means[arm]
            ),
        }
        for arm in arms
    }
    contrasts = []
    for index, (left, right) in enumerate(manifest.design.planned_contrasts):
        labels = manifest.participants if conditional else manifest.design.training_replicate_keys
        differences = np.asarray(
            [
                (
                    participant_means[(left, label)] - participant_means[(right, label)]
                    if conditional
                    else replicate_means[left][label] - replicate_means[right][label]
                )
                for label in labels
            ]
        )
        interval = paired_interval(
            differences,
            inference_scope=manifest.design.inference_scope,
            iterations=manifest.iterations,
            seed=manifest.seed + index,
            confidence_level=manifest.confidence_level,
        )
        contrasts.append(
            {
                "left": left,
                "right": right,
                "paired_unit": ("participant" if conditional else "independent_training_replicate"),
                "paired_unit_differences": dict(
                    zip(labels, differences.astype(float).tolist(), strict=True)
                ),
                "paired_bootstrap_interval": interval.record(),
            }
        )
    claim = (
        "conditional_on_manifest_declared_frozen_models_and_target_cohort"
        if conditional
        else "training_randomness_conditional_on_manifest_target_cohort"
    )
    return metrics, contrasts, claim


def _decision_endpoints(accounting: Mapping[str, Any], evidence_level: int) -> dict[str, Any]:
    counts = accounting["by_evidence_level"][str(evidence_level)]
    correct = int(counts["correct"])
    requested = int(accounting["requested"])
    data_eligible = int(accounting["data_eligible"])
    successful = int(accounting["evaluation_successful"])
    return {
        "estimand_unit": "planned_decision",
        "planned_decision_operational_hit_rate": (correct / requested if requested else None),
        "data_eligible_decision_operational_hit_rate": (
            correct / data_eligible if data_eligible else None
        ),
        "evaluation_successful_decision_hit_rate": (correct / successful if successful else None),
        "denominators": {
            "planned_decisions": requested,
            "data_eligible_decisions": data_eligible,
            "evaluation_successful_decisions": successful,
        },
    }


def analyze_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _parse_manifest(path)
    validated = tuple(_validate_result(manifest, run) for run in manifest.runs)
    arms, values, replicate_means = _cell_values(manifest, validated)
    metrics, contrasts, claim = _analysis_records(manifest, arms, values, replicate_means)
    conditional = manifest.design.inference_scope == "conditional_frozen_models"
    return {
        "schema": ANALYSIS_SCHEMA,
        "manifest": str(manifest.path),
        "manifest_sha256": _file_sha(manifest.path),
        "source_snapshot": {
            "archive_path": str(manifest.source_snapshot.path),
            "archive_sha256": manifest.source_snapshot.sha256,
            "verified": True,
        },
        "statistical_design_digest": manifest.design_digest,
        "inference": {
            "scope": manifest.design.inference_scope,
            "claim": claim,
            "sampling_unit": "participant" if conditional else "independent_training_replicate",
            "target_cohort_resampled": conditional,
            "p_values_computed": False,
            "intervals_are_pointwise": True,
        },
        "metric": {
            "name": SUPPORTED_METRIC,
            "evidence_level": manifest.evidence_level,
            "replicate_aggregation": manifest.replicate_aggregation,
            "estimand_unit": "requested_participant",
            "selection_failed_participant_value": 0.0,
            "all_noncorrect_statuses_remain_in_the_denominator": True,
        },
        "requested_participants": len(manifest.participants),
        "training_replicates": list(manifest.design.training_replicate_keys),
        "participant_partitions": list(manifest.design.partition_keys),
        "metrics": metrics,
        "planned_contrasts": contrasts,
        "runs": [
            {
                "run_id": run.spec.run_id,
                "arm_name": run.spec.arm,
                "training_replicate_key": run.spec.replicate,
                "partition_key": run.spec.partition,
                "result_sha256": run.spec.result.sha256,
                "evaluation_contract_digest": run.spec.evaluation_digest,
                "decision_accounting": dict(run.accounting),
                "decision_endpoints": _decision_endpoints(run.accounting, manifest.evidence_level),
            }
            for run in validated
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_manifest(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "inference": result["inference"]}))


if __name__ == "__main__":
    main()
