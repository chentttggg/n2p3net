"""Build one strict BI2014a experiment manifest from real runner results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research.contracts import (  # noqa: E402
    ArmContract,
    DecisionPlanContract,
    EvaluationRunContract,
    StatisticalDesignContract,
    assert_contract_digest,
    canonical_json_bytes,
    scratch_procedure_record,
    semantic_sha256,
    training_procedure_record,
)
from research.evaluation import source_snapshot_sha256_from_archive_manifest  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    checkpoint_training_contract,
    load_checkpoint_payload,
)

MANIFEST_SCHEMA = "n2p3_bi2014a_cross_decision_experiment/2"
RESULT_SCHEMA = "n2p3_bi2014a_cross_decision_result/2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON mapping.")
    return value


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _evaluation(result: Mapping[str, Any], name: str) -> EvaluationRunContract:
    record = result.get("evaluation_contract")
    digest = result.get("evaluation_contract_digest")
    if not isinstance(record, Mapping) or not isinstance(digest, str):
        raise ValueError(f"{name} lacks an evaluation contract.")
    assert_contract_digest(record, digest, name=f"{name} evaluation_contract")
    contract = EvaluationRunContract(**dict(record))
    if contract.digest() != digest:
        raise ValueError(f"{name} evaluation contract digest mismatch.")
    return contract


def _decision_plan(result: Mapping[str, Any], name: str) -> DecisionPlanContract:
    record = result.get("decision_plan")
    digest = result.get("decision_plan_digest")
    if not isinstance(record, Mapping) or not isinstance(digest, str):
        raise ValueError(f"{name} lacks a decision plan contract.")
    assert_contract_digest(record, digest, name=f"{name} decision_plan")
    plan = DecisionPlanContract.from_record(record)
    if plan.digest() != digest:
        raise ValueError(f"{name} decision plan digest mismatch.")
    return plan


def build_manifest(
    *,
    result_paths: Sequence[str | Path],
    source_snapshot_manifest: str | Path,
    inference_scope: str,
    planned_contrasts: Sequence[Sequence[str]],
    output: str | Path,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 0,
    confidence_level: float = 0.95,
    evidence_level: int = 2,
    power_plan_contract: Mapping[str, Any] | None = None,
) -> Path:
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_path = Path(source_snapshot_manifest).resolve()
    source_sha256 = source_snapshot_sha256_from_archive_manifest(source_manifest_path)

    loaded: list[tuple[Path, dict[str, Any], EvaluationRunContract, DecisionPlanContract]] = []
    for raw_path in result_paths:
        path = Path(raw_path).resolve()
        result = _read_json(path, f"result {path}")
        if result.get("schema") != RESULT_SCHEMA or result.get("run_status") != "completed":
            raise ValueError(f"{path} is not a completed BI result/2 artifact.")
        evaluation = _evaluation(result, str(path))
        plan = _decision_plan(result, str(path))
        if evaluation.source_snapshot_sha256 != source_sha256:
            raise ValueError(f"{path} source snapshot disagrees with physical freeze.")
        if Path(str(result.get("source_snapshot_manifest"))).resolve() != source_manifest_path:
            raise ValueError(f"{path} source snapshot manifest path mismatch.")
        if result.get("source_snapshot_sha256") != source_sha256:
            raise ValueError(f"{path} source snapshot digest mismatch.")
        if (
            plan.target_cache_sha256 != evaluation.target_cache_sha256
            or plan.target_identity_digest != evaluation.target_identity_digest
            or plan.requested_participant_keys != evaluation.requested_participant_keys
        ):
            raise ValueError(f"{path} decision plan disagrees with evaluation identity.")
        loaded.append((path, result, evaluation, plan))
    if not loaded:
        raise ValueError("at least one runner result is required.")

    target_cache_values = {item[2].target_cache_sha256 for item in loaded}
    target_identity_values = {item[2].target_identity_digest for item in loaded}
    if len(target_cache_values) != 1 or len(target_identity_values) != 1:
        raise ValueError("runner results disagree on target dataset identity.")

    checkpoint_registry: dict[str, dict[str, Any]] = {}
    checkpoint_by_result: dict[Path, str | None] = {}
    training_by_checkpoint: dict[str, Any] = {}
    for path, result, evaluation, _ in loaded:
        if evaluation.model_origin.get("kind") == "scratch":
            if result.get("checkpoint") is not None or result.get("checkpoint_sha256") is not None:
                raise ValueError(f"scratch result {path} carries checkpoint provenance.")
            checkpoint_by_result[path] = None
            continue
        checkpoint_path = Path(str(result.get("checkpoint"))).resolve()
        checkpoint_sha = _sha256(checkpoint_path)
        if checkpoint_sha != result.get("checkpoint_sha256"):
            raise ValueError(f"result {path} checkpoint SHA mismatch.")
        payload = load_checkpoint_payload(checkpoint_path)
        training = checkpoint_training_contract(payload)
        if training.source_snapshot_sha256 != source_sha256:
            raise ValueError(f"checkpoint {checkpoint_path} source snapshot mismatch.")
        if evaluation.model_origin != {
            "kind": "checkpoint",
            "checkpoint_sha256": checkpoint_sha,
            "training_contract_digest": training.digest(),
        }:
            raise ValueError(f"result {path} model origin disagrees with checkpoint payload.")
        checkpoint_id = f"sha256:{checkpoint_sha}"
        checkpoint_registry.setdefault(
            checkpoint_id,
            {
                "checkpoint_path": _relative(checkpoint_path, output_path.parent),
                "checkpoint_sha256": checkpoint_sha,
                "training_contract": training.record(),
                "training_contract_digest": training.digest(),
            },
        )
        checkpoint_by_result[path] = checkpoint_id
        training_by_checkpoint[checkpoint_id] = training

    arm_contracts: dict[str, dict[str, Any]] = {}
    for arm_name in sorted({item[2].arm_name for item in loaded}):
        arm_rows = [item for item in loaded if item[2].arm_name == arm_name]
        kinds = {row[2].model_origin.get("kind") for row in arm_rows}
        procedures = {canonical_json_bytes(row[2].adaptation["procedure"]) for row in arm_rows}
        source_procedures = set()
        for path, _, evaluation, _ in arm_rows:
            checkpoint_id = checkpoint_by_result[path]
            source = (
                scratch_procedure_record(evaluation.model_origin["initialization"])
                if checkpoint_id is None
                else training_procedure_record(training_by_checkpoint[checkpoint_id])
            )
            source_procedures.add(canonical_json_bytes(source))
        if len(kinds) != 1 or len(procedures) != 1 or len(source_procedures) != 1:
            raise ValueError(f"arm {arm_name!r} changes model or training/adaptation procedure.")
        contract = ArmContract(
            arm_name=arm_name,
            model_origin_kind=next(iter(kinds)),
            source_training_procedure=json.loads(next(iter(source_procedures))),
            adaptation_procedure=json.loads(next(iter(procedures))),
            allowed_variation_axes=("training_replicate", "participant_partition"),
        )
        arm_contracts[arm_name] = {
            "contract": contract.record(),
            "digest": contract.digest(),
        }

    replicates = tuple(sorted({str(item[1].get("training_replicate_key")) for item in loaded}))
    partitions = tuple(sorted({str(item[1].get("partition_key")) for item in loaded}))
    contrasts = tuple(tuple(str(value) for value in pair) for pair in planned_contrasts)
    power_digest = (
        semantic_sha256(power_plan_contract)
        if inference_scope == "training_procedure" and power_plan_contract is not None
        else None
    )
    design = StatisticalDesignContract(
        inference_scope=inference_scope,
        participant_cluster_key="participant_key",
        training_replicate_keys=replicates,
        partition_keys=partitions,
        planned_contrasts=contrasts,
        power_plan_digest=power_digest,
    )
    aggregation = (
        "mean_across_declared_frozen_models"
        if inference_scope == "conditional_frozen_models"
        else "independent_training_replicates"
    )
    runs = []
    for path, result, evaluation, plan in loaded:
        replicate = str(result["training_replicate_key"])
        partition = str(result["partition_key"])
        runs.append(
            {
                "run_id": f"{evaluation.arm_name}|{replicate}|{partition}",
                "result_path": _relative(path, output_path.parent),
                "result_sha256": _sha256(path),
                "training_replicate_key": replicate,
                "partition_key": partition,
                "checkpoint_id": checkpoint_by_result[path],
                "evaluation_contract": evaluation.record(),
                "evaluation_contract_digest": evaluation.digest(),
                "decision_plan": plan.record(),
                "decision_plan_digest": plan.digest(),
            }
        )
    participants = tuple(
        sorted({key for _, _, _, plan in loaded for key in plan.requested_participant_keys})
    )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "source_snapshot": {
            "manifest_path": _relative(source_manifest_path, output_path.parent),
            "manifest_sha256": _sha256(source_manifest_path),
        },
        "target_dataset": {
            "target_cache_sha256": next(iter(target_cache_values)),
            "target_identity_digest": next(iter(target_identity_values)),
            "requested_participant_keys": list(participants),
        },
        "metric": {
            "name": "requested_participant_operational_hit_rate",
            "evidence_level": evidence_level,
            "replicate_aggregation": aggregation,
        },
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
        },
        "checkpoint_registry": checkpoint_registry,
        "arm_contracts": arm_contracts,
        "statistical_design": design.record(),
        "statistical_design_digest": design.digest(),
        "runs": runs,
    }
    if power_plan_contract is not None:
        manifest["power_plan_contract"] = dict(power_plan_contract)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument(
        "--inference-scope",
        choices=("conditional_frozen_models", "training_procedure"),
        required=True,
    )
    parser.add_argument("--planned-contrast", nargs=2, action="append", required=True)
    parser.add_argument("--evidence-level", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--power-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    power = _read_json(args.power_plan, "power plan") if args.power_plan else None
    build_manifest(
        result_paths=args.result,
        source_snapshot_manifest=args.source_snapshot_manifest,
        inference_scope=args.inference_scope,
        planned_contrasts=args.planned_contrast,
        output=args.output,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        evidence_level=args.evidence_level,
        power_plan_contract=power,
    )


if __name__ == "__main__":
    main()
