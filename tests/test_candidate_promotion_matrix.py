from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.channel import build_channel_identity
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.domain import SOURCE_DOMAIN_AXIS_SCHEMA, SOURCE_DOMAIN_COLUMN
from data.epochs import EpochDataset, preprocessing_spec_from_contract, save_epoch_dataset
from data.events import observed_only_timeline
from data.identity import DatasetIdentityTable, ParticipantIdentityRecord, origin_subject_key
from experiments import run_candidate_promotion_matrix as matrix
from research.contracts import MIN_PROMOTION_HIT_AT_R


def _plan_record(*, output_root: str = "outputs") -> dict[str, Any]:
    return {
        "schema": matrix.PLAN_SCHEMA,
        "stage": "development",
        "source_snapshot_manifest": "source-freeze.manifest.json",
        "target_cache": "target.npz",
        "target_subjects": [
            {
                "target_subject": "TARGET::01",
                "source_holdout_subject": "SOURCE::01",
                "partition_key": "p01",
            },
            {
                "target_subject": "TARGET::02",
                "source_holdout_subject": "SOURCE::02",
                "partition_key": "p02",
            },
        ],
        "source_arms": [
            {
                "name": "source_a",
                "cache": "source-a.npz",
                "source_domain_mass": None,
                "risk_within_domain_unit": None,
                "selection_domain": None,
            },
            {
                "name": "source_ab",
                "cache": "source-ab.npz",
                "source_domain_mass": {"A": 0.8, "B": 0.2},
                "risk_within_domain_unit": "participant",
                "selection_domain": "A",
            },
        ],
        "training_replicates": [
            {"key": "rep01", "seed": 101},
            {"key": "rep02", "seed": 202},
        ],
        "evaluation_arms": [
            {
                "name": "zero_shot",
                "head": "zero_shot",
                "normalization": "source",
                "epoch_selection": "fixed_budget",
                "epochs": None,
                "batch_size": None,
                "lr": None,
                "target_stat_weight": 0.0,
                "fold_local_qc": False,
            },
            {
                "name": "linear_time_split",
                "head": "linear",
                "normalization": "source",
                "epoch_selection": "target_time_split",
                "epochs": 30,
                "batch_size": 64,
                "lr": 1e-3,
                "target_stat_weight": 0.0,
                "fold_local_qc": False,
            },
        ],
        "training": {
            "pooling_mode": "full_unfold",
            "temporal_kernel_size": 35,
            "epochs": 2,
            "batch_size": 16,
            "precision": "auto",
        },
        "calibration_selections": 5,
        "test_reps": 8,
        "identity_exclusion_policy": "source_or_global",
        "output_root": output_root,
        "statistical_design": {
            "inference_scope": "conditional_frozen_models",
            "planned_contrasts": [
                ["source_a__zero_shot", "source_ab__zero_shot"],
                ["source_a__linear_time_split", "source_ab__linear_time_split"],
            ],
            "evidence_level": 8,
            "bootstrap_iterations": 100,
            "bootstrap_seed": 303,
            "confidence_level": 0.95,
        },
    }


def _single_multidomain_arm_record() -> dict[str, Any]:
    record = _plan_record()
    record["source_arms"] = [record["source_arms"][1]]
    record["statistical_design"]["planned_contrasts"] = [
        ["source_ab__linear_time_split", "source_ab__zero_shot"]
    ]
    return record


def _write_plan(tmp_path: Path, record: dict[str, Any]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _load(tmp_path: Path, record: dict[str, Any] | None = None) -> matrix.PromotionPlan:
    return matrix.load_plan(_write_plan(tmp_path, record or _plan_record()))


def _cache_dataset(
    *,
    name: str,
    local_subjects: tuple[str, ...],
    authority_subjects: tuple[str, ...],
    position_offset: float = 0.0,
    source_domains: tuple[str, ...] | None = None,
    data_alignment: dict[str, Any] | None = None,
) -> EpochDataset:
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    positions = np.asarray(identity.coords, dtype=float).copy()
    positions[0, 0] += position_offset
    subjects = np.asarray(local_subjects)
    if source_domains is not None and len(source_domains) != len(subjects):
        raise ValueError("source_domains must align with subjects.")
    timeline = observed_only_timeline(
        dataset_id=name,
        subject_ids=subjects,
        stimulus_ids=np.arange(len(subjects), dtype=np.int64),
        onset_times_s=np.arange(len(subjects), dtype=float),
        evidence_available_times_s=np.arange(len(subjects), dtype=float) + 0.8,
        group_ids=subjects,
        online_causal=True,
        timing_source="promotion_preflight_fixture",
    )
    return EpochDataset(
        name=name,
        X=np.zeros(
            (
                len(subjects),
                len(identity.names),
                SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times,
            ),
            dtype=np.float32,
        ),
        y=np.zeros(len(subjects), dtype=np.int64),
        subject_ids=subjects,
        channel_names=identity.names,
        channel_positions_m=positions,
        channel_mask=np.ones(len(identity.names), dtype=bool),
        preprocessing=preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT),
        event_timeline=timeline,
        metadata=(
            pd.DataFrame(
                {
                    "subject": subjects,
                    SOURCE_DOMAIN_COLUMN: np.asarray(source_domains),
                }
            )
            if source_domains is not None
            else pd.DataFrame({"subject": subjects})
        ),
        provenance={
            "source": "unit_test",
            "source_reference": "average",
            **(
                {
                    "source_domain_axis": {
                        "schema": SOURCE_DOMAIN_AXIS_SCHEMA,
                        "column": SOURCE_DOMAIN_COLUMN,
                        "domains": sorted(set(source_domains)),
                    }
                }
                if source_domains is not None
                else {}
            ),
            **({"data_alignment": data_alignment} if data_alignment is not None else {}),
        },
        identity_table=DatasetIdentityTable(
            tuple(
                ParticipantIdentityRecord(
                    local_subject_id=local,
                    origin_subject_keys=(origin_subject_key("shared", authority),),
                    identity_status="source_verified",
                )
                for local, authority in zip(local_subjects, authority_subjects, strict=True)
            )
        ),
    )


def _write_preflight_inputs(
    tmp_path: Path,
    *,
    identity_mismatch: bool = False,
    geometry_mismatch: bool = False,
) -> None:
    source_member = tmp_path / "source.txt"
    source_member.write_text("source snapshot", encoding="utf-8")
    source_archive = tmp_path / "source.tar.gz"
    with tarfile.open(source_archive, "w:gz") as archive:
        archive.add(source_member, arcname="source.txt")
    (tmp_path / "source-freeze.manifest.json").write_text(
        json.dumps(
            {
                "schema": "n2p3_source_freeze/1",
                "archive": source_archive.name,
                "archive_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
                "source_commit": "a" * 40,
                "member_count": 1,
                "byte_size": source_archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    target_subjects = ("TARGET::01", "TARGET::02")
    source_subjects = ("SOURCE::01", "SOURCE::02")
    authority_subjects = ("01", "02")
    save_epoch_dataset(
        tmp_path / "target.npz",
        _cache_dataset(
            name="target",
            local_subjects=target_subjects,
            authority_subjects=authority_subjects,
        ),
    )
    for index, cache_name in enumerate(("source-a.npz", "source-ab.npz")):
        source_authorities = list(authority_subjects)
        if identity_mismatch and index == 0:
            source_authorities[0] = "different-person"
        save_epoch_dataset(
            tmp_path / cache_name,
            _cache_dataset(
                name=cache_name,
                local_subjects=source_subjects,
                authority_subjects=tuple(source_authorities),
                position_offset=(0.001 if geometry_mismatch and index == 0 else 0.0),
                source_domains=(("A", "B") if index == 1 else None),
            ),
        )


def test_plan_rejects_hit_at_r_below_promotion_floor(tmp_path: Path) -> None:
    record = _plan_record()
    record["test_reps"] = 7
    record["statistical_design"]["evidence_level"] = 7
    with pytest.raises(ValueError, match="at least 8"):
        _load(tmp_path, record)


def test_omitted_evidence_floor_keeps_promotion_default(tmp_path: Path) -> None:
    """Predeclared plans that never mention the floor keep the promotion depth."""

    assert "evidence_floor" not in _plan_record()
    assert _load(tmp_path).evidence_floor == MIN_PROMOTION_HIT_AT_R


def test_plan_accepts_declared_evidence_floor_below_promotion_default(tmp_path: Path) -> None:
    """A development estimand may target a shallower hit@R, but must say so."""

    record = _plan_record()
    record["evidence_floor"] = 2
    record["test_reps"] = 2
    record["statistical_design"]["evidence_level"] = 2
    plan = _load(tmp_path, record)
    assert plan.evidence_floor == 2
    assert plan.test_reps == 2
    assert plan.statistical_design.evidence_level == 2


def test_plan_rejects_evidence_floor_below_one(tmp_path: Path) -> None:
    for floor in (0, -3):
        record = _plan_record()
        record["evidence_floor"] = floor
        with pytest.raises(ValueError, match="at least 1"):
            _load(tmp_path, record)



def test_plan_metric_name_defaults_requested_and_validates(tmp_path: Path) -> None:
    legacy = _load(tmp_path, _plan_record())
    assert legacy.metric_name == "requested_participant_operational_hit_rate"

    record = _plan_record()
    record["metric_name"] = "valid_subject_operational_hit_rate"
    plan = _load(tmp_path, record)
    assert plan.metric_name == "valid_subject_operational_hit_rate"
    assert "metric_name" in plan.raw_record

    record["metric_name"] = "made_up_metric"
    with pytest.raises(ValueError, match="metric_name must be one of"):
        _load(tmp_path, record)

def test_plan_rejects_non_integer_evidence_floor(tmp_path: Path) -> None:
    for floor in (True, 2.5, "2"):
        record = _plan_record()
        record["evidence_floor"] = floor
        with pytest.raises(ValueError, match="must be an integer"):
            _load(tmp_path, record)


def test_plan_rejects_test_reps_below_declared_floor(tmp_path: Path) -> None:
    record = _plan_record()
    record["evidence_floor"] = 4
    record["test_reps"] = 3
    record["statistical_design"]["evidence_level"] = 3
    with pytest.raises(ValueError, match="at least 4"):
        _load(tmp_path, record)


def test_plan_rejects_evidence_level_below_declared_floor(tmp_path: Path) -> None:
    record = _plan_record()
    record["evidence_floor"] = 4
    record["test_reps"] = 8
    record["statistical_design"]["evidence_level"] = 2
    with pytest.raises(ValueError, match="at least 4"):
        _load(tmp_path, record)


def test_plan_rejects_evidence_level_above_test_reps(tmp_path: Path) -> None:
    record = _plan_record()
    record["evidence_floor"] = 2
    record["test_reps"] = 4
    record["statistical_design"]["evidence_level"] = 5
    with pytest.raises(ValueError, match=r"2 <= R <= 4 \(got 5\)"):
        _load(tmp_path, record)


def test_training_rejects_even_kernel_but_accepts_explicit_k65(tmp_path: Path) -> None:
    even = _plan_record()
    even["training"]["temporal_kernel_size"] = 34
    with pytest.raises(ValueError, match="must be odd"):
        _load(tmp_path, even)

    broad = _plan_record()
    broad["training"]["temporal_kernel_size"] = 65
    assert _load(tmp_path, broad).training.temporal_kernel_size == 65


def test_training_precision_is_explicit_and_validated(tmp_path: Path) -> None:
    record = _plan_record()
    record["training"]["precision"] = "bf16"
    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoint = next(task for task in dag.tasks if task.kind == "checkpoint_batch")

    assert checkpoint.argv[checkpoint.argv.index("--precision") + 1] == "bf16"

    record["training"]["precision"] = "tf32"
    with pytest.raises(ValueError, match="training.precision"):
        _load(tmp_path, record)


def test_training_forwards_feature_transfer_contract_to_pretraining(tmp_path: Path) -> None:
    record = _plan_record()
    record["source_arms"][1]["source_domain_mass"] = None
    record["source_arms"][1]["risk_within_domain_unit"] = None
    record["source_arms"][1]["selection_domain"] = None
    record["training"].update(
        {
            "feature_transfer_invariant_bottleneck": 2,
            "feature_transfer_private_channels": 3,
            "feature_transfer_target_bottleneck": 2,
            "feature_transfer_target_kernel": 5,
            "feature_transfer_max_invariant_residual": 0.2,
            "feature_transfer_max_target_residual": 0.3,
            "feature_alignment_weight": 0.1,
            "private_domain_weight": 0.05,
            "orthogonality_weight": 0.01,
        }
    )
    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoint = next(task for task in dag.tasks if task.kind == "checkpoint_batch")
    assert checkpoint.argv[checkpoint.argv.index("--feature-transfer-invariant-bottleneck") + 1] == "2"
    assert checkpoint.argv[checkpoint.argv.index("--feature-transfer-private-channels") + 1] == "3"
    assert checkpoint.argv[checkpoint.argv.index("--feature-transfer-target-kernel") + 1] == "5"
    assert checkpoint.argv[checkpoint.argv.index("--feature-alignment-weight") + 1] == "0.1"
    assert checkpoint.argv[checkpoint.argv.index("--private-domain-weight") + 1] == "0.05"
    assert plan.source_arms[1].selection_domain is None


def test_training_forwards_conditional_rbf_mmd_alignment_mode(tmp_path: Path) -> None:
    record = _plan_record()
    record["source_arms"][1]["source_domain_mass"] = None
    record["source_arms"][1]["risk_within_domain_unit"] = None
    record["source_arms"][1]["selection_domain"] = None
    record["training"].update(
        {
            "feature_transfer_invariant_bottleneck": 2,
            "feature_alignment_weight": 0.1,
            "feature_alignment_mode": "conditional_rbf_mmd",
        }
    )

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoint = next(task for task in dag.tasks if task.kind == "checkpoint_batch")

    assert plan.training.feature_alignment_mode == "conditional_rbf_mmd"
    assert checkpoint.argv[checkpoint.argv.index("--feature-alignment-mode") + 1] == (
        "conditional_rbf_mmd"
    )


def test_plan_round_trips_named_splitbn_and_source_residual_contract(
    tmp_path: Path,
) -> None:
    record = _single_multidomain_arm_record()
    record["source_arms"][0]["splitbn_source_weights"] = {"B": 0.7, "A": 0.3}
    record["training"].update(
        {
            "feature_transfer_invariant_bottleneck": 2,
            "splitbn_target_buffer_source": "source_weighted",
            "source_residual_drop_p": 0.2,
            "source_residual_drop_mode": "batch",
        }
    )

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoint = next(task for task in dag.tasks if task.kind == "checkpoint_batch")

    assert plan.raw_record == record
    assert plan.training.splitbn_target_buffer_source == "source_weighted"
    assert plan.training.source_residual_drop_p == 0.2
    assert plan.training.source_residual_drop_mode == "batch"
    assert dict(plan.source_arms[0].splitbn_source_weights) == {"A": 0.3, "B": 0.7}
    assert checkpoint.argv[
        checkpoint.argv.index("--splitbn-target-buffer-source") + 1
    ] == "source_weighted"
    weights = [
        checkpoint.argv[index + 1]
        for index, value in enumerate(checkpoint.argv)
        if value == "--splitbn-source-weight"
    ]
    assert weights == ["A=0.3", "B=0.7"]
    assert checkpoint.argv[
        checkpoint.argv.index("--source-residual-drop-p") + 1
    ] == "0.2"
    assert checkpoint.argv[
        checkpoint.argv.index("--source-residual-drop-mode") + 1
    ] == "batch"


def test_plan_round_trips_source_selection_record_and_forwards_to_pretraining(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    record["source_arms"][0]["source_selection_record"] = "selection/source-a.json"

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)

    expected = (tmp_path / "selection" / "source-a.json").resolve()
    assert plan.source_arms[0].source_selection_record == expected
    assert plan.source_arms[1].source_selection_record is None
    selected = next(
        task for task in dag.tasks if task.task_id == "checkpoint_batch/source_a/rep01"
    )
    unselected = next(
        task for task in dag.tasks if task.task_id == "checkpoint_batch/source_ab/rep01"
    )
    assert selected.argv[selected.argv.index("--source-selection-record") + 1] == str(expected)
    assert "--source-selection-record" not in unselected.argv

    selected_result_batch = next(
        task
        for task in dag.tasks
        if task.task_id == "result_batch/source_a/rep01/p01"
    )
    unselected_result_batch = next(
        task
        for task in dag.tasks
        if task.task_id == "result_batch/source_ab/rep01/p01"
    )
    for job in dag.result_batch_job_records[selected_result_batch.task_id]["jobs"]:
        argv = list(job["argv"])
        assert argv[argv.index("--source-selection-record") + 1] == str(expected)
    for job in dag.result_batch_job_records[unselected_result_batch.task_id]["jobs"]:
        assert "--source-selection-record" not in job["argv"]


def test_plan_rejects_invalid_source_selection_record(tmp_path: Path) -> None:
    record = _plan_record()
    record["source_arms"][0]["source_selection_record"] = 7

    with pytest.raises(ValueError, match="source_selection_record must be a non-empty string"):
        _load(tmp_path, record)



def test_default_training_omits_p0_p1_cli_flags(tmp_path: Path) -> None:
    plan = _load(tmp_path)
    checkpoint = next(
        task for task in matrix.build_dag(plan).tasks if task.kind == "checkpoint_batch"
    )

    assert plan.training.splitbn_target_buffer_source is None
    assert plan.training.source_residual_drop_p is None
    assert plan.training.source_residual_drop_mode is None
    assert not {
        "--splitbn-target-buffer-source",
        "--splitbn-source-weight",
        "--source-residual-drop-p",
        "--source-residual-drop-mode",
    } & set(checkpoint.argv)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("unknown_buffer", "splitbn_target_buffer_source"),
        ("missing_weights", "requires domain-keyed"),
        ("positional_weights", "must be a JSON mapping"),
        ("inactive_weights", "only valid"),
        ("weight_sum", "must sum to one"),
        ("weight_domain_mismatch", "source_domain_mass keys"),
        ("drop_gap", "preregistered"),
        ("drop_without_feature_transfer", "feature_transfer_invariant_bottleneck"),
        ("drop_without_mode", "requires an explicit"),
        ("mode_without_drop", "requires source_residual_drop_p"),
        ("unknown_drop_mode", "source_residual_drop_mode"),
    ],
)
def test_plan_rejects_invalid_p0_p1_combinations(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    record = _single_multidomain_arm_record()
    arm = record["source_arms"][0]
    training = record["training"]
    if mutation == "unknown_buffer":
        training["splitbn_target_buffer_source"] = "implicit_target"
    elif mutation == "missing_weights":
        training["splitbn_target_buffer_source"] = "source_weighted"
    elif mutation == "positional_weights":
        training["splitbn_target_buffer_source"] = "source_weighted"
        arm["splitbn_source_weights"] = [0.5, 0.5]
    elif mutation == "inactive_weights":
        arm["splitbn_source_weights"] = {"A": 0.5, "B": 0.5}
    elif mutation == "weight_sum":
        training["splitbn_target_buffer_source"] = "source_weighted"
        arm["splitbn_source_weights"] = {"A": 0.6, "B": 0.5}
    elif mutation == "weight_domain_mismatch":
        training["splitbn_target_buffer_source"] = "source_weighted"
        arm["splitbn_source_weights"] = {"A": 0.5, "C": 0.5}
    elif mutation == "drop_gap":
        training.update(
            {
                "feature_transfer_invariant_bottleneck": 2,
                "source_residual_drop_p": 0.1,
                "source_residual_drop_mode": "sample",
            }
        )
    elif mutation == "drop_without_feature_transfer":
        training.update(
            {
                "source_residual_drop_p": 0.2,
                "source_residual_drop_mode": "sample",
            }
        )
    elif mutation == "drop_without_mode":
        training.update(
            {
                "feature_transfer_invariant_bottleneck": 2,
                "source_residual_drop_p": 0.2,
            }
        )
    elif mutation == "mode_without_drop":
        training["source_residual_drop_mode"] = "sample"
    else:
        training.update(
            {
                "feature_transfer_invariant_bottleneck": 2,
                "source_residual_drop_p": 0.2,
                "source_residual_drop_mode": "epoch",
            }
        )

    with pytest.raises(ValueError, match=match):
        _load(tmp_path, record)


@pytest.mark.parametrize("drop_p", [0.0, 0.2, 0.5])
def test_plan_accepts_preregistered_source_residual_drop_arms(
    tmp_path: Path,
    drop_p: float,
) -> None:
    record = _single_multidomain_arm_record()
    record["training"].update(
        {
            "feature_transfer_invariant_bottleneck": 2,
            "source_residual_drop_p": drop_p,
            "source_residual_drop_mode": "sample",
        }
    )

    plan = _load(tmp_path, record)

    assert plan.training.source_residual_drop_p == drop_p


@pytest.mark.parametrize("buffer_source", ["pooled_source", "target_calibration"])
def test_plan_accepts_unweighted_splitbn_buffer_sources(
    tmp_path: Path,
    buffer_source: str,
) -> None:
    record = _single_multidomain_arm_record()
    record["training"]["splitbn_target_buffer_source"] = buffer_source

    plan = _load(tmp_path, record)
    checkpoint = next(
        task for task in matrix.build_dag(plan).tasks if task.kind == "checkpoint_batch"
    )

    assert plan.training.splitbn_target_buffer_source == buffer_source
    assert "--splitbn-source-weight" not in checkpoint.argv


def test_cloud_example_is_explicitly_row_column_development_not_brainsync() -> None:
    example = (
        Path(matrix.__file__).resolve().parent.parent
        / "configs"
        / "feature_transfer_cloud_20260902.example.json"
    )
    raw = example.read_text(encoding="utf-8")
    plan = matrix.load_plan(example)

    assert "ROW_COLUMN" in raw
    assert "PLACEHOLDER" in raw
    assert "brainsync" not in raw.casefold()
    assert plan.training.splitbn_target_buffer_source == "pooled_source"
    assert plan.training.source_residual_drop_p == 0.3



def test_domain_transfer_v2_ablation_example_parses_new_fields() -> None:
    example = (
        Path(matrix.__file__).resolve().parent.parent
        / "configs"
        / "domain_transfer_v2_ablation.example.json"
    )
    plan = matrix.load_plan(example)

    assert len(plan.source_arms) == 5
    assert plan.source_arms[0].euclidean_alignment is not None
    assert plan.source_arms[0].euclidean_alignment.mode == "none"
    assert plan.source_arms[0].sampler_kind == "uniform"
    assert plan.source_arms[2].euclidean_alignment is not None
    assert plan.source_arms[2].euclidean_alignment.mode == "target_calibration"
    assert plan.source_arms[2].euclidean_alignment.reference_mode == "native_preserve"
    assert plan.source_arms[3].sampler_kind == "target_similarity_diverse"
    assert plan.source_arms[3].source_selection_record is not None
    assert plan.source_arms[4].sampler_kind == "dense_pds"
    assert all(arm.as_mmd_lambda0 == 0.0 for arm in plan.evaluation_arms)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_domain_mass", {"A": 0.7, "B": 0.2}, "summing to one"),
        ("risk_within_domain_unit", "row_repeat", "epoch or participant"),
        ("selection_domain", "UNKNOWN", "weighted domain"),
    ],
)
def test_plan_rejects_invalid_multisource_risk(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    record = _plan_record()
    record["source_arms"][1][field] = value

    with pytest.raises(ValueError, match=match):
        _load(tmp_path, record)


def test_plan_supports_natural_ce_with_target_domain_selection(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["source_arms"][1]
    arm["source_domain_mass"] = None
    arm["risk_within_domain_unit"] = None
    arm["selection_domain"] = "A"

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoint = next(
        task for task in dag.tasks if task.task_id == "checkpoint_batch/source_ab/rep01"
    )

    assert "--source-domain-mass" not in checkpoint.argv
    assert "--risk-within-domain-unit" not in checkpoint.argv
    assert checkpoint.argv[checkpoint.argv.index("--selection-domain") + 1] == "A"



def test_plan_supports_candidate_loss_ablation_interface(tmp_path: Path) -> None:
    record = _plan_record()
    record["evaluation_arms"][1]["candidate_loss_weight"] = 0.25
    record["evaluation_arms"][1]["candidate_loss_temperature"] = 2.0
    plan = _load(tmp_path, record)
    assert plan.evaluation_arms[1].candidate_loss_weight == 0.25
    assert plan.evaluation_arms[1].candidate_loss_temperature == 2.0
    dag = matrix.build_dag(plan)
    batch_task = next(
        task
        for task in dag.tasks
        if task.task_id == "result_batch/source_a/rep01/p01"
    )
    argv = next(
        list(job["argv"])
        for job in dag.result_batch_job_records[batch_task.task_id]["jobs"]
        if job["arm_name"] == "source_a__linear_time_split"
    )
    assert "--candidate-loss-weight" in argv
    assert "--candidate-loss-temperature" in argv
    assert argv[argv.index("--candidate-loss-weight") + 1] == "0.25"
    assert argv[argv.index("--candidate-loss-temperature") + 1] == "2.0"


def test_plan_rejects_candidate_loss_on_zero_shot_arm(tmp_path: Path) -> None:
    record = _plan_record()
    record["evaluation_arms"][0]["candidate_loss_weight"] = 0.1
    with pytest.raises(ValueError, match="candidate_loss_weight must be 0"):
        _load(tmp_path, record)


@pytest.mark.parametrize("head", ["linear", "mlp16", "classifier_fine", "full_fine", "adapter"])
def test_plan_supports_current_fitted_runner_heads(tmp_path: Path, head: str) -> None:
    record = _plan_record()
    record["evaluation_arms"][1]["head"] = head
    assert _load(tmp_path, record).evaluation_arms[1].head == head


def test_plan_supports_checkpoint_deterministic_auto_head(tmp_path: Path) -> None:
    record = _plan_record()
    record["evaluation_arms"][0]["head"] = "auto"
    assert _load(tmp_path, record).evaluation_arms[0].head == "auto"


@pytest.mark.parametrize(
    ("arm_index", "field", "value", "match"),
    [
        (0, "normalization", "target_prefix", "zero-shot combination"),
        (0, "epoch_selection", "target_time_split", "zero-shot combination"),
        (0, "epochs", 1, "zero-shot combination"),
        (0, "batch_size", 64, "zero-shot combination"),
        (0, "lr", 1e-3, "zero-shot combination"),
        (0, "fold_local_qc", True, "zero-shot combination"),
        (1, "epochs", None, "requires explicit values"),
        (1, "target_stat_weight", 0.25, "unless normalization is shrinkage"),
    ],
)
def test_plan_rejects_invalid_evaluation_arm_combinations(
    tmp_path: Path,
    arm_index: int,
    field: str,
    value: object,
    match: str,
) -> None:
    record = _plan_record()
    record["evaluation_arms"][arm_index][field] = value
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, record)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate_target", "target subject mappings"),
        ("duplicate_holdout", "source holdout mappings"),
        ("missing_holdout", "missing required fields"),
    ],
)
def test_plan_rejects_ambiguous_or_missing_target_holdout_mapping(
    tmp_path: Path, mutation: str, match: str
) -> None:
    record = _plan_record()
    targets = record["target_subjects"]
    if mutation == "duplicate_target":
        targets[1]["target_subject"] = targets[0]["target_subject"]
    elif mutation == "duplicate_holdout":
        targets[1]["source_holdout_subject"] = targets[0]["source_holdout_subject"]
    else:
        del targets[0]["source_holdout_subject"]
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, record)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [("arm", "source arm names"), ("seed", "training replicate seeds")],
)
def test_plan_rejects_duplicate_arm_or_seed(tmp_path: Path, mutation: str, match: str) -> None:
    record = _plan_record()
    if mutation == "arm":
        record["source_arms"][1]["name"] = record["source_arms"][0]["name"]
    else:
        record["training_replicates"][1]["seed"] = record["training_replicates"][0]["seed"]
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, record)


def test_plan_rejects_output_component_escape(tmp_path: Path) -> None:
    record = _plan_record()
    record["target_subjects"][0]["partition_key"] = "../../outside"
    with pytest.raises(ValueError, match="portable path component"):
        _load(tmp_path, record)


def test_plan_rejects_output_root_escape_from_explicit_root(tmp_path: Path) -> None:
    record = _plan_record(output_root="../outside")
    plan_path = _write_plan(tmp_path, record)
    with pytest.raises(ValueError, match="explicit --root boundary"):
        matrix.load_plan(plan_path, root=tmp_path)


def test_dry_run_emits_full_matrix_without_writing_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _write_plan(tmp_path, _plan_record())
    resolved_inputs = {
        "source_snapshot": {"archive_sha256": "a" * 64},
        "digest": "b" * 64,
    }
    monkeypatch.setattr(matrix, "_verify_inputs", lambda _dag: resolved_inputs)
    matrix.main(["--plan", str(plan_path), "--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert output["task_counts"] == {
        "checkpoint_batch": 4,
        "result": 0,
        "result_batch": 8,
        "manifest": 1,
        "analysis": 1,
    }
    assert output["task_count"] == 14
    assert output["resolved_inputs"] == resolved_inputs
    assert all(task["argv"][0] == "$PYTHON" for task in output["tasks"])
    assert len(output["result_batch_records"]) == 8
    batch_record = next(
        record["record"]
        for record in output["result_batch_records"].values()
        if record["record"]["job_count"] == 2
    )
    assert batch_record["jobs"][0]["argv"][0] == "--dataset-cache"
    assert not (tmp_path / "outputs").exists()


def test_dag_passes_explicit_time_split_linear_configuration(tmp_path: Path) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    batch_task = next(
        task
        for task in dag.tasks
        if task.task_id == "result_batch/source_a/rep01/p01"
    )
    assert batch_task.kind == "result_batch"
    assert dag.result_batch_job_paths[batch_task.task_id].is_absolute()
    jobs = {
        job["arm_name"]: list(job["argv"])
        for job in dag.result_batch_job_records[batch_task.task_id]["jobs"]
    }
    assert set(jobs) == {"source_a__zero_shot", "source_a__linear_time_split"}
    linear = jobs["source_a__linear_time_split"]
    expected = {
        "--head": "linear",
        "--normalization": "source",
        "--epoch-selection": "target_time_split",
        "--epochs": "30",
        "--batch-size": "64",
        "--lr": "0.001",
        "--target-stat-weight": "0.0",
    }
    for flag, value in expected.items():
        assert linear[linear.index(flag) + 1] == value
    assert "--no-fold-local-qc" in linear

    weighted_checkpoint = next(
        task
        for task in dag.tasks
        if task.task_id == "checkpoint_batch/source_ab/rep01"
    )
    checkpoint_argv = list(weighted_checkpoint.argv)
    masses = [
        checkpoint_argv[index + 1]
        for index, value in enumerate(checkpoint_argv)
        if value == "--source-domain-mass"
    ]
    assert masses == ["A=0.8", "B=0.2"]
    assert checkpoint_argv[checkpoint_argv.index("--risk-within-domain-unit") + 1] == "participant"
    assert checkpoint_argv[checkpoint_argv.index("--selection-domain") + 1] == "A"
    assert "--subject-prefix-repeat" not in checkpoint_argv

    zero_shot = jobs["source_a__zero_shot"]
    assert "--zero-shot-feature-cache" in zero_shot
    assert not {
        "--epochs",
        "--batch-size",
        "--lr",
        "--fold-local-qc",
        "--no-fold-local-qc",
        "--adapt-batchnorm",
        "--epoch-selection",
        "--target-stat-weight",
    } & set(zero_shot)


def test_result_batch_embedded_record_validates_ledger_and_member_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    batch_task = next(task for task in dag.tasks if task.kind == "result_batch")
    jobs_path = dag.result_batch_job_paths[batch_task.task_id]
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_record = json.loads(json.dumps(dag.result_batch_job_records[batch_task.task_id]))
    jobs_path.write_text(json.dumps(jobs_record), encoding="utf-8")
    outputs: list[Path] = []
    for job in jobs_record["jobs"]:
        output = Path(job["argv"][job["argv"].index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema": matrix.RESULT_SCHEMA,
                    "run_status": "completed",
                    "evaluation_contract": {"probe": True},
                    "evaluation_contract_digest": "a" * 64,
                    "decision_plan": {"probe": True},
                    "decision_plan_digest": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        outputs.append(output)

    class DummyEvaluation:
        def __init__(self, **_: Any) -> None:
            pass

        def digest(self) -> str:
            return "a" * 64

    monkeypatch.setattr(matrix, "EvaluationRunContract", DummyEvaluation)
    monkeypatch.setattr(
        matrix,
        "assert_contract_digest",
        lambda *_args, **_kwargs: None,
    )
    ledger_jobs = []
    for job, output in zip(jobs_record["jobs"], outputs, strict=True):
        ledger_jobs.append(
            {
                "arm_name": job["arm_name"],
                "output": str(output.resolve()),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "byte_size": output.stat().st_size,
                "evaluation_contract_digest": "a" * 64,
                "decision_plan_digest": "b" * 64,
            }
        )
    ledger = {
        "schema": matrix.RESULT_BATCH_LEDGER_SCHEMA,
        "completed": True,
        "jobs_file": str(jobs_path.resolve()),
        "jobs_file_sha256": hashlib.sha256(jobs_path.read_bytes()).hexdigest(),
        "job_count": len(ledger_jobs),
        "jobs": ledger_jobs,
    }
    batch_task.output.parent.mkdir(parents=True, exist_ok=True)
    batch_task.output.write_text(json.dumps(ledger), encoding="utf-8")
    embedded = matrix._embedded_artifact_record(batch_task)
    assert embedded["schema"] == matrix.RESULT_BATCH_LEDGER_SCHEMA
    assert embedded["job_count"] == len(outputs)
    assert embedded["jobs_file_sha256"] == ledger["jobs_file_sha256"]

    outputs[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Expecting value"):
        matrix._embedded_artifact_record(batch_task)


def _fake_artifact_attestation(task: matrix.MatrixTask) -> dict[str, Any]:
    payload = task.output.read_bytes()
    return {
        "path": task.output.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
        "embedded": {"schema": task.kind},
    }


def _fake_embedded(task: matrix.MatrixTask) -> dict[str, str]:
    return {"schema": task.kind}


def _output_from_argv(argv: list[str]) -> Path:
    if "--batch-record" in argv:
        flag = "--batch-record"
    elif "--checkpoint" in argv and "--output" not in argv:
        flag = "--checkpoint"
    else:
        flag = "--output"
    return Path(argv[argv.index(flag) + 1])


def test_resume_reruns_tampered_output_and_transitive_dependents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    calls: list[str] = []
    artifact_serial = 0

    def succeed(argv: list[str], **_: Any) -> None:
        nonlocal artifact_serial
        output = _output_from_argv(argv)
        output.parent.mkdir(parents=True, exist_ok=True)
        calls.append(str(output))
        artifact_serial += 1
        output.write_text(f"artifact-call-{artifact_serial}", encoding="utf-8")

    monkeypatch.setattr(matrix, "_artifact_attestation", _fake_artifact_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)
    matrix.execute_dag(dag, subprocess_runner=succeed, verify_inputs=False)
    assert len(calls) == 14
    first_checkpoint = dag.tasks[0].output
    first_checkpoint.write_text("tampered", encoding="utf-8")
    calls.clear()
    matrix.execute_dag(dag, resume=True, subprocess_runner=succeed, verify_inputs=False)
    assert len(calls) == 5
    assert str(first_checkpoint) in calls
    assert str(dag.tasks[-2].output) in calls
    assert str(dag.tasks[-1].output) in calls


def test_attested_inputs_are_bound_to_journal_and_cache_tamper_blocks_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_preflight_inputs(tmp_path)
    power_plan = tmp_path / "power.json"
    power_plan.write_text('{"planned":true}', encoding="utf-8")
    record = _plan_record()
    record["statistical_design"]["inference_scope"] = "training_procedure"
    record["statistical_design"]["power_plan"] = power_plan.name
    dag = matrix.build_dag(_load(tmp_path, record))
    calls: list[str] = []

    def succeed(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"artifact-{len(calls)}", encoding="utf-8")
        calls.append(str(output))

    monkeypatch.setattr(matrix, "_artifact_attestation", _fake_artifact_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)
    matrix.execute_dag(dag, subprocess_runner=succeed)
    journal = json.loads(
        (dag.plan.output_root / "promotion.journal.json").read_text(encoding="utf-8")
    )
    resolved = journal["resolved_inputs"]
    assert resolved["source_snapshot"]["manifest_sha256"]
    assert resolved["target_cache"]["cache_sha256"]
    assert resolved["target_cache"]["decoded_dataset_sha256"]
    assert resolved["target_cache"]["verified_cache_attestation"]["sha256"] == (
        resolved["target_cache"]["cache_sha256"]
    )
    assert set(resolved["source_caches"]) == {"source_a", "source_ab"}
    assert resolved["preflight_loading"]["task_count"] == 3
    assert resolved["preflight_loading"]["effective_workers"] == 3
    assert resolved["power_plan"]["sha256"] == hashlib.sha256(power_plan.read_bytes()).hexdigest()
    assert journal["dag_digest"] == matrix._dag_digest(dag, resolved)

    replacement = _cache_dataset(
        name="target",
        local_subjects=("TARGET::01", "TARGET::02"),
        authority_subjects=("01", "02"),
    )
    replacement.X[0, 0, 0] = 1.0
    save_epoch_dataset(tmp_path / "target.npz", replacement)
    calls.clear()
    with pytest.raises(ValueError, match="journal inputs changed"):
        matrix.execute_dag(dag, resume=True, subprocess_runner=succeed)
    assert calls == []


def test_preflight_deduplicates_shared_cache_and_loads_unique_paths_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_preflight_inputs(tmp_path)
    record = _plan_record()
    shared = record["source_arms"][1]
    shared["cache"] = "source-a.npz"
    shared["source_domain_mass"] = None
    shared["risk_within_domain_unit"] = None
    shared["selection_domain"] = None
    dag = matrix.build_dag(_load(tmp_path, record))
    original_load = matrix._load_attested_cache_input
    barrier = threading.Barrier(2)
    calls: list[Path] = []
    worker_names: list[str] = []

    def concurrent_load(cache_path, *, name, dag):
        calls.append(Path(cache_path).resolve())
        worker_names.append(threading.current_thread().name)
        barrier.wait(timeout=5)
        return original_load(cache_path, name=name, dag=dag)

    monkeypatch.setattr(matrix, "_load_attested_cache_input", concurrent_load)

    resolved = matrix._verify_inputs(dag)

    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert len(set(worker_names)) == 2
    assert all(name.startswith("promotion-preflight") for name in worker_names)
    assert resolved["preflight_loading"]["task_count"] == 2
    assert resolved["preflight_loading"]["effective_workers"] == 2
    assert (
        resolved["source_caches"]["source_a"]["cache_sha256"]
        == resolved["source_caches"]["source_ab"]["cache_sha256"]
    )


def test_splitbn_weights_are_independent_of_risk_mass_and_checked_against_cache(
    tmp_path: Path,
) -> None:
    _write_preflight_inputs(tmp_path)
    record = _single_multidomain_arm_record()
    arm = record["source_arms"][0]
    arm["source_domain_mass"] = None
    arm["risk_within_domain_unit"] = None
    arm["selection_domain"] = None
    arm["splitbn_source_weights"] = {"A": 0.4, "B": 0.6}
    record["training"]["splitbn_target_buffer_source"] = "source_weighted"

    plan = _load(tmp_path, record)
    resolved = matrix._verify_inputs(matrix.build_dag(plan))

    assert plan.source_arms[0].source_domain_mass == ()
    assert resolved["source_caches"]["source_ab"]["source_domains"] == ["A", "B"]

    arm["splitbn_source_weights"] = {"A": 0.4, "C": 0.6}
    mismatched = _load(tmp_path, record)
    with pytest.raises(ValueError, match="Split-BN weights disagree with cache domains"):
        matrix._verify_inputs(matrix.build_dag(mismatched))


def test_p1_without_selection_domain_uses_cache_domain_axis(tmp_path: Path) -> None:
    _write_preflight_inputs(tmp_path)
    record = _single_multidomain_arm_record()
    arm = record["source_arms"][0]
    arm["source_domain_mass"] = None
    arm["risk_within_domain_unit"] = None
    arm["selection_domain"] = None
    record["training"].update(
        {
            "feature_transfer_invariant_bottleneck": 2,
            "source_residual_drop_p": 0.5,
            "source_residual_drop_mode": "sample",
        }
    )

    dag = matrix.build_dag(_load(tmp_path, record))
    matrix._verify_inputs(dag)
    checkpoint = next(task for task in dag.tasks if task.kind == "checkpoint_batch")

    assert "--selection-domain" not in checkpoint.argv
    assert checkpoint.argv[
        checkpoint.argv.index("--source-residual-drop-p") + 1
    ] == "0.5"


@pytest.mark.parametrize("mechanism", ["splitbn", "source_residual"])
def test_p0_p1_preflight_rejects_cache_without_multidomain_axis(
    tmp_path: Path,
    mechanism: str,
) -> None:
    _write_preflight_inputs(tmp_path)
    record = _plan_record()
    record["source_arms"] = [record["source_arms"][0]]
    record["statistical_design"]["planned_contrasts"] = [
        ["source_a__linear_time_split", "source_a__zero_shot"]
    ]
    if mechanism == "splitbn":
        record["training"]["splitbn_target_buffer_source"] = "pooled_source"
    else:
        record["training"].update(
            {
                "feature_transfer_invariant_bottleneck": 2,
                "source_residual_drop_p": 0.2,
                "source_residual_drop_mode": "sample",
            }
        )

    with pytest.raises(ValueError, match="does not declare at least two source domains"):
        matrix._verify_inputs(matrix.build_dag(_load(tmp_path, record)))


def test_identity_mismatch_is_rejected_before_any_subprocess(tmp_path: Path) -> None:
    _write_preflight_inputs(tmp_path, identity_mismatch=True)
    dag = matrix.build_dag(_load(tmp_path))
    calls: list[list[str]] = []

    def must_not_run(argv: list[str], **_: Any) -> None:
        calls.append(argv)

    with pytest.raises(ValueError, match="different.*authority keys"):
        matrix.execute_dag(dag, subprocess_runner=must_not_run)
    assert calls == []


def test_channel_geometry_mismatch_is_rejected_before_any_subprocess(
    tmp_path: Path,
) -> None:
    _write_preflight_inputs(tmp_path, geometry_mismatch=True)
    dag = matrix.build_dag(_load(tmp_path))
    calls: list[list[str]] = []

    def must_not_run(argv: list[str], **_: Any) -> None:
        calls.append(argv)

    with pytest.raises(ValueError, match="channel_positions_m"):
        matrix.execute_dag(dag, subprocess_runner=must_not_run)
    assert calls == []


def test_resume_rejects_unattested_existing_output(tmp_path: Path) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    dag.tasks[0].output.parent.mkdir(parents=True)
    dag.tasks[0].output.write_bytes(b"orphan")
    with pytest.raises(FileExistsError, match="mere existence"):
        matrix.execute_dag(dag, resume=True, verify_inputs=False)


def test_fresh_batch_rejects_unattested_secondary_checkpoint(tmp_path: Path) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    checkpoint = matrix._checkpoint_batch_outputs(dag.tasks[0])[0]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"orphan checkpoint")

    with pytest.raises(FileExistsError, match="mere existence"):
        matrix.execute_dag(dag, verify_inputs=False)


def test_subprocess_failure_is_atomically_recorded_and_reraised(tmp_path: Path) -> None:
    dag = matrix.build_dag(_load(tmp_path))

    def fail(argv: list[str], **_: Any) -> None:
        raise subprocess.CalledProcessError(17, argv)

    with pytest.raises(subprocess.CalledProcessError):
        matrix.execute_dag(dag, subprocess_runner=fail, verify_inputs=False)
    journal = json.loads(
        (dag.plan.output_root / "promotion.journal.json").read_text(encoding="utf-8")
    )
    first = journal["tasks"][dag.tasks[0].task_id]
    assert journal["status"] == "failed"
    assert first["status"] == "failed"
    assert first["started_at"] and first["finished_at"]
    assert first["error"]["returncode"] == 17
    assert first["argv"][0] == "$PYTHON"


def test_parallel_scheduler_respects_limit_dependencies_and_main_thread_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    main_thread = threading.get_ident()
    task_by_output = {task.output.resolve(): task for task in dag.tasks}
    active = 0
    peak = 0
    calls: list[str] = []
    dependency_violations: list[tuple[str, str]] = []
    worker_threads: set[int] = set()
    guard = threading.Lock()
    # Deterministic overlap gate (replaces the load-sensitive 15 ms sleep
    # window): the first wave of dependency-free tasks rendezvous until two
    # are concurrently in flight, so ``peak >= 2`` holds by construction
    # whenever the scheduler truly dispatches concurrently. The gate is
    # sticky (``overlap_confirmed``) because pairing tasks leave immediately
    # and a bare in-flight counter would starve the earliest waiter; once
    # confirmed, everyone passes without waiting. If a rendezvous never
    # happens (scheduler regressed to serial dispatch) a single bounded
    # attempt sets ``overlap_expired`` so the final ``1 < peak`` assert
    # fails fast instead of stalling once per task or hanging.
    overlap_gate = threading.Condition()
    inflight = 0
    overlap_confirmed = False
    overlap_expired = False

    original_atomic = matrix._atomic_write_json
    original_dependencies = matrix._dependency_digests

    def main_thread_atomic(path: Path, value: object) -> None:
        assert threading.get_ident() == main_thread
        original_atomic(path, value)

    def main_thread_dependencies(task, state_tasks):
        assert threading.get_ident() == main_thread
        return original_dependencies(task, state_tasks)

    def main_thread_attestation(task):
        assert threading.get_ident() == main_thread
        return _fake_artifact_attestation(task)

    def succeed(argv: list[str], **_: Any) -> None:
        nonlocal active, peak, inflight, overlap_confirmed, overlap_expired
        output = _output_from_argv(argv).resolve()
        task = task_by_output[output]
        with guard:
            worker_threads.add(threading.get_ident())
            active += 1
            peak = max(peak, active)
            calls.append(task.task_id)
            for dependency in task.dependencies:
                dependency_task = dag.task_by_id[dependency]
                if not dependency_task.output.is_file():
                    dependency_violations.append((task.task_id, dependency))
        try:
            if not task.dependencies and not (overlap_confirmed or overlap_expired):
                with overlap_gate:
                    if not (overlap_confirmed or overlap_expired):
                        inflight += 1
                        overlap_gate.notify_all()
                        overlap_gate.wait_for(
                            lambda: overlap_confirmed or inflight >= 2,
                            timeout=10,
                        )
                        if inflight >= 2 or overlap_confirmed:
                            overlap_confirmed = True
                        else:
                            overlap_expired = True
                        inflight -= 1
                        overlap_gate.notify_all()
            output.write_text(f"artifact:{task.task_id}", encoding="utf-8")
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(matrix, "_atomic_write_json", main_thread_atomic)
    monkeypatch.setattr(matrix, "_dependency_digests", main_thread_dependencies)
    monkeypatch.setattr(matrix, "_artifact_attestation", main_thread_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)

    result = matrix.execute_dag(
        dag,
        subprocess_runner=succeed,
        verify_inputs=False,
        max_workers=4,
    )

    assert result == dag.tasks[-1].output
    assert 1 < peak <= 4
    assert worker_threads and main_thread not in worker_threads
    assert dependency_violations == []
    journal = json.loads(matrix._journal_path(dag).read_text(encoding="utf-8"))
    assert journal["status"] == "completed"
    assert journal["execution"]["max_workers"] == 4
    assert journal["execution"]["state_writer"] == "orchestrator_main_thread_only"
    assert journal["execution"]["orchestrator_lock"]["dag_digest"] == journal["dag_digest"]
    first_ready = [task.task_id for task in dag.tasks if not task.dependencies][:4]
    assert journal["dispatch_sequence"][:4] == first_ready
    assert set(journal["dispatch_sequence"]) == set(calls)
    assert len(journal["dispatch_sequence"]) == len(set(journal["dispatch_sequence"])) == 14
    assert all(record["status"] == "completed" for record in journal["tasks"].values())
    assert all(len(record["attempts"]) == 1 for record in journal["tasks"].values())
    for record in journal["tasks"].values():
        for log in record["attempts"][0]["logs"].values():
            log_path = dag.plan.output_root / Path(log["path"].removeprefix("$OUTPUT_ROOT/"))
            assert hashlib.sha256(log_path.read_bytes()).hexdigest() == log["sha256"]


def test_parallel_failure_stops_dispatch_drains_and_resumes_strictly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    task_by_output = {task.output.resolve(): task for task in dag.tasks}
    release_siblings = threading.Event()
    failure_journal_seen = threading.Event()
    initial_calls: list[str] = []
    journal_snapshot: dict[str, Any] = {}
    journal_path = dag.plan.output_root / "promotion.journal.json"

    def fail_first_and_block_siblings(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv).resolve()
        task = task_by_output[output]
        initial_calls.append(task.task_id)
        if task.task_id == dag.tasks[0].task_id:
            raise subprocess.CalledProcessError(19, argv)
        assert release_siblings.wait(timeout=10)
        output.write_text(f"initial:{task.task_id}", encoding="utf-8")

    def observe_failure() -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if journal_path.is_file():
                current = json.loads(journal_path.read_text(encoding="utf-8"))
                if current.get("status") == "failed":
                    journal_snapshot.update(current)
                    failure_journal_seen.set()
                    release_siblings.set()
                    return
            time.sleep(0.01)
        release_siblings.set()

    observer = threading.Thread(target=observe_failure, daemon=True)
    observer.start()
    monkeypatch.setattr(matrix, "_artifact_attestation", _fake_artifact_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)

    with pytest.raises(subprocess.CalledProcessError):
        matrix.execute_dag(
            dag,
            subprocess_runner=fail_first_and_block_siblings,
            verify_inputs=False,
            max_workers=4,
        )
    observer.join(timeout=10)

    assert failure_journal_seen.is_set()
    assert journal_snapshot["tasks"][dag.tasks[0].task_id]["status"] == "failed"
    first_ready = [task.task_id for task in dag.tasks if not task.dependencies][:4]
    assert set(initial_calls) == set(first_ready)
    failed_state = json.loads(matrix._journal_path(dag).read_text(encoding="utf-8"))
    assert failed_state["status"] == "failed"
    assert "running" not in {record["status"] for record in failed_state["tasks"].values()}
    assert set(failed_state["dispatch_sequence"]) == set(initial_calls)

    resume_calls: list[str] = []

    def succeed(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv).resolve()
        task = task_by_output[output]
        resume_calls.append(task.task_id)
        output.write_text(f"resumed:{task.task_id}", encoding="utf-8")

    matrix.execute_dag(
        dag,
        resume=True,
        subprocess_runner=succeed,
        verify_inputs=False,
        max_workers=4,
    )
    resumed = json.loads(matrix._journal_path(dag).read_text(encoding="utf-8"))
    assert resumed["status"] == "completed"
    assert dag.tasks[0].task_id in resume_calls
    initial_successes = set(initial_calls) - {dag.tasks[0].task_id}
    assert not initial_successes & set(resume_calls)
    assert len(resume_calls) == 11
    assert all(record["status"] == "completed" for record in resumed["tasks"].values())


def test_resume_marks_old_running_attempt_orphaned_and_invalidates_dependents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    task_by_output = {task.output.resolve(): task for task in dag.tasks}

    def succeed(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv)
        output.write_text(f"artifact:{output.name}", encoding="utf-8")

    monkeypatch.setattr(matrix, "_artifact_attestation", _fake_artifact_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)
    matrix.execute_dag(dag, subprocess_runner=succeed, verify_inputs=False, max_workers=3)

    journal_path = matrix._journal_path(dag)
    state = json.loads(journal_path.read_text(encoding="utf-8"))
    first = state["tasks"][dag.tasks[0].task_id]
    first["status"] = "running"
    first["finished_at"] = None
    first["attempts"][-1]["status"] = "running"
    first["attempts"][-1]["finished_at"] = None
    state["status"] = "running"
    journal_path.write_text(json.dumps(state), encoding="utf-8")
    calls: list[str] = []

    def resume_succeed(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv).resolve()
        calls.append(task_by_output[output].task_id)
        output.write_text(f"resume:{len(calls)}", encoding="utf-8")

    matrix.execute_dag(
        dag,
        resume=True,
        subprocess_runner=resume_succeed,
        verify_inputs=False,
        max_workers=3,
    )
    resumed = json.loads(journal_path.read_text(encoding="utf-8"))
    first = resumed["tasks"][dag.tasks[0].task_id]
    assert first["attempts"][-2]["status"] == "orphaned"
    assert first["resume_validation"]["status"] == "rerun"
    assert len(calls) == 5
    assert str(dag.tasks[0].task_id) in calls
    assert all(record["status"] == "completed" for record in resumed["tasks"].values())


def test_resume_reruns_log_tamper_and_transitive_dependents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    calls: list[str] = []

    def succeed(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv)
        calls.append(str(output))
        output.write_text(f"artifact:{len(calls)}", encoding="utf-8")

    monkeypatch.setattr(matrix, "_artifact_attestation", _fake_artifact_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)
    matrix.execute_dag(
        dag,
        subprocess_runner=succeed,
        verify_inputs=False,
        max_workers=3,
    )
    state = json.loads(matrix._journal_path(dag).read_text(encoding="utf-8"))
    first = state["tasks"][dag.tasks[0].task_id]
    stdout_record = first["attempts"][-1]["logs"]["stdout"]
    stdout_path = dag.plan.output_root / Path(stdout_record["path"].removeprefix("$OUTPUT_ROOT/"))
    stdout_path.write_text("tampered log", encoding="utf-8")
    calls.clear()

    matrix.execute_dag(
        dag,
        resume=True,
        subprocess_runner=succeed,
        verify_inputs=False,
        max_workers=3,
    )

    resumed = json.loads(matrix._journal_path(dag).read_text(encoding="utf-8"))
    assert len(calls) == 5
    assert resumed["tasks"][dag.tasks[0].task_id]["resume_validation"]["reason"] == (
        "stdout log is missing or changed size"
    )
    assert resumed["status"] == "completed"


def test_orchestrator_lock_is_exclusive_bound_and_never_stolen(
    tmp_path: Path,
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    dag.plan.output_root.mkdir(parents=True)
    resolved_inputs = {
        "verification": "disabled_for_orchestrator_test",
        "digest": matrix.semantic_sha256({"verification": "disabled_for_orchestrator_test"}),
    }
    owner = matrix._acquire_orchestrator_lock(
        dag,
        matrix._dag_digest(dag, resolved_inputs),
    )
    try:
        assert owner["schema"] == matrix.ORCHESTRATOR_LOCK_SCHEMA
        assert owner["pid"] == os.getpid()
        assert owner["host"]
        with pytest.raises(FileExistsError, match="never stolen automatically"):
            matrix.execute_dag(dag, verify_inputs=False, max_workers=2)
        assert matrix._orchestrator_lock_path(dag).is_file()
        assert not matrix._journal_path(dag).exists()
    finally:
        matrix._release_orchestrator_lock(dag, owner)


def test_keyboard_interrupt_fail_closed_clears_all_running_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    original_wait = matrix.wait
    wait_calls = 0

    def interrupt_once(*args, **kwargs):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            raise KeyboardInterrupt
        return original_wait(*args, **kwargs)

    def succeed(argv: list[str], **_: Any) -> None:
        time.sleep(0.02)
        _output_from_argv(argv).write_text("finished but unattested", encoding="utf-8")

    monkeypatch.setattr(matrix, "wait", interrupt_once)
    monkeypatch.setattr(matrix, "_artifact_attestation", _fake_artifact_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)

    with pytest.raises(KeyboardInterrupt):
        matrix.execute_dag(
            dag,
            subprocess_runner=succeed,
            verify_inputs=False,
            max_workers=3,
        )

    state = json.loads(matrix._journal_path(dag).read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["execution"]["keyboard_interrupt_policy"].startswith("stop_dispatch")
    assert "running" not in {record["status"] for record in state["tasks"].values()}
    assert any(
        attempt["status"] == "orphaned_after_interrupt"
        for record in state["tasks"].values()
        for attempt in record["attempts"]
    )
    assert not matrix._orchestrator_lock_path(dag).exists()


def test_keyboard_interrupt_retains_lock_when_worker_drain_cannot_be_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))

    class UndrainableExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, *_args, **_kwargs):
            return matrix.Future()

        def shutdown(self, **_kwargs):
            raise RuntimeError("drain unavailable")

    monkeypatch.setattr(matrix, "ThreadPoolExecutor", UndrainableExecutor)
    monkeypatch.setattr(
        matrix,
        "wait",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt) as exc_info:
        matrix.execute_dag(dag, verify_inputs=False, max_workers=2)

    assert any("lock was retained" in note for note in exc_info.value.__notes__)
    lock_path = matrix._orchestrator_lock_path(dag)
    assert lock_path.is_file()
    state = json.loads(matrix._journal_path(dag).read_text(encoding="utf-8"))
    assert state["status"] == "interrupted_draining"
    assert "running" not in {record["status"] for record in state["tasks"].values()}
    matrix._release_orchestrator_lock(
        dag,
        json.loads(lock_path.read_text(encoding="utf-8")),
    )


def test_result_phase_uses_declared_result_worker_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))

    def succeed(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"artifact:{output.name}", encoding="utf-8")

    monkeypatch.setattr(matrix, "_artifact_attestation", _fake_artifact_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)
    matrix.execute_dag(
        dag,
        subprocess_runner=succeed,
        verify_inputs=False,
        max_workers=1,
        result_max_workers=16,
    )
    journal = json.loads(matrix._journal_path(dag).read_text(encoding="utf-8"))
    assert journal["execution"]["max_workers"] == 1
    assert journal["execution"]["result_max_workers"] == 16
    assert journal["execution"]["phase_policy"] == "checkpoint_phase_then_result_phase"
    checkpoints = [task.task_id for task in dag.tasks if task.kind == "checkpoint_batch"]
    result_batches = [task.task_id for task in dag.tasks if task.kind == "result_batch"]
    sequence = journal["dispatch_sequence"]
    assert max(sequence.index(task_id) for task_id in checkpoints) < min(
        sequence.index(task_id) for task_id in result_batches
    )


def test_result_workers_are_clamped_to_available_cpu_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    monkeypatch.setattr(matrix, "available_cpu_threads", lambda: 25)

    def succeed(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"artifact:{output.name}", encoding="utf-8")

    monkeypatch.setattr(matrix, "_artifact_attestation", _fake_artifact_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)
    matrix.execute_dag(
        dag,
        subprocess_runner=succeed,
        verify_inputs=False,
        max_workers=1,
        result_max_workers=100,
    )
    journal = json.loads(matrix._journal_path(dag).read_text(encoding="utf-8"))
    assert journal["execution"]["requested_result_max_workers"] == 100
    assert journal["execution"]["result_max_workers"] == 25
    assert journal["execution"]["available_cpu_threads"] == 25


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_execute_rejects_invalid_max_workers(tmp_path: Path, workers: object) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        matrix.execute_dag(
            dag,
            verify_inputs=False,
            max_workers=workers,
        )


def test_training_forwards_alignment_cell_and_inactive_policy(tmp_path: Path) -> None:
    record = _plan_record()
    record["training"].update(
        {
            "feature_transfer_invariant_bottleneck": 2,
            "feature_alignment_weight": 0.1,
            "feature_alignment_mode": "conditional_rbf_mmd",
            "feature_alignment_min_cell_count": 6,
            "feature_alignment_inactive_policy": "fail_closed",
        }
    )
    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoint = next(task for task in dag.tasks if task.kind == "checkpoint_batch")
    assert plan.training.feature_alignment_min_cell_count == 6
    assert plan.training.feature_alignment_inactive_policy == "fail_closed"
    assert "--feature-alignment-min-cell-count" in checkpoint.argv
    assert checkpoint.argv[checkpoint.argv.index("--feature-alignment-inactive-policy") + 1] == (
        "fail_closed"
    )


def test_plan_supports_chunked_checkpoint_batches(tmp_path: Path) -> None:
    record = _plan_record()
    record["checkpoint_batch_size"] = 1
    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    checkpoints = [task for task in dag.tasks if task.kind == "checkpoint_batch"]
    assert len(checkpoints) == 8
    assert {
        "checkpoint_batch/source_a/rep01/chunk-00",
        "checkpoint_batch/source_a/rep01/chunk-01",
    } <= {task.task_id for task in checkpoints}
    result_batch = next(
        task
        for task in dag.tasks
        if task.task_id == "result_batch/source_a/rep01/p01"
    )
    assert result_batch.dependencies == ("checkpoint_batch/source_a/rep01/chunk-00",)
    zero_shot_argv = next(
        list(job["argv"])
        for job in dag.result_batch_job_records[result_batch.task_id]["jobs"]
        if job["arm_name"] == "source_a__zero_shot"
    )
    assert "--zero-shot-feature-cache" in zero_shot_argv


def test_plan_supports_result_worker_and_thread_tuning(tmp_path: Path) -> None:
    record = _plan_record()
    record["result_max_workers"] = 16
    record["result_threads"] = 1
    plan = _load(tmp_path, record)
    assert plan.result_max_workers == 16
    assert plan.result_threads == 1
    dag = matrix.build_dag(plan)
    batch_task = next(
        task
        for task in dag.tasks
        if task.task_id == "result_batch/source_a/rep01/p01"
    )
    argv = next(
        list(job["argv"])
        for job in dag.result_batch_job_records[batch_task.task_id]["jobs"]
        if job["arm_name"] == "source_a__zero_shot"
    )
    assert "--cpu-threads" in argv
    assert argv[argv.index("--cpu-threads") + 1] == "1"


    record = _plan_record()
    record["result_max_workers"] = 8
    record.pop("result_threads", None)
    auto_plan = _load(tmp_path, record)
    auto_dag = matrix.build_dag(auto_plan, result_max_workers=8)
    auto_task = next(
        task
        for task in auto_dag.tasks
        if task.task_id == "result_batch/source_a/rep01/p01"
    )
    auto_argv = next(
        list(job["argv"])
        for job in auto_dag.result_batch_job_records[auto_task.task_id]["jobs"]
        if job["arm_name"] == "source_a__zero_shot"
    )
    assert auto_argv[auto_argv.index("--cpu-threads") + 1] == str(
        matrix.resolve_cpu_threads(8)
    )


def test_plan_round_trips_euclidean_alignment_acw_state_and_sampler_kind(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    arm = record["source_arms"][0]
    arm["euclidean_alignment"] = {
        "mode": "source",
        "cache": "source-a.npz",
        "reference_mode": "native_preserve",
    }
    arm["acw_state"] = "offline/source-a.acw.json"
    arm["sampler_kind"] = "uniform"

    plan = _load(tmp_path, record)
    parsed = plan.source_arms[0]

    assert parsed.sampler_kind == "uniform"
    assert parsed.acw_state == (tmp_path / "offline" / "source-a.acw.json").resolve()
    assert parsed.euclidean_alignment is not None
    assert parsed.euclidean_alignment.mode == "source"
    assert parsed.euclidean_alignment.cache == (tmp_path / "source-a.npz").resolve()
    assert parsed.euclidean_alignment.reference_mode == "native_preserve"

    # An omitted alignment cache defaults to the source arm cache.
    arm["euclidean_alignment"] = {"mode": "source", "reference_mode": "common_average"}
    plan = _load(tmp_path, record)
    alignment = plan.source_arms[0].euclidean_alignment
    assert alignment is not None
    assert alignment.cache == plan.source_arms[0].cache
    assert alignment.reference_mode == "common_average"

    arm["euclidean_alignment"] = {"mode": "none"}
    plan = _load(tmp_path, record)
    alignment = plan.source_arms[0].euclidean_alignment
    assert alignment is not None
    assert alignment.mode == "none"
    assert alignment.cache is None
    assert alignment.reference_mode is None


@pytest.mark.parametrize(
    ("alignment", "match"),
    [
        ({"mode": "bogus"}, "euclidean_alignment.mode"),
        ({"mode": "source", "cache": "source-ab.npz"}, "must resolve to the source arm cache"),
        ({"mode": "source", "reference_mode": "identity"}, "reference_mode must be one of"),
        ({"mode": "none", "cache": "source-a.npz"}, "cache must be null/absent"),
        (
            {"mode": "none", "reference_mode": "native_preserve"},
            "reference_mode must be null/absent",
        ),
    ],
)
def test_plan_rejects_invalid_euclidean_alignment_fields(
    tmp_path: Path,
    alignment: dict[str, Any],
    match: str,
) -> None:
    record = _plan_record()
    record["source_arms"][0]["euclidean_alignment"] = alignment
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, record)


def test_sampler_kind_is_cross_checked_against_existing_selection_record(
    tmp_path: Path,
) -> None:
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    selection_path = selection_dir / "source-a.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "n2p3_source_selection/2",
                "sampler_kind": "target_similarity_diverse",
            }
        ),
        encoding="utf-8",
    )
    record = _plan_record()
    arm = record["source_arms"][0]
    arm["source_selection_record"] = "selection/source-a.json"
    arm["sampler_kind"] = "target_similarity_diverse"

    assert _load(tmp_path, record).source_arms[0].sampler_kind == (
        "target_similarity_diverse"
    )

    selection_path.write_text(
        json.dumps(
            {"schema": "n2p3_source_selection/2", "sampler_kind": "dense_pds"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disagrees with declared"):
        _load(tmp_path, record)

    selection_path.write_text(
        json.dumps(
            {"schema": "n2p3_source_selection/1", "sampler_kind": "target_similarity_diverse"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported schema"):
        _load(tmp_path, record)


def test_sampler_kind_record_combinations_and_missing_files(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["source_arms"][0]
    arm["sampler_kind"] = "uniform"
    arm["source_selection_record"] = "selection/source-a.json"
    with pytest.raises(ValueError, match="cannot be combined"):
        _load(tmp_path, record)

    arm.pop("source_selection_record")
    arm["sampler_kind"] = "dense_pds"
    with pytest.raises(ValueError, match="requires source_selection_record"):
        _load(tmp_path, record)

    arm["source_selection_record"] = "selection/missing.json"
    plan = _load(tmp_path, record)
    assert plan.source_arms[0].sampler_kind == "dense_pds"
    assert plan.source_arms[0].source_selection_record == (
        tmp_path / "selection" / "missing.json"
    ).resolve()


def test_plan_parses_as_mmd_source_stream_full_fine(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["evaluation_arms"][1]
    arm["head"] = "full_fine"
    arm["as_mmd"] = {
        "lambda0": 0.25,
        "warmup_epochs": 3,
        "target_weight_min": 2.0,
        "target_weight_max": 8.0,
        "input": "source_stream",
    }

    plan = _load(tmp_path, record)
    parsed = plan.evaluation_arms[1]

    assert parsed.as_mmd_lambda0 == 0.25
    assert parsed.as_mmd_warmup_epochs == 3
    assert parsed.as_mmd_target_weight_min == 2.0
    assert parsed.as_mmd_target_weight_max == 8.0
    assert parsed.as_mmd_input == "source_stream"
    assert parsed.as_mmd_source_logits is None

    arm["as_mmd"] = {"lambda0": 0.1}
    parsed = _load(tmp_path, record).evaluation_arms[1]
    assert parsed.as_mmd_warmup_epochs == 1
    assert parsed.as_mmd_target_weight_min == 1.0
    assert parsed.as_mmd_target_weight_max == 6.0
    assert parsed.as_mmd_input == "source_stream"


def test_plan_parses_frozen_logits_as_mmd_and_requires_source_logits(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    arm = record["evaluation_arms"][1]
    arm["as_mmd"] = {
        "lambda0": 0.5,
        "input": "frozen_logits",
        "source_logits": "offline/logits.npz",
    }

    plan = _load(tmp_path, record)
    parsed = plan.evaluation_arms[1]
    assert parsed.head == "linear"
    assert parsed.as_mmd_input == "frozen_logits"
    assert parsed.as_mmd_source_logits == (tmp_path / "offline" / "logits.npz").resolve()

    arm["as_mmd"].pop("source_logits")
    with pytest.raises(ValueError, match="source_logits is required"):
        _load(tmp_path, record)

    arm["as_mmd"] = {"lambda0": 0.5, "source_logits": "offline/logits.npz"}
    with pytest.raises(ValueError, match="source_logits must be null/absent"):
        _load(tmp_path, record)


def test_as_mmd_zero_lambda_forbids_input_and_source_logits(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["evaluation_arms"][1]
    arm["head"] = "full_fine"
    arm["as_mmd"] = {"lambda0": 0.0, "input": "source_stream"}
    with pytest.raises(ValueError, match="input must be null/absent when lambda0 is 0"):
        _load(tmp_path, record)

    arm["as_mmd"] = {"lambda0": 0.0, "source_logits": "offline/logits.npz"}
    with pytest.raises(ValueError, match="source_logits must be null/absent when lambda0 is 0"):
        _load(tmp_path, record)

    arm["as_mmd"] = {"lambda0": 0.0}
    parsed = _load(tmp_path, record).evaluation_arms[1]
    assert parsed.as_mmd_lambda0 == 0.0
    assert parsed.as_mmd_input is None
    assert parsed.as_mmd_source_logits is None


def test_as_mmd_source_stream_requires_full_fine(tmp_path: Path) -> None:
    record = _plan_record()
    record["evaluation_arms"][1]["as_mmd"] = {"lambda0": 0.1}
    with pytest.raises(ValueError, match="requires head='full_fine'"):
        _load(tmp_path, record)


def test_as_mmd_splitbn_target_buffer_source_must_match_training(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    record["training"]["splitbn_target_buffer_source"] = "pooled_source"
    arm = record["evaluation_arms"][1]
    arm["head"] = "full_fine"
    arm["as_mmd"] = {
        "lambda0": 0.1,
        "splitbn_target_buffer_source": "target_calibration",
    }
    with pytest.raises(ValueError, match="must equal training.splitbn_target_buffer_source"):
        _load(tmp_path, record)

    arm["as_mmd"]["splitbn_target_buffer_source"] = "pooled_source"
    plan = _load(tmp_path, record)
    assert plan.training.splitbn_target_buffer_source == "pooled_source"


def test_build_dag_forwards_as_mmd_and_source_selection_to_result_argv(
    tmp_path: Path,
) -> None:
    record = _plan_record()
    source_arm = record["source_arms"][0]
    source_arm["source_selection_record"] = "selection/source-a.json"
    evaluation_arm = record["evaluation_arms"][1]
    evaluation_arm["head"] = "full_fine"
    evaluation_arm["as_mmd"] = {
        "lambda0": 0.25,
        "warmup_epochs": 5,
        "target_weight_min": 2.0,
        "target_weight_max": 8.0,
    }

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    batch_task = next(
        task for task in dag.tasks if task.task_id == "result_batch/source_a/rep01/p01"
    )
    jobs = dag.result_batch_job_records[batch_task.task_id]["jobs"]
    argv = next(
        list(job["argv"])
        for job in jobs
        if job["arm_name"] == "source_a__linear_time_split"
    )

    assert argv[argv.index("--as-mmd-lambda0") + 1] == "0.25"
    assert argv[argv.index("--as-mmd-warmup-epochs") + 1] == "5"
    assert argv[argv.index("--as-mmd-target-weight-min") + 1] == "2.0"
    assert argv[argv.index("--as-mmd-target-weight-max") + 1] == "8.0"
    assert argv[argv.index("--source-stream-cache") + 1] == str(plan.source_arms[0].cache)
    assert argv[argv.index("--source-selection-record") + 1] == str(
        plan.source_arms[0].source_selection_record
    )
    assert argv.index("--as-mmd-lambda0") < argv.index("--source-stream-cache")
    assert argv.index("--source-stream-cache") < argv.index("--source-selection-record")
    assert argv.index("--source-selection-record") < argv.index("--seed")

    zero_shot_argv = next(
        list(job["argv"]) for job in jobs if job["arm_name"] == "source_a__zero_shot"
    )
    assert not {
        "--as-mmd-lambda0",
        "--as-mmd-warmup-epochs",
        "--as-mmd-target-weight-min",
        "--as-mmd-target-weight-max",
        "--source-stream-cache",
        "--as-mmd-source-logits",
    } & set(zero_shot_argv)
    assert "--source-selection-record" in zero_shot_argv


def test_build_dag_forwards_frozen_logits_as_mmd_source_logits(tmp_path: Path) -> None:
    record = _plan_record()
    arm = record["evaluation_arms"][1]
    arm["as_mmd"] = {
        "lambda0": 0.5,
        "input": "frozen_logits",
        "source_logits": "offline/logits.npz",
    }

    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    batch_task = next(
        task for task in dag.tasks if task.task_id == "result_batch/source_a/rep01/p01"
    )
    argv = next(
        list(job["argv"])
        for job in dag.result_batch_job_records[batch_task.task_id]["jobs"]
        if job["arm_name"] == "source_a__linear_time_split"
    )
    assert "--source-stream-cache" not in argv
    assert argv[argv.index("--as-mmd-source-logits") + 1] == str(
        plan.evaluation_arms[1].as_mmd_source_logits
    )


def test_euclidean_alignment_provenance_gate(tmp_path: Path) -> None:
    _write_preflight_inputs(tmp_path)
    record = _plan_record()
    record["source_arms"][0]["euclidean_alignment"] = {
        "mode": "source",
        "reference_mode": "native_preserve",
    }
    dag = matrix.build_dag(_load(tmp_path, record))

    with pytest.raises(ValueError, match="mode='source'.*no data_alignment"):
        matrix._verify_inputs(dag)

    save_epoch_dataset(
        tmp_path / "source-a.npz",
        _cache_dataset(
            name="source-a",
            local_subjects=("SOURCE::01", "SOURCE::02"),
            authority_subjects=("01", "02"),
            data_alignment={
                "schema": "n2p3_euclidean_alignment_fit/1",
                "scope": "source_rows",
                "reference_mode": "native_preserve",
            },
        ),
    )
    resolved = matrix._verify_inputs(dag)
    assert resolved["source_caches"]["source_a"]["cache_sha256"]

    record["source_arms"][0]["euclidean_alignment"]["mode"] = "target_calibration"
    mismatched = matrix.build_dag(_load(tmp_path, record))
    with pytest.raises(ValueError, match="scope='target_calibration'"):
        matrix._verify_inputs(mismatched)

    record["source_arms"][0]["euclidean_alignment"] = {
        "mode": "source",
        "reference_mode": "common_average",
    }
    reference_mismatch = matrix.build_dag(_load(tmp_path, record))
    with pytest.raises(ValueError, match="reference_mode=.*disagrees"):
        matrix._verify_inputs(reference_mismatch)


def test_euclidean_alignment_mode_none_accepts_raw_cache(tmp_path: Path) -> None:
    _write_preflight_inputs(tmp_path)
    record = _plan_record()
    record["source_arms"][0]["euclidean_alignment"] = {"mode": "none"}
    matrix._verify_inputs(matrix.build_dag(_load(tmp_path, record)))



def _selected_arm_record(initial_checkpoint_arm: str | None = "source_a") -> dict[str, Any]:
    record = _plan_record()
    record["source_arms"][1] = {
        "name": "source_sel",
        "cache": "source-a.npz",
        "source_domain_mass": None,
        "risk_within_domain_unit": None,
        "selection_domain": None,
        "splitbn_source_weights": None,
        "euclidean_alignment": {"mode": "none"},
        "sampler_kind": "target_similarity_diverse",
        "source_selection_record": "selection-does-not-need-to-exist.json",
        "initial_checkpoint_arm": initial_checkpoint_arm,
    }
    record["statistical_design"]["planned_contrasts"] = [
        ["source_a__zero_shot", "source_sel__zero_shot"],
        ["source_a__linear_time_split", "source_sel__linear_time_split"],
    ]
    return record


def test_selection_arm_declares_uniform_pretrain_checkpoint_dependency(tmp_path: Path) -> None:
    plan = _load(tmp_path, _selected_arm_record())
    assert plan.source_arms[1].initial_checkpoint_arm == "source_a"
    dag = matrix.build_dag(plan)
    selected = next(
        task for task in dag.tasks if task.task_id.startswith("checkpoint_batch/source_sel/")
    )
    initial = next(
        task for task in dag.tasks if task.task_id.startswith("checkpoint_batch/source_a/")
    )
    assert selected.dependencies == (initial.task_id,)
    assert "--initial-checkpoint" in selected.argv
    assert "--source-selection-record" in selected.argv


def test_selection_initial_checkpoint_requires_earlier_uniform_same_cache_arm(
    tmp_path: Path,
) -> None:
    record = _selected_arm_record()
    record["source_arms"][1]["initial_checkpoint_arm"] = "source_sel"
    with pytest.raises(ValueError, match="earlier source arm"):
        _load(tmp_path, record)

    record = _selected_arm_record()
    record["source_arms"][1]["initial_checkpoint_arm"] = "missing"
    with pytest.raises(ValueError, match="unknown arm"):
        _load(tmp_path, record)

    record = _selected_arm_record()
    record["source_arms"][1]["cache"] = "source-ab.npz"
    with pytest.raises(ValueError, match="share its source cache"):
        _load(tmp_path, record)
