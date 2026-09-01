from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.channel import build_channel_identity
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import EpochDataset, preprocessing_spec_from_contract, save_epoch_dataset
from data.events import observed_only_timeline
from data.identity import DatasetIdentityTable, ParticipantIdentityRecord, origin_subject_key
from experiments import run_candidate_promotion_matrix as matrix


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
                "subject_prefix_repeat": {"A::": 1},
                "input_stats_prefix": "A::",
            },
            {
                "name": "source_ab",
                "cache": "source-ab.npz",
                "subject_prefix_repeat": {"A::": 1, "B::": 1},
                "input_stats_prefix": None,
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
) -> EpochDataset:
    identity = build_channel_identity(("Fz", "Cz", "Pz"), allow_missing_positions=False)
    positions = np.asarray(identity.coords, dtype=float).copy()
    positions[0, 0] += position_offset
    subjects = np.asarray(local_subjects)
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
        provenance={"source": "unit_test", "source_reference": "average"},
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
            ),
        )


def test_plan_rejects_hit_at_r_below_promotion_floor(tmp_path: Path) -> None:
    record = _plan_record()
    record["test_reps"] = 7
    record["statistical_design"]["evidence_level"] = 7
    with pytest.raises(ValueError, match="at least 8"):
        _load(tmp_path, record)


def test_training_rejects_even_kernel_but_accepts_explicit_k65(tmp_path: Path) -> None:
    even = _plan_record()
    even["training"]["temporal_kernel_size"] = 34
    with pytest.raises(ValueError, match="must be odd"):
        _load(tmp_path, even)

    broad = _plan_record()
    broad["training"]["temporal_kernel_size"] = 65
    assert _load(tmp_path, broad).training.temporal_kernel_size == 65


@pytest.mark.parametrize("head", ["linear", "mlp16", "classifier_fine", "full_fine"])
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
        "checkpoint": 8,
        "result": 16,
        "manifest": 1,
        "analysis": 1,
    }
    assert output["task_count"] == 26
    assert output["resolved_inputs"] == resolved_inputs
    assert all(task["argv"][0] == "$PYTHON" for task in output["tasks"])
    assert not (tmp_path / "outputs").exists()


def test_dag_passes_explicit_time_split_linear_configuration(tmp_path: Path) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    linear = next(
        task for task in dag.tasks if task.task_id == "result/source_a__linear_time_split/rep01/p01"
    )
    argv = list(linear.argv)
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
        assert argv[argv.index(flag) + 1] == value
    assert "--no-fold-local-qc" in argv

    zero_shot = next(
        task for task in dag.tasks if task.task_id == "result/source_a__zero_shot/rep01/p01"
    )
    assert not {
        "--epochs",
        "--batch-size",
        "--lr",
        "--fold-local-qc",
        "--no-fold-local-qc",
        "--adapt-batchnorm",
        "--epoch-selection",
        "--target-stat-weight",
    } & set(zero_shot.argv)


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
    flag = "--checkpoint" if "--checkpoint" in argv and "--output" not in argv else "--output"
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
    assert len(calls) == 26
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
    assert resolved["target_cache"]["cache_attestation_sha256"]
    assert set(resolved["source_caches"]) == {"source_a", "source_ab"}
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
