"""Promotion input attestation, participant identity and geometry admission."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from data.epochs import save_epoch_dataset
from experiments import run_candidate_promotion_matrix as matrix
from tests.research.promotion.support import (
    _argument,
    _cache_dataset,
    _load,
    _plan_record,
    _single_multidomain_arm_record,
    _write_preflight_inputs,
)
from tests.support.snapshots import write_source_snapshot


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
    assert _argument(checkpoint.argv, "--source-residual-drop-p") == "0.5"


@pytest.mark.parametrize(
    "training",
    [
        pytest.param({"splitbn_target_buffer_source": "pooled_source"}, id="splitbn"),
        pytest.param(
            {
                "feature_transfer_invariant_bottleneck": 2,
                "source_residual_drop_p": 0.2,
                "source_residual_drop_mode": "sample",
            },
            id="source-residual",
        ),
    ],
)
def test_p0_p1_preflight_rejects_cache_without_multidomain_axis(tmp_path, training):
    _write_preflight_inputs(tmp_path)
    record = _plan_record()
    record["source_arms"] = [record["source_arms"][0]]
    record["statistical_design"]["planned_contrasts"] = [
        ["source_a__linear_time_split", "source_a__zero_shot"]
    ]
    record["training"].update(training)
    with pytest.raises(ValueError, match="does not declare at least two source domains"):
        matrix._verify_inputs(matrix.build_dag(_load(tmp_path, record)))


def test_identity_mismatch_is_rejected_before_any_subprocess(tmp_path: Path) -> None:
    _write_preflight_inputs(tmp_path)
    save_epoch_dataset(
        tmp_path / "source-a.npz",
        _cache_dataset(
            name="source-a.npz",
            local_subjects=("SOURCE::01", "SOURCE::02"),
            authority_subjects=("different-person", "02"),
        ),
    )
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
    _write_preflight_inputs(tmp_path)
    save_epoch_dataset(
        tmp_path / "source-a.npz",
        _cache_dataset(
            name="source-a.npz",
            local_subjects=("SOURCE::01", "SOURCE::02"),
            authority_subjects=("01", "02"),
            position_offset=0.001,
        ),
    )
    dag = matrix.build_dag(_load(tmp_path))
    calls: list[list[str]] = []

    def must_not_run(argv: list[str], **_: Any) -> None:
        calls.append(argv)

    with pytest.raises(ValueError, match="channel_positions_m"):
        matrix.execute_dag(dag, subprocess_runner=must_not_run)
    assert calls == []


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


@pytest.mark.parametrize("separate_evaluation_snapshot", [False, True])
def test_dag_binds_training_and_evaluation_snapshots_separately(
    tmp_path: Path, separate_evaluation_snapshot: bool
) -> None:
    _write_preflight_inputs(tmp_path)
    record = _plan_record()
    if separate_evaluation_snapshot:
        directory = tmp_path / "evaluation"
        directory.mkdir()
        manifest, _, _ = write_source_snapshot(directory, source_value=2, source_commit="b" * 40)
        record["evaluation_snapshot_manifest"] = str(manifest)
    plan = _load(tmp_path, record)
    dag = matrix.build_dag(plan)
    for task in dag.tasks:
        if task.kind == "checkpoint_batch":
            assert _argument(task.argv, "--source-snapshot-manifest") == str(
                plan.source_snapshot_manifest
            )
            assert "--evaluation-snapshot-manifest" not in task.argv
        elif task.kind == "manifest":
            assert _argument(task.argv, "--evaluation-snapshot-manifest") == str(
                plan.evaluation_snapshot_manifest
            )
    for batch_record in dag.result_batch_job_records.values():
        for job in batch_record["jobs"]:
            argv = job["argv"]
            assert _argument(argv, "--evaluation-snapshot-manifest") == str(
                plan.evaluation_snapshot_manifest
            )
    resolved = matrix._verify_inputs(dag)
    assert (
        resolved["source_snapshot"]["archive_sha256"]
        != resolved["evaluation_snapshot"]["archive_sha256"]
    ) is separate_evaluation_snapshot
