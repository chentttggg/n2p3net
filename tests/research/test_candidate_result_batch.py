from __future__ import annotations

import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from experiments import run_candidate_cross_decision as result_runner
from experiments import run_candidate_result_batch as batch
from experiments.result_batch_contract import (
    RESULT_BATCH_JOBS_SCHEMA,
    RESULT_BATCH_LEDGER_SCHEMA,
)
from tests.support.io import read_json, write_json


def test_candidate_runner_starts_directly_outside_repository(tmp_path: Path) -> None:
    runner = Path(result_runner.__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-I", str(runner), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--calibration-selections" in result.stdout


@pytest.mark.parametrize(
    "runner_name,extra_arguments",
    [
        ("run_candidate_cross_decision", ["--training-replicate-key", "r1", "--partition-key", "p1"]),
        ("run_within_subject_transfer", []),
    ],
)
def test_inner_selection_rejects_outer_qc_before_input_access(tmp_path, runner_name, extra_arguments):
    runner = Path(result_runner.__file__).resolve().with_name(f"{runner_name}.py")
    arguments = [
        "--dataset-cache", str(tmp_path / "absent.npz"),
        "--source-snapshot-manifest", str(tmp_path / "absent.manifest.json"),
        "--identity-exclusion-policy", "source",
        "--arm-name", "qc-scope",
        "--head", "linear",
        "--fold-local-qc",
        "--epoch-selection", "target_time_split",
        "--output", str(tmp_path / "result.json"),
    ]
    arguments.extend(extra_arguments)
    result = subprocess.run(
        [sys.executable, "-I", str(runner), *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 2
    assert "separate inner-training scopes" in result.stderr
    assert "FileNotFoundError" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def _argv(*, arm_name, output, dataset_cache="target.npz", head="zero_shot"):
    options = {
        "dataset-cache": dataset_cache,
        "checkpoint": "checkpoint.pt",
        "arm-name": arm_name,
        "training-replicate-key": "rep01",
        "partition-key": "p01",
        "source-snapshot-manifest": "source.manifest.json",
        "identity-exclusion-policy": "source_or_global",
        "calibration-selections": 5,
        "test-reps": 8,
        "min-evidence-level": 8,
        "head": head,
        "normalization": "source",
        "seed": 101,
        "target-subjects-file": "p01.subjects.json",
        "device": "cpu",
        "cpu-threads": 1,
        "zero-shot-feature-cache": output.parent / "zero_shot.npz",
        "output": output,
    }
    return [part for name, value in options.items() for part in ("--" + name, str(value))]


def test_batch_jobs_record_requires_shared_flags_and_unique_outputs(tmp_path: Path) -> None:
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    jobs = {
        "schema": RESULT_BATCH_JOBS_SCHEMA,
        "job_count": 2,
        "partition_key": "p01",
        "jobs": [
            {"arm_name": "a", "argv": _argv(arm_name="a", output=first_output)},
            {"arm_name": "b", "argv": _argv(arm_name="b", output=first_output)},
        ],
    }
    jobs_path = tmp_path / "jobs.json"
    write_json(jobs_path, jobs)
    with pytest.raises(ValueError, match="repeats an output path"):
        batch._validate_jobs_record(batch._read_json_mapping(jobs_path, "jobs"), jobs_path)

    jobs["jobs"][1] = {"arm_name": "b", "argv": _argv(arm_name="b", output=second_output, head="adapter")}
    write_json(jobs_path, jobs)
    validated = batch._validate_jobs_record(batch._read_json_mapping(jobs_path, "jobs"), jobs_path)
    assert [job["arm_name"] for job in validated] == ["a", "b"]

    jobs["jobs"][1]["argv"].extend(["--evaluation-snapshot-manifest", "different.manifest.json"])
    with pytest.raises(ValueError, match="shared evaluation snapshot"):
        batch._validate_jobs_record(jobs, jobs_path)

    jobs["jobs"][0]["argv"].extend(["--evaluation-snapshot-manifest", "different.manifest.json"])
    assert len(batch._validate_jobs_record(jobs, jobs_path)) == 2


def test_batch_main_writes_ledger_and_delegates_each_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    jobs_path = tmp_path / "jobs.json"
    write_json(jobs_path, {
                "schema": RESULT_BATCH_JOBS_SCHEMA,
                "job_count": 2,
                "partition_key": "p01",
                "jobs": [
                    {"arm_name": "a", "argv": _argv(arm_name="a", output=first_output)},
                    {"arm_name": "b", "argv": _argv(arm_name="b", output=second_output)},
                ],
            })
    calls: list[list[str]] = []

    def fake_main(argv: list[str]) -> None:
        calls.append(list(argv))
        output = Path(next(value for flag, value in pairwise(argv) if flag == "--output"))
        write_json(output, {
            "schema": "n2p3_candidate_cross_decision_result/1",
            "run_status": "completed",
            "evaluation_contract": {}, "evaluation_contract_digest": "a" * 64,
            "decision_plan": {}, "decision_plan_digest": "b" * 64,
        })

    monkeypatch.setattr(result_runner, "main", fake_main)
    ledger_path = tmp_path / "ledger.json"
    batch.main(["--jobs-file", str(jobs_path), "--output", str(ledger_path)])
    ledger = read_json(ledger_path)
    assert ledger["schema"] == RESULT_BATCH_LEDGER_SCHEMA
    assert ledger["completed"] is True
    assert ledger["job_count"] == 2
    assert [job["arm_name"] for job in ledger["jobs"]] == ["a", "b"]
    assert all(Path(job["output"]).is_file() for job in ledger["jobs"])
    assert len(calls) == 2


@pytest.mark.parametrize(
    "mismatch_field,wrong_value", [("partition_key", "p02"), ("evaluation_snapshot_sha256", "f" * 64)]
)
def test_zero_shot_feature_cache_round_trip_and_header_mismatch(
    tmp_path: Path, mismatch_field: str, wrong_value
) -> None:
    header = result_runner._zero_shot_cache_header(
        checkpoint_sha256="c" * 64,
        target_cache_sha256="d" * 64,
        evaluation_snapshot_sha256="e" * 64,
        partition_key="p01",
        epoch_rows=np.asarray([4, 9]),
        test_reps=8,
    )
    path = tmp_path / "features.npz"
    features = np.arange(20, dtype=np.float32).reshape(2, 10)
    scores = np.asarray([0.25, -0.5], dtype=np.float64)
    result_runner._write_zero_shot_feature_cache(
        path, header=header, features=features, scores=scores
    )
    loaded = result_runner._read_zero_shot_feature_cache(path, header)
    assert loaded is not None
    assert np.array_equal(loaded["features"], features)
    assert np.array_equal(loaded["scores"], scores)
    header[mismatch_field] = wrong_value
    assert result_runner._read_zero_shot_feature_cache(path, header) is None
