from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments import run_candidate_result_batch as batch
from experiments import run_candidate_cross_decision as result_runner
from experiments.result_batch_contract import (
    RESULT_BATCH_JOBS_SCHEMA,
    RESULT_BATCH_LEDGER_SCHEMA,
    ZERO_SHOT_FEATURE_CACHE_SCHEMA,
)


def _argv(
    *,
    arm_name: str,
    output: Path,
    dataset_cache: str = "target.npz",
    head: str = "zero_shot",
) -> list[str]:
    return [
        "--dataset-cache",
        dataset_cache,
        "--checkpoint",
        "checkpoint.pt",
        "--arm-name",
        arm_name,
        "--training-replicate-key",
        "rep01",
        "--partition-key",
        "p01",
        "--source-snapshot-manifest",
        "source.manifest.json",
        "--identity-exclusion-policy",
        "source_or_global",
        "--calibration-selections",
        "5",
        "--test-reps",
        "8",
        "--min-evidence-level",
        "8",
        "--head",
        head,
        "--normalization",
        "source",
        "--seed",
        "101",
        "--target-subjects-file",
        "p01.subjects.json",
        "--device",
        "cpu",
        "--cpu-threads",
        "1",
        "--zero-shot-feature-cache",
        str(output.parent / "zero_shot.npz"),
        "--output",
        str(output),
    ]


def _write_result(path: Path, *, completed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "n2p3_candidate_cross_decision_result/1",
                "run_status": "completed" if completed else "running",
                "evaluation_contract": {},
                "evaluation_contract_digest": "a" * 64,
                "decision_plan": {},
                "decision_plan_digest": "b" * 64,
            }
        ),
        encoding="utf-8",
    )


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
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    with pytest.raises(ValueError, match="repeats an output path"):
        batch._validate_jobs_record(batch._read_json_mapping(jobs_path, "jobs"), jobs_path)

    jobs["jobs"][1] = {"arm_name": "b", "argv": _argv(arm_name="b", output=second_output, head="adapter")}
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    validated = batch._validate_jobs_record(batch._read_json_mapping(jobs_path, "jobs"), jobs_path)
    assert [job["arm_name"] for job in validated] == ["a", "b"]


def test_batch_main_writes_ledger_and_delegates_each_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "schema": RESULT_BATCH_JOBS_SCHEMA,
                "job_count": 2,
                "partition_key": "p01",
                "jobs": [
                    {"arm_name": "a", "argv": _argv(arm_name="a", output=first_output)},
                    {"arm_name": "b", "argv": _argv(arm_name="b", output=second_output)},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_main(argv: list[str]) -> None:
        calls.append(list(argv))
        output = Path(argv[argv.index("--output") + 1])
        _write_result(output)

    monkeypatch.setattr(result_runner, "main", fake_main)
    ledger_path = tmp_path / "ledger.json"
    batch.main(["--jobs-file", str(jobs_path), "--output", str(ledger_path)])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["schema"] == RESULT_BATCH_LEDGER_SCHEMA
    assert ledger["completed"] is True
    assert ledger["job_count"] == 2
    assert [job["arm_name"] for job in ledger["jobs"]] == ["a", "b"]
    assert all(Path(job["output"]).is_file() for job in ledger["jobs"])
    assert len(calls) == 2


def test_zero_shot_feature_cache_round_trip_and_header_mismatch(tmp_path: Path) -> None:
    header = {
        "schema": ZERO_SHOT_FEATURE_CACHE_SCHEMA,
        "checkpoint_sha256": "c" * 64,
        "target_cache_sha256": "d" * 64,
        "partition_key": "p01",
        "epoch_rows": [4, 9],
        "test_reps": 8,
    }
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
    header["partition_key"] = "p02"
    assert result_runner._read_zero_shot_feature_cache(path, header) is None
