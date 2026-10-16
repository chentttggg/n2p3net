from __future__ import annotations

import pytest

from research.execution import (
    ExpectedSubjectError,
    SubjectFailureCode,
    atomic_write_json,
    partial_run_record,
)
from tests.support.io import read_json


def test_expected_subject_failure_has_a_structured_code() -> None:
    failure = ExpectedSubjectError(
        SubjectFailureCode.NONFINITE_OPTIMIZATION,
        stage="adapter_fit",
        detail="loss became non-finite",
    )
    assert failure.record(subject="P01")["error_code"] == "nonfinite_optimization"


def test_completed_partial_record_accounts_for_every_requested_subject() -> None:
    record = partial_run_record(
        schema="example/1",
        run_status="completed",
        requested_subjects=("P01", "P02"),
        completed_subjects=("P01",),
        subject_failures=(
            {"subject": "P02", "stage": "fit", "error_code": "numerical_failure"},
        ),
    )
    assert record["run_status"] == "completed"
    with pytest.raises(ValueError, match="account"):
        partial_run_record(
            schema="example/1",
            run_status="completed",
            requested_subjects=("P01", "P02"),
            completed_subjects=("P01",),
            subject_failures=(),
        )


def test_unknown_failure_aborts_instead_of_becoming_a_subject_miss(tmp_path) -> None:
    record = partial_run_record(
        schema="example/1",
        run_status="aborted",
        requested_subjects=("P01", "P02"),
        completed_subjects=("P01",),
        subject_failures=(),
        fatal_error={"type": "RuntimeError", "detail": "programming defect"},
    )
    output = atomic_write_json(tmp_path / "partial.json", record)
    loaded = read_json(output)
    assert loaded["run_status"] == "aborted"
    assert loaded["fatal_error"]["type"] == "RuntimeError"


def test_atomic_result_serialization_failure_preserves_previous_record(tmp_path) -> None:
    output = atomic_write_json(tmp_path / "result.json", {"status": "completed", "auc": 0.8})
    previous = output.read_bytes()

    with pytest.raises(ValueError, match="Out of range float"):
        atomic_write_json(output, {"status": "completed", "auc": float("nan")})

    assert output.read_bytes() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]


@pytest.mark.parametrize("failures", [2, 7])
def test_atomic_replace_read_lock_is_bounded_and_preserves_old_bytes(tmp_path, monkeypatch, failures):
    from pathlib import Path

    output = atomic_write_json(tmp_path / "progress.json", {"completed": 1})
    previous = output.read_bytes()
    original = Path.replace
    attempts, delays = [], []

    def locked(path, target):
        attempts.append(1)
        assert output.read_bytes() == previous
        if len(attempts) <= failures:
            raise PermissionError("temporary reader denies delete sharing")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("research.execution.time.sleep", delays.append)
    if failures == 7:
        with pytest.raises(PermissionError):
            atomic_write_json(output, {"completed": 2})
        assert output.read_bytes() == previous
        assert len(attempts) == 7 and sum(delays) == pytest.approx(.63)
    else:
        atomic_write_json(output, {"completed": 2})
        assert read_json(output) == {"completed": 2}
        assert len(attempts) == 3
    assert list(tmp_path.iterdir()) == [output]
