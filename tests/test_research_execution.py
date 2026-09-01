from __future__ import annotations

import json

import pytest

from research.execution import (
    ExpectedSubjectError,
    SubjectFailureCode,
    atomic_write_json,
    partial_run_record,
)


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
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["run_status"] == "aborted"
    assert loaded["fatal_error"]["type"] == "RuntimeError"
