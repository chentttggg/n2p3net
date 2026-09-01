"""Typed subject-level failures and atomic partial-run records."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4


class SubjectFailureCode(StrEnum):
    INSUFFICIENT_CLASSES = "insufficient_classes"
    NONFINITE_OPTIMIZATION = "nonfinite_optimization"
    ACCELERATOR_OOM = "accelerator_oom"
    NUMERICAL_FAILURE = "numerical_failure"


class ExpectedSubjectError(RuntimeError):
    """A predeclared subject-local failure that counts in operational results."""

    def __init__(self, code: SubjectFailureCode | str, *, stage: str, detail: str) -> None:
        self.code = SubjectFailureCode(code)
        self.stage = str(stage).strip()
        self.detail = str(detail).strip()
        if not self.stage or not self.detail:
            raise ValueError("ExpectedSubjectError stage and detail must be non-empty.")
        super().__init__(f"{self.stage}:{self.code.value}: {self.detail}")

    def record(self, *, subject: str) -> dict[str, str]:
        return {
            "subject": str(subject),
            "stage": self.stage,
            "error_code": self.code.value,
            "detail": self.detail,
        }


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace one JSON record without exposing a partial document."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target


def partial_run_record(
    *,
    schema: str,
    run_status: str,
    requested_subjects: Sequence[str],
    completed_subjects: Sequence[str],
    subject_failures: Sequence[Mapping[str, str]],
    fatal_error: Mapping[str, str] | None = None,
) -> dict[str, object]:
    status = str(run_status)
    if status not in {"running", "completed", "aborted"}:
        raise ValueError("run_status must be running, completed, or aborted.")
    requested = tuple(str(value) for value in requested_subjects)
    completed = tuple(str(value) for value in completed_subjects)
    if not schema or not requested or len(set(requested)) != len(requested):
        raise ValueError("partial run requires a schema and unique requested subjects.")
    if not set(completed) <= set(requested) or len(set(completed)) != len(completed):
        raise ValueError("completed_subjects must be a unique subset of requested_subjects.")
    failure_subjects = [str(record.get("subject", "")) for record in subject_failures]
    if any(not value for value in failure_subjects) or len(set(failure_subjects)) != len(
        failure_subjects
    ):
        raise ValueError("subject_failures must identify unique non-empty subjects.")
    if not set(failure_subjects) <= set(requested) or set(failure_subjects) & set(completed):
        raise ValueError("failed subjects must be requested and cannot also be completed.")
    if status == "completed" and set(completed) | set(failure_subjects) != set(requested):
        raise ValueError("completed run must account for every requested subject.")
    if status == "aborted" and fatal_error is None:
        raise ValueError("aborted run requires fatal_error.")
    if status != "aborted" and fatal_error is not None:
        raise ValueError("fatal_error is only valid for an aborted run.")
    return {
        "schema": schema,
        "run_status": status,
        "requested_subjects": list(requested),
        "completed_subjects": list(completed),
        "subject_failures": [dict(record) for record in subject_failures],
        "fatal_error": None if fatal_error is None else dict(fatal_error),
    }


__all__ = [
    "ExpectedSubjectError",
    "SubjectFailureCode",
    "atomic_write_json",
    "partial_run_record",
]
