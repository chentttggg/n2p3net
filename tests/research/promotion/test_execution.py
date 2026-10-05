"""Promotion task execution, artifact integrity, failure and resume behavior."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
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
    _write_preflight_inputs,
)
from tests.support.io import read_json, write_json


def test_result_batch_embedded_record_validates_ledger_and_member_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    batch_task = next(task for task in dag.tasks if task.kind == "result_batch")
    jobs_path = dag.result_batch_job_paths[batch_task.task_id]
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_record = json.loads(json.dumps(dag.result_batch_job_records[batch_task.task_id]))
    write_json(jobs_path, jobs_record)
    outputs: list[Path] = []
    for job in jobs_record["jobs"]:
        output = Path(_argument(job["argv"], "--output"))
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            output,
            {
                "schema": matrix.RESULT_SCHEMA,
                "run_status": "completed",
                "evaluation_contract": {"probe": True},
                "evaluation_contract_digest": "a" * 64,
                "decision_plan": {"probe": True},
                "decision_plan_digest": "b" * 64,
            },
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
    write_json(batch_task.output, ledger)
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


@pytest.fixture
def fake_artifacts(monkeypatch):
    monkeypatch.setattr(matrix, "_artifact_attestation", _fake_artifact_attestation)
    monkeypatch.setattr(matrix, "_embedded_artifact_record", _fake_embedded)


def _output_from_argv(argv: list[str]) -> Path:
    flag = "--batch-record" if "--batch-record" in argv else "--output"
    return Path(_argument(argv, flag))


def _completed_dag(tmp_path, workers):
    dag = matrix.build_dag(_load(tmp_path))
    calls = []
    serial = 0

    def succeed(argv, **_):
        nonlocal serial
        output = _output_from_argv(argv)
        calls.append(str(output))
        serial += 1
        output.write_text(f"artifact:{serial}", encoding="utf-8")

    matrix.execute_dag(dag, subprocess_runner=succeed, verify_inputs=False, max_workers=workers)
    assert len(calls) == len(dag.tasks)
    calls.clear()
    return dag, calls, succeed


def _resume_dag(dag, calls, succeed, workers):
    matrix.execute_dag(
        dag, resume=True, subprocess_runner=succeed, verify_inputs=False, max_workers=workers
    )
    state = read_json(matrix._journal_path(dag))
    expected_tasks = {
        "checkpoint_batch/source_a/rep01",
        "result_batch/source_a/rep01/p01",
        "result_batch/source_a/rep01/p02",
        "manifest",
        "analysis",
    }
    assert set(calls) == {str(dag.task_by_id[key].output) for key in expected_tasks}
    assert len(calls) == len(expected_tasks)
    assert state["status"] == "completed"
    assert all(record["status"] == "completed" for record in state["tasks"].values())
    return state["tasks"]["checkpoint_batch/source_a/rep01"]


def test_resume_reruns_changed_output_and_transitive_dependents(tmp_path, fake_artifacts):
    dag, calls, succeed = _completed_dag(tmp_path, workers=1)
    dag.task_by_id["checkpoint_batch/source_a/rep01"].output.write_text(
        "tampered", encoding="utf-8"
    )
    _resume_dag(dag, calls, succeed, workers=1)


def test_resume_reruns_changed_log_and_transitive_dependents(tmp_path, fake_artifacts):
    dag, calls, succeed = _completed_dag(tmp_path, workers=3)
    state = read_json(matrix._journal_path(dag))
    first = state["tasks"]["checkpoint_batch/source_a/rep01"]
    log = first["attempts"][-1]["logs"]["stdout"]
    path = dag.plan.output_root / log["path"].removeprefix("$OUTPUT_ROOT/")
    path.write_text("tampered log", encoding="utf-8")
    resumed = _resume_dag(dag, calls, succeed, workers=3)
    assert resumed["resume_validation"]["reason"] == "stdout log is missing or changed size"


def test_resume_orphans_running_attempt_and_invalidates_dependents(tmp_path, fake_artifacts):
    dag, calls, succeed = _completed_dag(tmp_path, workers=3)
    journal_path = matrix._journal_path(dag)
    state = read_json(journal_path)
    first = state["tasks"]["checkpoint_batch/source_a/rep01"]
    first["status"] = first["attempts"][-1]["status"] = "running"
    first["finished_at"] = first["attempts"][-1]["finished_at"] = None
    state["status"] = "running"
    write_json(journal_path, state)
    resumed = _resume_dag(dag, calls, succeed, workers=3)
    assert resumed["attempts"][-2]["status"] == "orphaned"
    assert resumed["resume_validation"]["status"] == "rerun"


def test_attested_inputs_are_bound_to_journal_and_cache_tamper_blocks_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_artifacts
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

    matrix.execute_dag(dag, subprocess_runner=succeed)
    journal = read_json(dag.plan.output_root / "promotion.journal.json")
    resolved = journal["resolved_inputs"]
    assert resolved["source_snapshot"]["manifest_sha256"]
    assert resolved["target_cache"]["cache_sha256"]
    assert resolved["target_cache"]["decoded_dataset_sha256"]
    assert (
        resolved["target_cache"]["verified_cache_attestation"]["sha256"]
        == (resolved["target_cache"]["cache_sha256"])
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
    journal = read_json(dag.plan.output_root / "promotion.journal.json")
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
    first_ready = [task.task_id for task in dag.tasks if not task.dependencies][:4]
    overlap_gate = threading.Barrier(len(first_ready))

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
        nonlocal active, peak
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
            if task.task_id in first_ready:
                overlap_gate.wait(timeout=10)
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
    journal = read_json(matrix._journal_path(dag))
    assert journal["status"] == "completed"
    assert journal["execution"]["max_workers"] == 4
    assert journal["execution"]["state_writer"] == "orchestrator_main_thread_only"
    assert journal["execution"]["orchestrator_lock"]["dag_digest"] == journal["dag_digest"]
    assert journal["dispatch_sequence"][: len(first_ready)] == first_ready
    assert set(journal["dispatch_sequence"]) == set(calls)
    assert (
        len(journal["dispatch_sequence"])
        == len(set(journal["dispatch_sequence"]))
        == len(dag.tasks)
    )
    assert all(record["status"] == "completed" for record in journal["tasks"].values())
    assert all(len(record["attempts"]) == 1 for record in journal["tasks"].values())
    for record in journal["tasks"].values():
        for log in record["attempts"][0]["logs"].values():
            log_path = dag.plan.output_root / Path(log["path"].removeprefix("$OUTPUT_ROOT/"))
            assert hashlib.sha256(log_path.read_bytes()).hexdigest() == log["sha256"]


def test_parallel_failure_stops_dispatch_drains_and_resumes_strictly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_artifacts
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    task_by_output = {task.output.resolve(): task for task in dag.tasks}
    release_siblings = threading.Event()
    failure_journal_seen = threading.Event()
    initial_calls: list[str] = []
    journal_snapshot: dict[str, Any] = {}
    journal_path = dag.plan.output_root / "promotion.journal.json"
    original_atomic = matrix._atomic_write_json

    def fail_first_and_block_siblings(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv).resolve()
        task = task_by_output[output]
        initial_calls.append(task.task_id)
        if task.task_id == dag.tasks[0].task_id:
            raise subprocess.CalledProcessError(19, argv)
        assert release_siblings.wait(timeout=10)
        output.write_text(f"initial:{task.task_id}", encoding="utf-8")

    def observe_completed_write(path: Path, value: object) -> None:
        original_atomic(path, value)
        if (
            path == journal_path
            and value["status"] == "failed"
            and not failure_journal_seen.is_set()
        ):
            journal_snapshot.update(json.loads(json.dumps(value)))
            failure_journal_seen.set()
            release_siblings.set()

    monkeypatch.setattr(matrix, "_atomic_write_json", observe_completed_write)

    with pytest.raises(subprocess.CalledProcessError):
        matrix.execute_dag(
            dag,
            subprocess_runner=fail_first_and_block_siblings,
            verify_inputs=False,
            max_workers=4,
        )
    assert failure_journal_seen.is_set()
    assert journal_snapshot["tasks"][dag.tasks[0].task_id]["status"] == "failed"
    first_ready = [task.task_id for task in dag.tasks if not task.dependencies][:4]
    assert set(initial_calls) == set(first_ready)
    failed_state = read_json(matrix._journal_path(dag))
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
    resumed = read_json(matrix._journal_path(dag))
    assert resumed["status"] == "completed"
    assert dag.tasks[0].task_id in resume_calls
    initial_successes = set(initial_calls) - {dag.tasks[0].task_id}
    assert not initial_successes & set(resume_calls)
    assert len(resume_calls) == 11
    assert all(record["status"] == "completed" for record in resumed["tasks"].values())


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_artifacts
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

    with pytest.raises(KeyboardInterrupt):
        matrix.execute_dag(
            dag,
            subprocess_runner=succeed,
            verify_inputs=False,
            max_workers=3,
        )

    state = read_json(matrix._journal_path(dag))
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
    state = read_json(matrix._journal_path(dag))
    assert state["status"] == "interrupted_draining"
    assert "running" not in {record["status"] for record in state["tasks"].values()}
    matrix._release_orchestrator_lock(
        dag,
        read_json(lock_path),
    )


@pytest.mark.parametrize("requested,effective", [(16, 16), (100, 25)])
def test_result_worker_budget_and_phase_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_artifacts, requested, effective
) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    monkeypatch.setattr(matrix, "available_cpu_threads", lambda: 25)

    def succeed(argv: list[str], **_: Any) -> None:
        output = _output_from_argv(argv)
        output.write_text(f"artifact:{output.name}", encoding="utf-8")

    matrix.execute_dag(
        dag,
        subprocess_runner=succeed,
        verify_inputs=False,
        max_workers=1,
        result_max_workers=requested,
    )
    journal = read_json(matrix._journal_path(dag))
    execution = journal["execution"]
    assert execution["max_workers"] == 1
    assert execution["requested_result_max_workers"] == requested
    assert execution["result_max_workers"] == effective
    assert execution["available_cpu_threads"] == 25
    assert execution["phase_policy"] == "checkpoint_phase_then_result_phase"
    checkpoints = [task.task_id for task in dag.tasks if task.kind == "checkpoint_batch"]
    result_batches = [task.task_id for task in dag.tasks if task.kind == "result_batch"]
    sequence = journal["dispatch_sequence"]
    assert max(sequence.index(task_id) for task_id in checkpoints) < min(
        sequence.index(task_id) for task_id in result_batches
    )


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_execute_rejects_invalid_max_workers(tmp_path: Path, workers: object) -> None:
    dag = matrix.build_dag(_load(tmp_path))
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        matrix.execute_dag(
            dag,
            verify_inputs=False,
            max_workers=workers,
        )
