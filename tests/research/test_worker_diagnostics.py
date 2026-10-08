from __future__ import annotations

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.worker_diagnostics import (
    PoolFailureDiagnostics,
    diagnostic_task,
    initialize_worker_diagnostics,
    worker_diagnostic_phase,
)


def _tiny_worker(arguments):
    output, mode = arguments
    with diagnostic_task(output, {"phase_kind": "tiny_process_probe", "mode": mode}):
        assert not Path(output).exists()
        worker_diagnostic_phase("tiny_input_admitted", source_shape=[3, 2, 5], source_dtype="float32")
        print(f"stdout-{mode}", flush=True)
        print(f"stderr-{mode}", file=sys.stderr, flush=True)
        os.write(2, f"native-fd2-{mode}\n".encode())
        if mode == "python_exception":
            raise ValueError("tiny worker Python failure")
        if mode == "native_exit":
            worker_diagnostic_phase("before_native_exit7")
            os._exit(7)
        return os.getpid()


def _peer(arguments):
    output, ready, release = arguments
    with diagnostic_task(output, {"phase_kind": "peer_probe"}):
        worker_diagnostic_phase("peer_waiting")
        ready.set()
        release.wait(60)


def _reuse_worker(output):
    with pytest.raises(ValueError):
        with diagnostic_task(output, {"phase_kind": "first"}):
            worker_diagnostic_phase("old_fit", source_shape=[9, 3, 5], source_seed=2026,
                                    resolved_config={"old": True}, source_cache_sha256="f" * 64)
            raise ValueError("first task failure")
    with diagnostic_task(str(Path(output).with_name("second_output")), {"phase_kind": "second"}) as observer:
        return dict(observer.state)


def _persistence_failure_worker(output):
    import research.worker_diagnostics as module

    with diagnostic_task(output, {"phase_kind": "failed_persistence_probe"}):
        def unwritable(*args, **kwargs):
            raise PermissionError("diagnostic storage became unwritable")

        original = module.atomic_write_json
        module.atomic_write_json = unwritable
        try:
            try:
                raise RuntimeError("original fit failure")
            except RuntimeError as error:
                worker_diagnostic_phase("source_fit_python_failure", failure={
                    "type": type(error).__name__, "message": str(error)})
                raise
        finally:
            module.atomic_write_json = original


def _malformed_preparation_worker(output):
    from experiments.run_source_training_chain import _source_preparation_worker

    def prepare(*args, **kwargs):
        Path(output).mkdir()
        (Path(output) / "preparation.json").write_text('{"status":')

    sys.modules["experiments.prepare_source_records"] = SimpleNamespace(main=prepare)
    sys.modules["train.runtime"] = SimpleNamespace(cpu_thread_budget=lambda threads: nullcontext())
    staging = SimpleNamespace(reserve=lambda phase: nullcontext())
    return _source_preparation_worker(("manifest.json", output, "profile.json", 1, staging))


def test_preparation_result_parse_failure_is_inside_diagnostic_task(tmp_path):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker_diagnostics,
                             initargs=(str(tmp_path / "diagnostics"), {"pool_instance_id": "preparation-probe"})) as pool:
        with pytest.raises(json.JSONDecodeError):
            pool.submit(_malformed_preparation_worker, str(tmp_path / "preparation_output")).result(timeout=30)
    state = json.loads(next((tmp_path / "diagnostics").glob("*.state.json")).read_text())
    assert state["status"] == "failed" and state["failure"]["type"] == "JSONDecodeError"
    assert state["role"]["phase_kind"] == "source_preparation"


def test_failed_worker_diagnostic_write_preserves_python_failure(tmp_path):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker_diagnostics,
                             initargs=(str(tmp_path / "diagnostics"), {"pool_instance_id": "write-failure"})) as pool:
        with pytest.raises(RuntimeError, match="original fit failure"):
            pool.submit(_persistence_failure_worker, str(tmp_path / "scientific_output")).result(timeout=30)
    state = json.loads(next((tmp_path / "diagnostics").glob("*.state.json")).read_text())
    assert state["failure"]["type"] == "RuntimeError"
    persistence = state["diagnostic_persistence_failure"]
    assert persistence["type"] == "PermissionError" and persistence["phase"] == "source_fit_python_failure"
    assert persistence["primary_failure"] == state["failure"]


def test_failed_parent_diagnostic_write_preserves_pool_observation(tmp_path, monkeypatch):
    import research.worker_diagnostics as module

    observer = PoolFailureDiagnostics(tmp_path / "diagnostics", {"actual_run_sha256": "e" * 64})

    def unwritable(*args, **kwargs):
        raise PermissionError("diagnostic storage became unwritable")

    monkeypatch.setattr(module, "atomic_write_json", unwritable)
    result = observer.capture(BrokenProcessPool("original pool death"), {}, stage="future_result")
    assert observer.first["exception"] == {"type": "BrokenProcessPool", "message": "original pool death"}
    assert len(result["persistence_errors"]) == 2
    assert all(error["type"] == "PermissionError" for error in result["persistence_errors"])
    assert not (observer.directory / "first_pool_failure.json").exists()


def test_process_handles_survive_manager_clearing_executor_registry(tmp_path, monkeypatch):
    import research.worker_diagnostics as module

    observer = PoolFailureDiagnostics(tmp_path / "diagnostics", {"actual_run_sha256": "e" * 64})
    handles = {17: object(), 18: object()}
    executor = SimpleNamespace(_processes=dict(handles))

    def process_identity(pid):
        executor._processes.clear()
        return SimpleNamespace(create_time=lambda: 123.0, ppid=lambda: 11, exe=lambda: "python")

    monkeypatch.setattr(module.psutil, "Process", process_identity)
    observer.remember_processes(executor)
    assert observer.processes == handles
    assert all(identity["create_time"] == 123.0 for identity in observer.process_identities.values())


def _resource_sensor_failure_worker(output):
    import research.worker_diagnostics as module

    with diagnostic_task(output, {"phase_kind": "failed_sensor_probe"}):
        def unreadable():
            raise PermissionError("resource sensor became unavailable")

        original = module.psutil.virtual_memory
        module.psutil.virtual_memory = unreadable
        try:
            try:
                raise RuntimeError("original fit failure")
            except RuntimeError as error:
                worker_diagnostic_phase("source_fit_python_failure", failure={
                    "type": type(error).__name__, "message": str(error)})
                raise
        finally:
            module.psutil.virtual_memory = original


def test_failed_resource_sensor_preserves_original_fit_failure(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker_diagnostics,
                             initargs=(str(diagnostics), {"pool_instance_id": "sensor-failure"})) as pool:
        with pytest.raises(RuntimeError, match="original fit failure"):
            pool.submit(_resource_sensor_failure_worker, str(tmp_path / "scientific_output")).result(timeout=30)
    events = [json.loads(line) for line in next(diagnostics.glob("*.events.jsonl")).read_text().splitlines()]
    failure = next(event for event in events if event["phase"] == "source_fit_python_failure")
    assert failure["resources"]["status"] == "unavailable"
    assert failure["resources"]["error"]["type"] == "PermissionError"
    assert events[-1]["failure"]["type"] == "RuntimeError"


def test_failed_parent_resource_sensor_preserves_pool_failure(tmp_path, monkeypatch):
    import research.worker_diagnostics as module

    observer = PoolFailureDiagnostics(tmp_path / "diagnostics", {"actual_run_sha256": "e" * 64})

    def unreadable():
        raise PermissionError("resource sensor became unavailable")

    monkeypatch.setattr(module.psutil, "virtual_memory", unreadable)
    observer.capture(BrokenProcessPool("original pool death"), {}, stage="future_result")
    failure = json.loads((observer.directory / "first_pool_failure.json").read_text())
    assert failure["exception"]["type"] == "BrokenProcessPool"
    assert failure["parent_resources"]["status"] == "unavailable"
    assert failure["parent_resources"]["error"]["type"] == "PermissionError"


def test_reused_worker_starts_without_previous_task_facts(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker_diagnostics,
                             initargs=(str(diagnostics), {"pool_instance_id": "reuse-probe"})) as pool:
        state = pool.submit(_reuse_worker, str(tmp_path / "first_output")).result(timeout=30)
    assert state["role"] == {"phase_kind": "second"}
    assert not {"source_shape", "source_seed", "resolved_config", "source_cache_sha256", "failure"} & state.keys()
    events = [json.loads(line) for line in next(diagnostics.glob("*.events.jsonl")).read_text().splitlines()]
    assert any(event.get("source_seed") == 2026 for event in events)


def test_failure_capture_keeps_original_error_when_state_or_phase_is_unreadable(tmp_path):
    observer = PoolFailureDiagnostics(tmp_path / "diagnostics", {"actual_run_sha256": "e" * 64})
    observer.directory.mkdir(parents=True, exist_ok=True)
    (observer.directory / "bad.state.json").write_text('{"pid":')
    output = tmp_path / "scientific_output"
    output.mkdir()
    (output / "source_input_stages.json").write_bytes(b"\xff")
    (observer.directory / "current.state.json").write_text(json.dumps({
        "identity": observer.identity, "pid": 17, "create_time": 123.0, "output": str(output)}))
    observer.capture(BrokenProcessPool("original pool death"), {"outer": "held_out"}, stage="future_result")
    failure = json.loads((observer.directory / "first_pool_failure.json").read_text())
    assert failure["exception"] == {"type": "BrokenProcessPool", "message": "original pool death"}
    bad = next(item for item in failure["worker_states"] if item["state_path"].endswith("bad.state.json"))
    assert bad["state_status"] == "unreadable" and bad["state_sha256"]
    current = next(item for item in failure["worker_states"] if item.get("worker", {}).get("pid") == 17)
    assert current["saved_phase_state"]["source_input_stages.json"]["status"] == "unreadable"
    assert current["saved_phase_state"]["worker_progress.json"]["status"] == "unavailable"


def test_failure_capture_excludes_states_from_another_pool_instance(tmp_path):
    root = tmp_path / "diagnostics"
    first = PoolFailureDiagnostics(root, {"actual_run_sha256": "e" * 64})
    second = PoolFailureDiagnostics(root, {"actual_run_sha256": "e" * 64})
    assert first.directory != second.directory
    second.directory.mkdir(parents=True, exist_ok=True)
    (second.directory / "historical.state.json").write_text(json.dumps({
        "identity": first.identity, "pid": 19, "create_time": 123.0}))
    second.capture(BrokenProcessPool("current pool death"), {"outer": "held_out"}, stage="future_result")
    failure = json.loads((second.directory / "first_pool_failure.json").read_text())
    assert failure["worker_states"] == []
    assert failure["excluded_worker_states"][0]["worker"]["pid"] == 19


def test_worker_stderr_phase_and_python_exception_leave_output_empty(tmp_path):
    context = multiprocessing.get_context("spawn")
    diagnostics = tmp_path / "diagnostics"
    identity = {"source_archive_sha256": "a" * 64, "actual_run_sha256": "b" * 64}
    with ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=initialize_worker_diagnostics,
                             initargs=(str(diagnostics), identity)) as pool:
        pid = pool.submit(_tiny_worker, (str(tmp_path / "scientific_output"), "normal")).result(timeout=30)
        with pytest.raises(ValueError, match="tiny worker Python failure"):
            pool.submit(_tiny_worker, (str(tmp_path / "second_output"), "python_exception")).result(timeout=30)
    state = json.loads(next(diagnostics.glob("*.state.json")).read_text())
    assert state["pid"] == pid and state["create_time"] > 0
    assert state["identity"] == identity and state["faulthandler_enabled"]
    assert state["phase"] == "python_exception" and state["failure"]["type"] == "ValueError"
    stderr = Path(state["stderr_path"]).read_text(encoding="utf-8")
    assert "stderr-normal" in stderr and "stderr-python_exception" in stderr and "Traceback" in stderr
    assert "native-fd2-normal" in stderr and "native-fd2-python_exception" in stderr
    assert not (tmp_path / "scientific_output").exists() and not (tmp_path / "second_output").exists()
    assert not (diagnostics / "first_pool_failure.json").exists()


def test_native_exit_captures_exit7_and_separates_peer_pool_failure(tmp_path):
    context = multiprocessing.get_context("spawn")
    diagnostics = tmp_path / "diagnostics"
    identity = {"source_archive_sha256": "c" * 64, "actual_run_sha256": "d" * 64}
    observer = PoolFailureDiagnostics(diagnostics, identity)
    diagnostics, identity = observer.directory, observer.identity
    with context.Manager() as manager:
        ready, release = manager.Event(), manager.Event()
        with ProcessPoolExecutor(max_workers=2, mp_context=context, initializer=initialize_worker_diagnostics,
                                 initargs=(str(diagnostics), identity)) as pool:
            peer = pool.submit(_peer, (str(tmp_path / "peer_output"), ready, release))
            assert ready.wait(15)
            # Complete a task on the second worker before inducing death. In
            # CPython 3.14 a wakeup can precede dynamic worker registration;
            # this probe tests diagnostics with all sentinels already known.
            pool.submit(_tiny_worker, (str(tmp_path / "warm_output"), "normal")).result(timeout=30)
            fatal = pool.submit(_tiny_worker, (str(tmp_path / "native_output"), "native_exit"))
            observer.remember_processes(pool)
            with pytest.raises(BrokenProcessPool) as first:
                fatal.result(timeout=30)
            original = observer.capture(first.value, {"mode": "native_exit"}, stage="future_result")
            original_bytes = (diagnostics / "first_pool_failure.json").read_bytes()
            with pytest.raises(BrokenProcessPool) as second:
                peer.result(timeout=30)
            followup = observer.capture(second.value, {"mode": "peer_waiting"}, stage="future_result")
            assert (diagnostics / "first_pool_failure.json").read_bytes() == original_bytes
    record = json.loads(original_bytes)
    assert original["relationship"] == "first_parent_observation"
    assert followup["relationship"] == "same_pool_failure_followup"
    assert record["native_first_cause"] == "unknown"
    assert any(process["exit_code"] == 7 for process in record["process_exit_observations"])
    assert all(process["create_time"] for process in record["process_exit_observations"])
    assert len(record["worker_states"]) == 2
    native = next(item["worker"] for item in record["worker_states"] if item["worker"]["role"]["phase_kind"] == "tiny_process_probe")
    assert native["status"] == "running" and native["phase"] == "before_native_exit7"
    assert "stderr-native_exit" in Path(native["stderr_path"]).read_text(encoding="utf-8")
    assert native["source_shape"] == [3, 2, 5] and native["identity"] == identity
    assert not (tmp_path / "native_output").exists()
