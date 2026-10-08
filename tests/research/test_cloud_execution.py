"""Behavior probes for the shared executor; no EEG arrays or model fits."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import psutil
import pytest

from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
from research.source_execution import (
    HostMemoryAdmission,
    SourceInputAdmission,
    current_input_admission,
)
from research.worker_diagnostics import worker_diagnostic_phase

MIB = 2**20
ADAPTER = "tests.research.test_cloud_execution:_probe"


def _probe(arguments):
    mode, output = arguments["mode"], Path(arguments["output"])
    worker_diagnostic_phase("probe_started", probe_mode=mode)
    if mode == "native_exit":
        os._exit(7)
    if mode == "native_input_exit":
        with current_input_admission().reserve("nn", 8):
            os._exit(7)
    if mode == "native_artifact_oversize":
        temporary = output.with_suffix(".part")
        with current_input_admission().artifact_writer(output, 8, temporary_paths=(temporary,)):
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(b"native-partial16")
            os._exit(7)
    if mode == "native_nested_exit":
        with arguments["host_admission"].target_workers(2, device="cpu") as workers:
            assert workers == 2
            os._exit(7)
    if mode in {"native_exit_with_child", "native_exit_with_forked_child"}:
        method = "fork" if mode.endswith("forked_child") else "spawn"
        child = multiprocessing.get_context(method).Process(target=_grandchild, args=(arguments["child_record"],))
        child.start()
        deadline = time.monotonic() + 10
        while not Path(arguments["child_record"]).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("grandchild did not initialize")
            time.sleep(.02)
        worker_diagnostic_phase("grandchild_started", descendant_pid=child.pid)
        os._exit(7)
    if mode == "python_failure":
        raise ValueError("original science failure")
    if mode == "wait_marker":
        deadline = time.monotonic() + 20
        while not Path(arguments["marker"]).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("dependent task never started")
            time.sleep(.02)
    if arguments.get("delay"):
        time.sleep(arguments["delay"])
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "mode": mode, "native_threads": os.environ["OMP_NUM_THREADS"]}
    output.write_text(json.dumps(record))
    return str(output)


def _grandchild(path):
    Path(path).write_text(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time()}))
    time.sleep(30)


def _cuda_probe(arguments):
    import torch

    value = torch.zeros(16, device="cuda")
    record = {"elements": value.numel(), "device": str(value.device)}
    output = Path(arguments["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record))
    return str(output)


def _task(root, key, mode="write", *, depends_on=(), phase="source", **arguments):
    output = root / key / "artifact.json"
    return CloudTask(key, ADAPTER, {"mode": mode, "output": str(output), **arguments},
                     str(output.parent), PhaseResources(phase, 1, 128 * MIB, 1, "cpu"), tuple(depends_on))


def _admission(manager):
    return HostMemoryAdmission(manager, {
        "host_memory_budget_bytes": 256 * MIB, "shared_staging_memory_bytes": 0,
        "source_worker_memory_bytes": 128 * MIB, "target_worker_memory_bytes": 128 * MIB,
        "source_workers": 2, "target_workers": 2, "gpu_slots": 2,
        "native_threads_per_worker": 1, "cpu_budget": 3, "coordinator_cpu_threads": 1})


def _executor(root, admission, **options):
    return CloudExecutor(admission, max_workers=2, diagnostic_directory=root / "diagnostics",
                         identity={"source_archive_sha256": "a" * 64}, poll_seconds=.1, **options)


def test_completed_dependency_unlocks_work_before_other_source_job_finishes(tmp_path):
    marker = tmp_path / "followup" / "artifact.json"
    context = multiprocessing.get_context("spawn")
    snapshots = []
    with context.Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()

        def accept(outcome):
            if outcome.task_id == "fast" and outcome.status == "completed":
                assert Path(outcome.value).exists()
                assert outcome.worker["alive"] is False
                return [_task(tmp_path, "followup", phase="preparation", depends_on=("fast",))]
            return []

        run = _executor(tmp_path, admission).run([
            _task(tmp_path, "slow", "wait_marker", marker=str(marker)), _task(tmp_path, "fast")],
            on_finished=accept, on_observation=snapshots.append)
        assert admission.record() == initial
    assert run.status == "completed" and set(run.results) == {"slow", "fast", "followup"}
    assert len({outcome.worker["pid"] for outcome in run.results.values()}) == 3
    assert all(json.loads(Path(outcome.value).read_text())["native_threads"] == "1"
               for outcome in run.results.values())
    assert all(row["resources"]["reserved_cpu_threads"] <= 3 for row in snapshots)
    assert all(row["resources"]["free_worker_memory_bytes"] >= 0 for row in snapshots)
    assert any("followup" in row["running"] and "slow" in row["running"] for row in snapshots)


def test_native_exit_is_terminal_and_does_not_kill_an_independent_peer(tmp_path):
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()
        run = _executor(tmp_path, admission).run([
            _task(tmp_path, "fatal", "native_exit"), _task(tmp_path, "peer", delay=.3),
            _task(tmp_path, "dependent", depends_on=("fatal",))])
        assert admission.record() == initial
    assert run.status == "failed" and len(run.results) == 3
    fatal = run.results["fatal"]
    assert fatal.status == "failed" and fatal.worker["exit_code"] == 7
    assert fatal.worker["create_time"] > 0 and fatal.worker["alive"] is False
    assert run.results["peer"].status == "completed" and Path(run.results["peer"].value).exists()
    assert run.results["dependent"].status == "blocked"
    assert not (tmp_path / "dependent").exists()
    first = json.loads((Path(run.diagnostics["directory"]) / "first_pool_failure.json").read_text())
    assert first["native_first_cause"] == "unknown"
    assert first["task"]["task_id"] == "fatal" and first["stage"] == "process_exit"


def test_python_failure_keeps_original_error_and_blocks_only_its_descendants(tmp_path):
    with multiprocessing.get_context("spawn").Manager() as manager:
        run = _executor(tmp_path, _admission(manager)).run([
            _task(tmp_path, "bad", "python_failure"), _task(tmp_path, "peer"),
            _task(tmp_path, "dependent", depends_on=("bad",))])
    assert run.results["bad"].failure["type"] == "ValueError"
    assert run.results["bad"].failure["message"] == "original science failure"
    assert run.results["bad"].worker["exit_code"] == 0
    assert run.results["dependent"].status == "blocked" and run.results["peer"].status == "completed"


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_native_actor_exit_retires_its_grandchild_before_returning_phase_budget(tmp_path, method):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip("fork unavailable on this platform")
    child_record = tmp_path / "child.json"
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()
        run = _executor(tmp_path, admission).run([
            _task(tmp_path, "fatal", "native_exit_with_forked_child" if method == "fork" else "native_exit_with_child",
                  child_record=str(child_record)),
            _task(tmp_path, "peer", delay=.5)])
        assert admission.record() == initial
    child = json.loads(child_record.read_text())
    try:
        process = psutil.Process(child["pid"])
        still_owned = process.create_time() == child["create_time"]
        assert not still_owned or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    assert run.results["peer"].status == "completed"
    assert run.results["fatal"].worker["process_tree_retirement"]["active_owned_processes_after_exit"] == 0


@pytest.mark.parametrize("mode", ["native_input_exit", "native_nested_exit", "native_artifact_oversize"])
def test_native_exit_settles_same_owner_without_stranding_peer_or_budget(tmp_path, mode):
    with multiprocessing.get_context("spawn").Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=16, load_workers=2)
        admission = HostMemoryAdmission(manager, {
            "host_memory_budget_bytes": 512 * MIB + 16, "shared_staging_memory_bytes": 16,
            "source_worker_memory_bytes": 128 * MIB, "target_worker_memory_bytes": 128 * MIB,
            "source_workers": 2, "target_workers": 2, "gpu_slots": 2,
            "native_threads_per_worker": 1, "cpu_budget": 5, "coordinator_cpu_threads": 1},
            input_admission=inputs)
        initial = admission.record()
        arguments = {"host_admission": admission} if mode == "native_nested_exit" else {}
        run = _executor(tmp_path, admission).run([
            _task(tmp_path, "fatal", mode, **arguments), _task(tmp_path, "peer", delay=.2)])
        assert admission.record() == initial
        record = inputs.record()
    assert run.results["fatal"].failure["stage"] == "process_exit"
    assert run.results["peer"].status == "completed"
    assert record["free_staging_bytes"] == 16 and record["nn_active"] == 0 and record["nn_waiting"] == 0
    if mode == "native_artifact_oversize":
        assert (tmp_path / "fatal" / "artifact.part").read_bytes() == b"native-partial16"
        lease = next(row for row in record["active_leases"] if row["path"] == str((tmp_path / "fatal" / "artifact.json").resolve()))
        assert lease["disk_bytes"] == 16
        assert run.results["fatal"].supplementary_failures[0]["stage"] == "artifact_settlement"
    else:
        assert all(row["kind"] == "persistent_artifact" for row in record["active_leases"])
        assert not any(row["path"].startswith(str(tmp_path / "fatal")) for row in record["active_leases"])
    assert all(row["disk_bytes"] == sum(Path(path).stat().st_size for path in row["paths"] if Path(path).exists())
               - row["baseline_bytes"] for row in record["active_leases"])


def test_exit_zero_does_not_admit_an_artifact_rejected_by_the_scientific_owner(tmp_path):
    def reject(outcome):
        if outcome.task_id == "producer":
            raise ValueError("actual producer binding was not admitted")
        return []

    with multiprocessing.get_context("spawn").Manager() as manager:
        run = _executor(tmp_path, _admission(manager)).run([
            _task(tmp_path, "producer"), _task(tmp_path, "dependent", depends_on=("producer",))],
            on_finished=reject)
    assert run.results["producer"].worker["exit_code"] == 0
    assert run.results["producer"].status == "failed"
    assert run.results["producer"].failure["stage"] == "artifact_admission"
    assert run.results["dependent"].status == "blocked"


def test_unreachable_dependency_and_cycle_remain_in_the_requested_denominator(tmp_path):
    with multiprocessing.get_context("spawn").Manager() as manager:
        run = _executor(tmp_path, _admission(manager)).run([
            _task(tmp_path, "missing", depends_on=("not_requested",)),
            _task(tmp_path, "cycle_a", depends_on=("cycle_b",)),
            _task(tmp_path, "cycle_b", depends_on=("cycle_a",))])
    assert len(run.results) == 3 and all(outcome.status == "blocked" for outcome in run.results.values())
    assert all(outcome.worker is None for outcome in run.results.values())


def test_spawn_failure_returns_the_reservation_exactly_once(tmp_path, monkeypatch):
    import research.cloud_execution as module

    def cannot_spawn(*args, **kwargs):
        raise OSError("process creation failed")

    monkeypatch.setattr(module, "_make_process", cannot_spawn)
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()
        run = _executor(tmp_path, admission).run([_task(tmp_path, "spawn_failed")])
        assert admission.record() == initial
    assert run.results["spawn_failed"].failure["type"] == "OSError"
    assert run.results["spawn_failed"].failure["stage"] == "spawn"


def test_pipe_creation_failure_returns_the_reservation_exactly_once(tmp_path, monkeypatch):
    def cannot_pipe(*args, **kwargs):
        raise OSError("result pipe creation failed")

    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()
        executor = _executor(tmp_path, admission)
        monkeypatch.setattr(executor.context, "Pipe", cannot_pipe)
        run = executor.run([_task(tmp_path, "pipe_failed")])
        assert admission.record() == initial
    assert run.results["pipe_failed"].failure["type"] == "OSError"
    assert run.results["pipe_failed"].failure["message"] == "result pipe creation failed"


@pytest.mark.parametrize("error_type", [ValueError, OSError])
def test_bootstrap_containment_failure_preserves_independent_peer(tmp_path, monkeypatch, error_type):
    import research.cloud_execution as module

    admit = module._admit_bootstrap

    def reject_one(item, message):
        if item.task.task_id == "rejected":
            raise error_type("owned containment rejected")
        admit(item, message)

    monkeypatch.setattr(module, "_admit_bootstrap", reject_one)
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()
        run = _executor(tmp_path, admission).run([
            _task(tmp_path, "rejected"), _task(tmp_path, "peer", delay=.2),
            _task(tmp_path, "descendant", depends_on=("rejected",))])
        assert admission.record() == initial
    assert run.results["rejected"].failure["type"] == error_type.__name__
    assert run.results["rejected"].failure["message"] == "owned containment rejected"
    assert run.results["peer"].status == "completed"
    assert run.results["descendant"].status == "blocked"
    assert not (tmp_path / "rejected" / "artifact.json").exists()


def test_observer_error_cannot_replace_the_original_scientific_failure(tmp_path):
    def failed_observer(snapshot):
        raise PermissionError("observation storage unavailable")

    with multiprocessing.get_context("spawn").Manager() as manager:
        run = _executor(tmp_path, _admission(manager)).run([_task(tmp_path, "bad", "python_failure")],
            on_observation=failed_observer)
    assert run.results["bad"].failure["type"] == "ValueError"
    assert run.observation_failures and run.observation_failures[0]["type"] == "PermissionError"


def test_duplicate_task_ids_are_rejected_before_work_is_reserved(tmp_path):
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()
        with pytest.raises(ValueError, match="duplicate"):
            _executor(tmp_path, admission).run([_task(tmp_path, "same"), _task(tmp_path, "same")])
        assert admission.record() == initial


def test_initialized_cuda_sensor_and_cleanup_preserve_completed_artifact(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable for actual context cleanup probe")
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        admission = HostMemoryAdmission(manager, {
            "host_memory_budget_bytes": 4 * 2**30, "shared_staging_memory_bytes": 0,
            "source_worker_memory_bytes": 2 * 2**30, "target_worker_memory_bytes": 2 * 2**30,
            "source_workers": 1, "target_workers": 1, "gpu_slots": 1,
            "native_threads_per_worker": 1, "cpu_budget": 2, "coordinator_cpu_threads": 1})
        initial = admission.record()
        output = tmp_path / "cuda" / "artifact.json"
        task = CloudTask("cuda", "tests.research.test_cloud_execution:_cuda_probe", {"output": str(output)},
                         str(output.parent), PhaseResources("source", 1, 2 * 2**30, 1, "cuda"))
        run = _executor(tmp_path, admission).run([task])
        assert admission.record() == initial
    assert run.status == "completed"
    worker = run.results["cuda"].worker
    assert worker["exit_code"] == 0 and worker["alive"] is False
    assert worker["cleanup_before"]["device"]["status"] == "read"
    assert worker["cleanup_after"]["device"]["status"] == "read"
    assert json.loads(output.read_text())["elements"] == 16
