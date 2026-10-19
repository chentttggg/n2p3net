"""Scientific admission must preserve dependencies while ready work computes."""
from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from research.cloud_execution import CloudExecutor, CloudTask, PhaseResources
from research.source_execution import HostMemoryAdmission

MIB = 2**20
ADAPTER = "tests.research.test_cloud_admission_scheduling:_write"


def _write(arguments):
    path = Path(arguments["output"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump({"pid": os.getpid(), "computed_unix": time.time()}, stream)
    if arguments.get("wait_marker"):
        deadline = time.monotonic() + 30
        while not Path(arguments["wait_marker"]).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("probe release marker never arrived")
            time.sleep(.02)
    return str(path)


def _native_probe(arguments):
    import torch
    from threadpoolctl import threadpool_info

    return {"native": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
            "pools": [row["num_threads"] for row in threadpool_info()],
            "omp": os.environ["OMP_NUM_THREADS"], "task_id": os.environ["N2P3_CLOUD_TASK_ID"],
            "tree_token": os.environ["N2P3_CLOUD_TREE_TOKEN"]}


def _task(root, name, *, depends_on=(), **arguments):
    output = root / name / "artifact.json"
    return CloudTask(name, ADAPTER, {"output": str(output), **arguments}, str(output.parent),
                     PhaseResources("source", 1, 128 * MIB, 1, "cpu"), depends_on)


def _admission(manager, *, workers=1):
    return HostMemoryAdmission(manager, {
        "host_memory_budget_bytes": workers * 128 * MIB, "shared_staging_memory_bytes": 0,
        "source_worker_memory_bytes": 128 * MIB, "target_worker_memory_bytes": 128 * MIB,
        "source_workers": workers, "target_workers": workers, "gpu_slots": workers,
        "native_threads_per_worker": 1, "cpu_budget": workers + 1, "coordinator_cpu_threads": 1})


@pytest.mark.parametrize("reject", [False, True], ids=["admitted", "rejected"])
def test_ready_adapter_computes_during_parent_admission_without_unlocking_dependents(tmp_path, reject):
    sibling = tmp_path / "sibling" / "artifact.json"
    dependent = tmp_path / "dependent" / "artifact.json"
    admission_events, snapshots = [], []
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()

        def accept(outcome):
            admission_events.append(outcome.task_id)
            if outcome.task_id == "producer":
                assert outcome.worker["alive"] is False
                assert outcome.worker["process_tree_retirement"]["active_owned_processes_after_exit"] == 0
                assert not dependent.exists(), "process exit cannot unlock scientific dependencies"
                deadline = time.monotonic() + 10
                while not sibling.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                assert sibling.exists(), "ready adapter stayed idle during parent admission"
                assert not dependent.exists(), "dependent computed before science admission"
                if reject:
                    raise ValueError("declared scientific rejection")
            return ()

        run = CloudExecutor(admission, max_workers=1, diagnostic_directory=tmp_path / "diagnostics",
            identity={"source_archive_sha256": "a" * 64}, poll_seconds=.05).run([
                _task(tmp_path, "producer"), _task(tmp_path, "sibling"),
                _task(tmp_path, "dependent", depends_on=("producer",)),
            ], on_finished=accept, on_observation=snapshots.append)
        assert admission.record() == initial

    assert set(run.results) == {"producer", "sibling", "dependent"}
    assert sorted(admission_events) == ["dependent", "producer", "sibling"]
    assert run.results["sibling"].status == "completed"
    if reject:
        assert run.results["producer"].failure["message"] == "declared scientific rejection"
        assert run.results["dependent"].status == "blocked" and not dependent.exists()
    else:
        assert run.status == "completed" and dependent.exists()
        assert len({result.worker["pid"] for result in run.results.values()}) == 3
        assert all(set(row["admitting"]).isdisjoint(row["running"])
                   and set(row["admitting"]).isdisjoint(row["pending"]) for row in snapshots)
    assert all(row["resources"]["reserved_cpu_threads"] <= 2 for row in snapshots)


def test_no_active_worker_admits_producer_then_dispatches_dynamic_and_existing_dependents(tmp_path):
    events = []

    def accept(outcome):
        events.append(outcome.task_id)
        if outcome.task_id == "producer":
            assert not (tmp_path / "dependent").exists()
            return (_task(tmp_path, "dynamic", depends_on=("producer",)),)
        return ()

    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()
        run = CloudExecutor(admission, max_workers=1, diagnostic_directory=tmp_path / "diagnostics",
            identity={}, poll_seconds=.05).run([
                _task(tmp_path, "producer"), _task(tmp_path, "dependent", depends_on=("producer",)),
            ], on_finished=accept)
        assert admission.record() == initial
    assert run.status == "completed" and set(run.results) == {"producer", "dependent", "dynamic"}
    assert events == ["producer", "dependent", "dynamic"]


@pytest.mark.parametrize("failure", ["pipe", "spawn"])
def test_dispatch_failure_releases_capacity_before_its_long_admission(tmp_path, monkeypatch, failure):
    import research.cloud_execution as module

    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()
        executor = CloudExecutor(admission, max_workers=1, diagnostic_directory=tmp_path / "diagnostics",
                                 identity={}, poll_seconds=.05)
        if failure == "spawn":
            original = module._make_process

            def create(context, task, *args):
                if task.task_id == "failed":
                    raise OSError("declared spawn failure")
                return original(context, task, *args)

            monkeypatch.setattr(module, "_make_process", create)
        else:
            original = executor.context.Pipe
            calls = []

            def pipe(*args, **kwargs):
                calls.append(True)
                if len(calls) == 1:
                    raise OSError("declared pipe failure")
                return original(*args, **kwargs)

            monkeypatch.setattr(executor.context, "Pipe", pipe)

        def accept(outcome):
            if outcome.task_id == "failed":
                deadline = time.monotonic() + 10
                marker = tmp_path / "sibling" / "artifact.json"
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                assert marker.exists(), "dispatch failure blocked independent compute"
            return ()

        run = executor.run([_task(tmp_path, "failed"), _task(tmp_path, "sibling")], on_finished=accept)
        assert admission.record() == initial
    assert run.results["sibling"].status == "completed"
    failed = run.results["failed"]
    assert failed.failure["message"] == f"declared {failure} failure"
    assert not failed.supplementary_failures


@pytest.mark.parametrize("bootstrap", ["late", "rejected"])
def test_bootstrap_is_handled_during_long_admission_and_rejection_retires_peer(tmp_path, monkeypatch, bootstrap):
    import research.cloud_execution as module

    events = []
    original = module._admit_bootstrap

    def admit(item, message):
        events.append(("bootstrap", item.task.task_id))
        if item.task.task_id == "peer":
            if bootstrap == "rejected":
                raise ValueError("declared bootstrap rejection")
            time.sleep(.2)
        return original(item, message)

    monkeypatch.setattr(module, "_admit_bootstrap", admit)

    def accept(outcome):
        if outcome.task_id == "producer":
            deadline = time.monotonic() + 10
            while ("bootstrap", "peer") not in events and time.monotonic() < deadline:
                time.sleep(.02)
            assert ("bootstrap", "peer") in events
        events.append(("admission", outcome.task_id))
        return ()

    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager, workers=2)
        initial = admission.record()
        run = CloudExecutor(admission, max_workers=2, diagnostic_directory=tmp_path / "diagnostics",
            identity={}, poll_seconds=.05).run([
                _task(tmp_path, "producer"), _task(tmp_path, "peer"),
            ], on_finished=accept)
        assert admission.record() == initial
    assert run.results["producer"].status == "completed"
    if bootstrap == "rejected":
        assert run.results["peer"].failure["message"] == "declared bootstrap rejection"
        assert not (tmp_path / "peer").exists()
    else:
        assert run.status == "completed"


def test_callback_interrupt_retires_computing_peer_and_preserves_partial_artifact(tmp_path):
    import psutil

    peer = tmp_path / "peer" / "artifact.json"

    def interrupt(outcome):
        if outcome.task_id == "producer":
            deadline = time.monotonic() + 10
            while not peer.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert peer.exists(), "peer did not reach its actual adapter"
            raise KeyboardInterrupt("declared admission interruption")
        return ()

    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager, workers=2)
        initial = admission.record()
        with pytest.raises(KeyboardInterrupt, match="declared admission interruption"):
            CloudExecutor(admission, max_workers=2, diagnostic_directory=tmp_path / "diagnostics",
                identity={}, poll_seconds=.05).run([
                    _task(tmp_path, "producer"), _task(tmp_path, "peer", wait_marker=str(tmp_path / "release")),
                ], on_finished=interrupt)
        assert admission.record() == initial
    assert not psutil.pid_exists(json.loads(peer.read_text())["pid"])
    assert (tmp_path / "producer" / "artifact.json").exists() and peer.exists()


@pytest.mark.parametrize("reject", [False, True], ids=["admitted", "rejected"])
def test_short_worker_retires_and_tail_computes_while_producer_admission_is_still_open(tmp_path, reject):
    import threading

    tail, dependent = tmp_path / "tail" / "artifact.json", tmp_path / "dependent" / "artifact.json"
    callback_open = tmp_path / "callback_open"
    calling_thread = threading.get_ident()
    observations = []

    def observe(snapshot):
        assert threading.get_ident() == calling_thread
        observations.append(snapshot)

    def accept(outcome):
        assert threading.get_ident() == calling_thread
        if outcome.task_id == "producer":
            callback_open.write_text(str(time.time()))
            deadline = time.monotonic() + 10
            while not tail.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert tail.exists(), "completed sibling did not release capacity for the ready tail"
            assert not dependent.exists(), "tail progress cannot grant scientific authority to dependents"
            assert json.loads(tail.read_text())["computed_unix"] >= float(callback_open.read_text())
            if reject:
                raise ValueError("declared refill admission rejection")
        return ()

    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()
        run = CloudExecutor(admission, max_workers=1, diagnostic_directory=tmp_path / "diagnostics",
            identity={}, poll_seconds=.05).run([
                _task(tmp_path, "producer"), _task(tmp_path, "sibling", wait_marker=str(callback_open)),
                _task(tmp_path, "tail"),
                _task(tmp_path, "dependent", depends_on=("producer",)),
            ], on_finished=accept, on_observation=observe)
        assert admission.record() == initial
    assert len(run.results) == 4
    if reject:
        assert run.results["producer"].failure["message"] == "declared refill admission rejection"
        assert run.results["dependent"].status == "blocked" and not dependent.exists()
    else:
        assert run.status == "completed"
        assert len({result.worker["pid"] for result in run.results.values()}) == 4
    assert observations[-1]["running"] == observations[-1]["admitting"] == observations[-1]["pending"] == []


def test_mixed_worker_budgets_do_not_mutate_parent_environment_or_callback_thread(tmp_path, monkeypatch):
    import threading

    parent = {"OMP_NUM_THREADS": "7", "N2P3_CLOUD_TASK_ID": "parent", "N2P3_CLOUD_TREE_TOKEN": "parent-token"}
    for name, value in parent.items():
        monkeypatch.setenv(name, value)
    calling_thread, events = threading.get_ident(), []

    def accept(outcome):
        assert threading.get_ident() == calling_thread
        assert {name: os.environ[name] for name in parent} == parent
        value = outcome.value
        native = 2 if outcome.task_id == "two" else 3
        assert value["native"] == native and value["interop"] == 1
        assert all(count <= native for count in value["pools"])
        assert value["omp"] == str(native) and value["task_id"] == outcome.task_id
        assert value["tree_token"] != parent["N2P3_CLOUD_TREE_TOKEN"]
        events.append(value["tree_token"])
        return ()

    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, {
            "host_memory_budget_bytes": 256 * MIB, "shared_staging_memory_bytes": 0,
            "source_worker_memory_bytes": 128 * MIB, "target_worker_memory_bytes": 128 * MIB,
            "source_workers": 2, "target_workers": 2, "gpu_slots": 2,
            "native_threads_per_worker": 1, "cpu_budget": 8, "coordinator_cpu_threads": 1})
        initial = admission.record()
        tasks = [CloudTask(name, "tests.research.test_cloud_admission_scheduling:_native_probe", {},
                 str(tmp_path / name), PhaseResources("evaluation", threads + 1, 128 * MIB,
                                                      threads, "cpu", interop_threads=1))
                 for name, threads in (("two", 2), ("three", 3))]
        run = CloudExecutor(admission, max_workers=2, diagnostic_directory=tmp_path / "diagnostics",
                            identity={}, poll_seconds=.05).run(tasks, on_finished=accept)
        assert admission.record() == initial
    assert run.status == "completed" and len(set(events)) == 2
    assert {name: os.environ[name] for name in parent} == parent


def test_controller_error_wakes_caller_and_keeps_budget(tmp_path, monkeypatch):
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        initial = admission.record()

        def failure(*args, **kwargs):
            raise OSError("declared controller resource failure")

        monkeypatch.setattr(admission, "reserve_phase", failure)
        with pytest.raises(OSError, match="declared controller resource failure"):
            CloudExecutor(admission, max_workers=1, diagnostic_directory=tmp_path / "diagnostics",
                identity={}, poll_seconds=.05).run([_task(tmp_path, "producer")])
        assert admission.record() == initial


def test_dynamic_duplicate_and_generator_failure_keep_original_results(tmp_path):
    def accept(outcome):
        if outcome.task_id == "duplicate":
            return (_task(tmp_path, "duplicate"),)
        if outcome.task_id == "generator":
            def additions():
                yield _task(tmp_path, "never_admitted")
                raise ValueError("declared additions generator failure")
            return additions()
        return ()

    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = _admission(manager)
        run = CloudExecutor(admission, max_workers=1, diagnostic_directory=tmp_path / "diagnostics",
            identity={}, poll_seconds=.05).run([
                _task(tmp_path, "duplicate"), _task(tmp_path, "generator"), _task(tmp_path, "peer"),
            ], on_finished=accept)
    assert set(run.results) == {"duplicate", "generator", "peer"}
    assert run.results["duplicate"].failure["stage"] == "artifact_admission"
    assert run.results["generator"].failure["message"] == "declared additions generator failure"
    assert run.results["peer"].status == "completed" and not (tmp_path / "never_admitted").exists()
