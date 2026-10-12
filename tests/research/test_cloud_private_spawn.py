"""Linux spawn ABI, launch-window cleanup and birth authority witnesses."""
from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import research.cloud_execution as cloud

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux private exec environment")


def _mark(path):
    Path(path).write_text("executed", encoding="utf-8")


def _token_mark(path):
    Path(path).write_text(json.dumps({"python_token": os.environ["N2P3_CLOUD_TREE_TOKEN"],
                                    "exec_token": psutil.Process().environ()["N2P3_CLOUD_TREE_TOKEN"]}))


def _overlapping_executor_probe(arguments):
    ready, release = Path(arguments["ready"]), Path(arguments["release"])
    if arguments["mode"] == "wait_peer":
        Path(arguments["first_ready"]).write_text("first ready")
        while not ready.exists():
            time.sleep(.01)
        return "peer ready"
    ready.write_text("ready")
    while not release.exists():
        time.sleep(.01)
    child = os.fork()
    if child == 0:
        os.setsid()
        os.closerange(3, 2**20)
        Path(arguments["child"]).write_text(json.dumps(cloud._process_identity(psutil.Process())))
        time.sleep(30)
        os._exit(0)
    while not Path(arguments["child"]).exists():
        time.sleep(.01)
    os._exit(73)


def _subreaper_state():
    value = ctypes.c_int()
    assert ctypes.CDLL(None, use_errno=True).prctl(37, ctypes.byref(value), 0, 0, 0) == 0
    return value.value


def _journal_failure_probe(arguments):
    own = Path(arguments["own"])
    own.write_text(json.dumps(cloud._process_identity(psutil.Process())))
    if arguments["fatal"]:
        while not Path(arguments["peer"]).exists():
            time.sleep(.01)
        os._exit(73)
    while True:
        time.sleep(.01)


@pytest.mark.parametrize("minor,argument_count", [(11, 23), (12, 23), (14, 22)])
def test_missing_builtin_signature_uses_declared_cpython_abi(monkeypatch, minor, argument_count):
    import _posixsubprocess
    import subprocess

    calls = []

    class ForkExec:
        __text_signature__ = None

        def __call__(self, *arguments):
            calls.append(arguments)
            return 12345

    monkeypatch.setattr(_posixsubprocess, "fork_exec", ForkExec())
    monkeypatch.setattr(sys, "version_info", (3, minor, 0))
    monkeypatch.setattr(subprocess, "_USE_VFORK", True, raising=False)
    created = []
    assert cloud._spawnv_passfds_env(b"python", ["python"], (), {"TOKEN": "private"},
                                     on_created=created.append) == 12345
    assert len(calls[0]) == argument_count and calls[0][5] == [b"TOKEN=private"]
    assert created == [12345]
    for descriptor in calls[0][12:14]:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_unknown_abi_fails_before_allocating_errpipe(monkeypatch):
    import _posixsubprocess

    monkeypatch.setattr(_posixsubprocess, "fork_exec", SimpleNamespace(__text_signature__=None))
    monkeypatch.setattr(sys, "version_info", (3, 15, 0))
    descriptors = set(os.listdir("/proc/self/fd"))
    with pytest.raises(RuntimeError, match="known fork_exec ABI"):
        cloud._spawnv_passfds_env(b"python", ["python"], (), {}, on_created=lambda pid: None)
    assert set(os.listdir("/proc/self/fd")) == descriptors


def test_launch_retirement_refuses_a_different_birth_of_a_live_pid():
    process = psutil.Process()
    identity = cloud._process_identity(process)
    identity["create_time"] -= 1
    with pytest.raises(cloud.CloudLaunchUnverifiedError, match="another process birth"):
        cloud._retire_unregistered_process(process.pid, identity)
    assert process.is_running()


def test_pre_fork_failure_keeps_original_error(tmp_path, monkeypatch):
    from multiprocessing import resource_tracker

    resource_tracker.getfd()
    process = cloud._CloudSpawnProcess(target=_mark, args=(str(tmp_path / "executed"),))
    process._cloud_tree_token = "pre-fork-failure"

    def fail_pipe():
        raise OSError("original pipe creation failure")

    monkeypatch.setattr(os, "pipe", fail_pipe)
    with pytest.raises(OSError, match="original pipe creation failure"):
        process.start()
    assert not hasattr(process, "_cloud_spawn_pid")
    assert not (tmp_path / "executed").exists()


def test_errpipe_close_failure_records_and_retires_real_created_pid(tmp_path, monkeypatch):
    from multiprocessing import resource_tracker

    resource_tracker.getfd()
    original_close = os.close
    after_fork = False
    injected = False
    original_spawn = cloud._spawnv_passfds_env

    def launch(*args, on_created, **kwargs):
        def created(pid):
            nonlocal after_fork
            on_created(pid)
            after_fork = True
        return original_spawn(*args, on_created=created, **kwargs)

    def close(descriptor):
        nonlocal injected
        original_close(descriptor)
        if after_fork and not injected:
            injected = True
            raise OSError("original errpipe close failure")

    monkeypatch.setattr(cloud, "_spawnv_passfds_env", launch)
    monkeypatch.setattr(os, "close", close)
    process = cloud._CloudSpawnProcess(target=_mark, args=(str(tmp_path / "executed"),))
    process._cloud_tree_token = "errpipe-close-failure"
    with pytest.raises(OSError, match="original errpipe close failure"):
        process.start()
    assert injected and process._cloud_spawn_pid == process._cloud_launch_identity["pid"]
    assert process._cloud_launch_retirement["active_owned_processes_after_exit"] == 0
    assert not cloud._identity_live(process._cloud_launch_identity)
    assert not (tmp_path / "executed").exists()


@pytest.mark.parametrize("changed", ["actual_birth", "launch_birth"])
def test_bootstrap_birth_rejection_precedes_execute(monkeypatch, changed):
    process = psutil.Process()
    launch = cloud._process_identity(process)
    message = {"pid": process.pid, "create_time": process.create_time(), "tree_token": "token"}
    if changed == "actual_birth":
        message["create_time"] -= 1
    else:
        launch["create_time"] -= 1
    sent = []
    item = SimpleNamespace(process=SimpleNamespace(pid=process.pid), bootstrap=None,
                           launch_identity=launch, tree_token="token",
                           receiver=SimpleNamespace(send=sent.append))
    with pytest.raises(ValueError, match="birth identity"):
        cloud._admit_bootstrap(item, message)
    assert sent == []


def test_actual_private_spawn_keeps_parent_token_and_exec_environment(tmp_path, monkeypatch):
    # Two actual launches share a parent but never mutate its environment.
    monkeypatch.setenv("N2P3_CLOUD_TREE_TOKEN", "parent")
    def launch(index):
        process = cloud._CloudSpawnProcess(target=_token_mark, args=(str(tmp_path / str(index)),))
        process._cloud_tree_token = f"worker-{index}"
        process.start()
        assert os.environ["N2P3_CLOUD_TREE_TOKEN"] == "parent"
        return process

    with ThreadPoolExecutor(max_workers=2) as pool:
        processes = list(pool.map(launch, range(2)))
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
        assert not cloud._identity_live(process._cloud_launch_identity)
        process.close()
    assert [json.loads((tmp_path / str(index)).read_text()) for index in range(2)] == [
        {"python_token": f"worker-{index}", "exec_token": f"worker-{index}"} for index in range(2)]


def test_overlapping_executors_share_subreaper_until_last_tree_retires(tmp_path):
    from research.source_execution import HostMemoryAdmission

    initial_state = _subreaper_state()
    ready, release, child_path, first_ready = (
        tmp_path / name for name in ("ready", "release", "child.json", "first_ready"))
    after_first, child_live, owned = None, None, None
    try:
        with multiprocessing.get_context("spawn").Manager() as manager:
            admission = HostMemoryAdmission(manager, {
                "host_memory_budget_bytes": 256 * 2**20, "shared_staging_memory_bytes": 0,
                "source_worker_memory_bytes": 128 * 2**20, "target_worker_memory_bytes": 128 * 2**20,
                "source_workers": 2, "target_workers": 2, "gpu_slots": 2,
                "native_threads_per_worker": 1, "cpu_budget": 3, "coordinator_cpu_threads": 1})
            initial_budget = admission.record()

            def run(name, mode):
                task = cloud.CloudTask(name, f"{__name__}:_overlapping_executor_probe",
                    {"mode": mode, "ready": str(ready), "release": str(release), "child": str(child_path),
                     "first_ready": str(first_ready)},
                    str(tmp_path / name), cloud.PhaseResources("source", 1, 128 * 2**20, 1))
                return cloud.CloudExecutor(admission, max_workers=1,
                    diagnostic_directory=tmp_path / ("diagnostics_" + name), identity={}, poll_seconds=.02).run([task])

            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(run, "first", "wait_peer")
                deadline = time.monotonic() + 15
                while not first_ready.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                assert first_ready.exists()
                second = pool.submit(run, "second", "late_fork")
                try:
                    assert first.result(timeout=15).status == "completed"
                    after_first = _subreaper_state()
                finally:
                    release.write_text("release")
                outcome = second.result(timeout=15).results["second"]
                assert outcome.worker["exit_code"] == 73
                owned = outcome.worker["process_tree_retirement"]["owned_processes"]
            assert admission.record() == initial_budget
        child = json.loads(child_path.read_text())
        child_live = cloud._identity_live(child)
        assert after_first == 1
        assert not child_live and child in owned
        assert _subreaper_state() == initial_state
    finally:
        release.write_text("release")
        if child_path.exists():
            child = json.loads(child_path.read_text())
            if cloud._identity_live(child):
                process = psutil.Process(child["pid"])
                process.kill()
                process.wait(timeout=2)
        # Preserve the host's original flag even when running the old RED.
        ctypes.CDLL(None, use_errno=True).prctl(36, initial_state, 0, 0, 0)


def test_retirement_journal_failure_keeps_peer_cleanup_and_original_error(tmp_path, monkeypatch):
    from research.source_execution import HostMemoryAdmission

    original = cloud._CloudOwnershipJournal.record_item

    def fail_retirement_journal(self, item, *, status=None):
        if item.retirement is not None:
            raise PermissionError("original ownership journal write failure")
        return original(self, item, status=status)

    monkeypatch.setattr(cloud._CloudOwnershipJournal, "record_item", fail_retirement_journal)
    initial_subreaper = _subreaper_state()
    identities = [tmp_path / "fatal.json", tmp_path / "peer.json"]
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, {
            "host_memory_budget_bytes": 256 * 2**20, "shared_staging_memory_bytes": 0,
            "source_worker_memory_bytes": 128 * 2**20, "target_worker_memory_bytes": 128 * 2**20,
            "source_workers": 2, "target_workers": 2, "gpu_slots": 2,
            "native_threads_per_worker": 1, "cpu_budget": 3, "coordinator_cpu_threads": 1})
        initial_budget = admission.record()
        tasks = [cloud.CloudTask(name, f"{__name__}:_journal_failure_probe",
            {"own": str(identities[index]), "peer": str(identities[1]), "fatal": index == 0},
            str(tmp_path / name), cloud.PhaseResources("source", 1, 128 * 2**20, 1))
            for index, name in enumerate(("fatal", "peer"))]
        with pytest.raises(PermissionError, match="original ownership journal write failure") as captured:
            cloud.CloudExecutor(admission, max_workers=2, diagnostic_directory=tmp_path / "diagnostics",
                                identity={}, poll_seconds=.02).run(tasks)
        assert admission.record() == initial_budget
        assert captured.value.__notes__
    assert not any(cloud._identity_live(json.loads(path.read_text())) for path in identities)
    assert _subreaper_state() == initial_subreaper
    receipt = next((tmp_path / "diagnostics").rglob(".cloud_controller.*.json"))
    assert json.loads(receipt.read_text())["status"] == "unverified"
