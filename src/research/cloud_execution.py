"""One resource-admitted ready queue for cache-group scientific adapters.

Adapters keep their recipes, RNG and result schemas. Arguments carry input
identities and lightweight configuration; results carry artifact paths/facts,
never a live model or an EEG matrix. Each cache group owns one fresh process,
so a returned result does not return its phase reservation until that process
has actually exited. Persistent input/artifact leases belong to the supplied
admission owner and are not released with a process reservation.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import io
import json
import multiprocessing
import os
import pickle
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Pipe, wait
from pathlib import Path
from queue import Empty, Queue

import psutil

from research.execution import atomic_write_json
from research.source_execution import process_file_lock
from research.worker_diagnostics import (
    PoolFailureDiagnostics,
    diagnostic_task,
    initialize_worker_diagnostics,
    system_snapshot,
    worker_diagnostic_phase,
)

CLOUD_OWNERSHIP_SCHEMA = "n2p3_cloud_controller_ownership/1"
_SUBREAPER_LOCK = threading.Lock()
_SUBREAPER_USERS = 0
_SUBREAPER_BASE_STATE = None


class CloudLaunchUnverifiedError(RuntimeError):
    """A child was created but its launch handoff could not be settled."""


if sys.platform == "linux":
    from multiprocessing.context import SpawnProcess as _SpawnProcess
    from multiprocessing.popen_spawn_posix import Popen as _SpawnPopen

    def _spawnv_passfds_env(path, args, passfds, environment, *, on_created):
        import _posixsubprocess
        import subprocess

        abi = _fork_exec_abi()
        passfds = tuple(sorted(map(int, passfds)))
        env_list = [os.fsencode(key) + b"=" + os.fsencode(value)
                    for key, value in environment.items()]
        errpipe_read, errpipe_write = os.pipe()
        try:
            fork_args = [args, [path], True, passfds, None, env_list,
                         -1, -1, -1, -1, -1, -1, errpipe_read, errpipe_write,
                         False, False, -1, None, None, None, -1, None]
            if abi["uses_allow_vfork"]:
                fork_args.append(subprocess._USE_VFORK)
            pid = _posixsubprocess.fork_exec(*fork_args)
            on_created(pid)
            return pid
        finally:
            try:
                os.close(errpipe_read)
            finally:
                os.close(errpipe_write)

    def _retire_unregistered_process(pid, identity=None):
        if identity is None or identity["pid"] != pid:
            raise CloudLaunchUnverifiedError("launch retirement requires the original PID/birth identity")
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return {"backend": "launch_window_process", "owned_processes": [identity],
                    "active_owned_processes_after_exit": 0}
        if process.create_time() != identity["create_time"]:
            raise CloudLaunchUnverifiedError("launch PID now belongs to another process birth; retirement refused")
        if _owned_process_alive(process):
            process.terminate()
        try:
            process.wait(timeout=.5)
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=.5)
        return {"backend": "launch_window_process", "owned_processes": [identity],
                "active_owned_processes_after_exit": int(_owned_process_alive(process))}

    class _CloudSpawnPopen(_SpawnPopen):
        """Spawn a worker with a private exec environment."""

        def _launch(self, process_obj):
            from multiprocessing import resource_tracker, spawn, util
            from multiprocessing.context import reduction, set_spawning_popen

            self.pid = None
            tracker_fd = resource_tracker.getfd()
            self._fds.append(tracker_fd)
            prep_data = spawn.get_preparation_data(process_obj._name)
            fp = io.BytesIO()
            set_spawning_popen(self)
            try:
                reduction.dump(prep_data, fp)
                reduction.dump(process_obj, fp)
            finally:
                set_spawning_popen(None)

            parent_r = child_w = child_r = parent_w = None

            def remember_created(pid):
                # Record the handle before errpipe cleanup can replace the
                # successful fork_exec return with an exception.
                self.pid = pid
                process_obj._cloud_spawn_pid = pid
                process_obj._cloud_launch_identity = _process_identity(psutil.Process(pid))

            try:
                parent_r, child_w = os.pipe()
                child_r, parent_w = os.pipe()
                command = spawn.get_command_line(tracker_fd=tracker_fd, pipe_handle=child_r)
                self._fds.extend([child_r, child_w])
                environment = os.environ.copy()
                token = getattr(process_obj, "_cloud_tree_token", None)
                if token is not None:
                    environment["N2P3_CLOUD_TREE_TOKEN"] = token
                self.pid = _spawnv_passfds_env(spawn.get_executable(), command, self._fds, environment,
                                              on_created=remember_created)
                # Match the stdlib spawn order: once fork_exec has returned,
                # the parent must close both child-side descriptors before
                # writing the preparation payload. Otherwise a dead child can
                # keep the pipe's reader reference alive and block a large
                # payload indefinitely.
                os.close(child_r)
                child_r = None
                os.close(child_w)
                child_w = None
                self.sentinel = parent_r
                with open(parent_w, "wb", closefd=False) as stream:
                    stream.write(fp.getbuffer())
            except BaseException as error:
                if self.pid is not None:
                    try:
                        process_obj._cloud_launch_retirement = _retire_unregistered_process(
                            self.pid, getattr(process_obj, "_cloud_launch_identity", None))
                    except BaseException as cleanup_error:
                        raise CloudLaunchUnverifiedError(
                            f"spawned worker {self.pid} could not be retired after launch handoff failure"
                        ) from cleanup_error
                    if process_obj._cloud_launch_retirement["active_owned_processes_after_exit"]:
                        raise CloudLaunchUnverifiedError(
                            f"spawned worker {self.pid} remained live after launch handoff failure"
                        ) from error
                raise
            finally:
                fds_to_close = [fd for fd in (parent_r, parent_w) if fd is not None]
                self.finalizer = util.Finalize(self, util.close_fds, fds_to_close)
                for fd in (child_r, child_w):
                    if fd is not None:
                        os.close(fd)

    class _CloudSpawnProcess(_SpawnProcess):
        @staticmethod
        def _Popen(process_obj):  # noqa: N802
            return _CloudSpawnPopen(process_obj)
else:
    _CloudSpawnProcess = None


def _fork_exec_abi():
    if sys.platform != "linux":
        return None
    import _posixsubprocess

    signature = getattr(_posixsubprocess.fork_exec, "__text_signature__", None)
    if isinstance(signature, str) and signature.strip():
        return {"text_signature": signature, "uses_allow_vfork": "allow_vfork" in signature,
                "authority": "builtin_text_signature"}
    minor = sys.version_info[:2]
    if sys.implementation.name != "cpython" or not (3, 11) <= minor <= (3, 14):
        raise RuntimeError("private Linux spawn requires a known fork_exec ABI")
    return {"text_signature": None, "uses_allow_vfork": minor < (3, 14),
            "authority": "cpython_minor", "python_minor": list(minor)}


class CloudOwnershipError(RuntimeError):
    """A previous cloud controller has no verified, settled ownership receipt."""


def _process_identity(process):
    """Read the PID/birth pair and process group without inferring descendants."""
    return {"pid": int(process.pid), "create_time": float(process.create_time()),
            "process_group": int(os.getpgid(process.pid)) if os.name != "nt" else None}


def _identity_live(identity):
    """Return liveness for the exact PID/birth pair; access errors stay blocking."""
    try:
        process = psutil.Process(int(identity["pid"]))
        if process.create_time() != float(identity["create_time"]):
            return False
        return _owned_process_alive(process)
    except psutil.NoSuchProcess:
        return False


def _ownership_files(directory):
    directory = Path(directory).resolve()
    return sorted(directory.glob(".cloud_controller.*.json"))


def _ownership_legacy_files(directory):
    directory = Path(directory).resolve()
    return [*directory.glob("*.state.json"), *directory.glob("first_pool_failure.json"),
            *directory.glob("pool_failure_observations.json")]


def _validate_owned_identity(identity):
    if not isinstance(identity, dict) or type(identity.get("pid")) is not int:
        raise CloudOwnershipError("cloud ownership receipt has no integer PID identity")
    created = identity.get("create_time")
    if not isinstance(created, (int, float)) or not created >= 0:
        raise CloudOwnershipError("cloud ownership receipt has no valid PID birth time")
    if os.name != "nt" and type(identity.get("process_group")) is not int:
        raise CloudOwnershipError("cloud ownership receipt has no process-group containment identity")


def _validate_retired_identities(identities, path):
    for identity in identities:
        _validate_owned_identity(identity)
        if _identity_live(identity):
            raise CloudOwnershipError(f"prior owned process remains live: {path.name}")


def _validate_settled_ownership(record, path):
    if record.get("schema") != CLOUD_OWNERSHIP_SCHEMA:
        raise CloudOwnershipError(f"unsupported cloud ownership receipt: {path.name}")
    if record.get("status") != "settled":
        raise CloudOwnershipError(f"cloud ownership receipt is unverified: {path.name}")
    identity = record.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("pool_instance_id"), str) \
            or not identity["pool_instance_id"] or identity["pool_instance_id"] != path.parent.name:
        raise CloudOwnershipError(f"cloud ownership receipt lacks its pool instance identity: {path.parent.name}/{path.name}")
    controller = record.get("controller")
    _validate_owned_identity(controller)
    tasks = record.get("tasks")
    if not isinstance(tasks, list):
        raise CloudOwnershipError(f"cloud ownership receipt has no task ledger: {path.name}")
    for task in tasks:
        if not isinstance(task, dict) or not task.get("task_id"):
            raise CloudOwnershipError(f"cloud ownership receipt has an invalid task row: {path.name}")
        state = task.get("status")
        if state == "spawn_failed":
            retirement = task.get("retirement", {})
            if retirement.get("active_owned_processes_after_exit") != 0:
                raise CloudOwnershipError(f"spawn failure lacks settled retirement: {path.name}")
            launch = task.get("launch")
            owned = retirement.get("owned_processes", [])
            if launch is not None:
                if launch not in owned:
                    raise CloudOwnershipError(f"spawn failure omitted its created process: {path.name}")
            _validate_retired_identities(owned, path)
            continue
        token = task.get("tree_token")
        if not isinstance(token, str) or not token:
            raise CloudOwnershipError(f"task lacks its inherited ownership token: {path.name}")
        bootstrap = task.get("bootstrap")
        if not isinstance(bootstrap, dict):
            raise CloudOwnershipError(f"task lacks an admitted bootstrap receipt: {path.name}")
        if bootstrap.get("tree_token") != token:
            raise CloudOwnershipError(f"task bootstrap token changed: {path.name}")
        _validate_owned_identity({"pid": bootstrap.get("pid"),
                                  "create_time": bootstrap.get("create_time"),
                                  "process_group": bootstrap.get("process_group")})
        containment = task.get("containment")
        if not isinstance(containment, dict) or containment.get("status") != "admitted":
            raise CloudOwnershipError(f"task lacks admitted containment evidence: {path.name}")
        retirement = task.get("retirement")
        if not isinstance(retirement, dict) or retirement.get("active_owned_processes_after_exit") != 0:
            raise CloudOwnershipError(f"task lacks settled retirement evidence: {path.name}")
        owned = task.get("owned_processes")
        if not isinstance(owned, list) or not owned:
            raise CloudOwnershipError(f"task lacks actual owned PID/birth records: {path.name}")
        _validate_retired_identities(owned, path)


def assert_no_prior_owned_live(directory):
    """Reject live or unverified prior CloudExecutor trees before a new run.

    The controller sole-owner writes ``.cloud_controller.<run-id>.json`` while
    it owns workers.  This gate consumes only its durable bootstrap, token,
    containment and PID/birth evidence; discovery and retirement remain solely
    in :class:`CloudExecutor`.  A missing/partial receipt is a HOLD even when
    the recorded PID has already disappeared, because the historical tree was
    never proved retired.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    pool_directories = {directory}
    for pattern in (".cloud_controller.*.json", "*.state.json", "first_pool_failure.json",
                    "pool_failure_observations.json"):
        pool_directories.update(path.parent for path in directory.rglob(pattern))
    for pool_directory in sorted(pool_directories):
        paths = _ownership_files(pool_directory)
        legacy = _ownership_legacy_files(pool_directory)
        if legacy and not paths:
            raise CloudOwnershipError(
                f"historical cloud diagnostics have no ownership receipt: {pool_directory}")
        for path in paths:
            try:
                record = json.loads(path.read_bytes())
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise CloudOwnershipError(f"cloud ownership receipt is unreadable: {path.name}") from error
            _validate_settled_ownership(record, path)
            for state_path in pool_directory.glob("*.state.json"):
                state = json.loads(state_path.read_bytes())
                if state.get("identity", {}).get("pool_instance_id") != record["identity"]["pool_instance_id"]:
                    raise CloudOwnershipError(f"worker state differs from its ownership pool: {state_path}")
    return None


@dataclass(frozen=True)
class PhaseResources:
    phase: str
    cpu_threads: int
    memory_bytes: int
    native_threads: int
    device: str = "cpu"
    interop_threads: int | None = None
    defer_gpu: bool = False


def target_phase_resources(resources, device):
    """Honor explicit target native/inter-op claims and activate CUDA lazily."""
    native = resources["native_threads_per_worker"]
    target_native = resources.get("target_native_threads_per_worker")
    if target_native is not None:
        return PhaseResources("calibration", target_native + 1, resources["target_worker_memory_bytes"],
                              target_native, str(device), interop_threads=1,
                              defer_gpu=str(device).startswith("cuda"))
    return PhaseResources("calibration", native, resources["target_worker_memory_bytes"], native, str(device))


_CURRENT_WORKER_PHASES = None


class _TargetWorkerPhases:
    """GPU admission precedes CUDA initialization and lasts through tree exit."""

    def __init__(self, admission, task):
        self.admission, self.task = admission, task
        self.gpu_active = not task.resources.defer_gpu

    def enter_compute(self, device):
        if not self.task.resources.defer_gpu:
            return
        import torch

        if (torch.get_num_threads() != self.task.resources.native_threads
                or torch.get_num_interop_threads() != self.task.resources.interop_threads):
            raise RuntimeError("target compute differs from its actual native/inter-op claim")
        if str(device).startswith("cuda") and not self.gpu_active:
            if str(device) != self.task.resources.device:
                raise ValueError("target compute device differs from its declared phase device")
            if torch.cuda.is_initialized():
                raise RuntimeError("target CUDA initialized before acquiring its GPU claim")
            worker_diagnostic_phase("waiting_target_gpu_resources", native_threads=torch.get_num_threads(),
                                    interop_threads=torch.get_num_interop_threads())
            row = None
            while row is None:
                row = self.admission._ledger.try_activate_target_gpu(self.task.task_id)
                if row is None:
                    time.sleep(.05)
            self.gpu_active = True
            worker_diagnostic_phase("target_gpu_resources_admitted", actual_phase_reservation=row,
                                    native_threads=torch.get_num_threads(),
                                    interop_threads=torch.get_num_interop_threads())
        worker_diagnostic_phase("target_compute", compute_device=str(device),
                                retained_gpu_claim=int(self.gpu_active))

    def verify_cuda_admission(self):
        torch = sys.modules.get("torch")
        if self.task.resources.defer_gpu and torch is not None and torch.cuda.is_initialized() and not self.gpu_active:
            raise RuntimeError("target adapter initialized CUDA outside an admitted compute phase")


@contextmanager
def worker_compute_phase(device):
    """Caller marks actual fit/predict work; inline scientific calls stay valid."""
    if _CURRENT_WORKER_PHASES is not None:
        _CURRENT_WORKER_PHASES.enter_compute(device)
    yield
    # No downgrade: model state, CUDA context and allocator may remain resident.


@dataclass(frozen=True)
class CloudTask:
    task_id: str
    adapter: str
    arguments: object
    output: str
    resources: PhaseResources
    depends_on: tuple[str, ...] = ()


@dataclass
class CloudResult:
    task_id: str
    status: str
    value: object = None
    failure: dict | None = None
    worker: dict | None = None
    supplementary_failures: list = field(default_factory=list)


@dataclass
class CloudRun:
    results: dict
    diagnostics: dict
    observation_failures: list

    @property
    def status(self):
        return "completed" if all(result.status == "completed" for result in self.results.values()) else "failed"


def _failure(error, stage):
    return {"type": type(error).__name__, "message": str(error), "stage": stage}


@contextmanager
def _native_environment(threads, task_id, tree_token):
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "TORCHINDUCTOR_COMPILE_THREADS", "N2P3_CLOUD_TASK_ID", "N2P3_CLOUD_TREE_TOKEN")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = (tree_token if name == "N2P3_CLOUD_TREE_TOKEN" else
                               task_id if name == "N2P3_CLOUD_TASK_ID" else
                               "1" if name == "TORCHINDUCTOR_COMPILE_THREADS" else str(threads))
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _adapter(location):
    module, name = location.split(":", 1)
    function = importlib.import_module(module)
    for item in name.split("."):
        function = getattr(function, item)
    return function


def _device_snapshot(device):
    try:
        torch = sys.modules.get("torch")
        if torch is None or not str(device).startswith("cuda") or not torch.cuda.is_initialized():
            return {"device": device, "status": "not_initialized"}
        from train.runtime import GpuPerformanceScheduler

        return {"status": "read", **GpuPerformanceScheduler(torch.device(device), precision="fp32").memory_snapshot().record()}
    except (OSError, RuntimeError, ValueError) as error:
        return {"device": device, "status": "unavailable", "error": _failure(error, "device_sensor")}


def _worker_entry(encoded, sender, directory, identity, launch_environment):
    # Spawn may already have imported its main module. The actual adapter's
    # numeric pools are configured again before compute in the science entry.
    with _native_environment(*launch_environment):
        _worker_science_entry(encoded, sender, directory, identity)


def _worker_science_entry(encoded, sender, directory, identity):
    global _CURRENT_WORKER_PHASES

    if os.name != "nt":
        os.setsid()
    task, input_admission, admission = pickle.loads(encoded)
    from research.source_execution import install_input_admission

    install_input_admission(input_admission)
    _CURRENT_WORKER_PHASES = _TargetWorkerPhases(admission, task)
    initialize_worker_diagnostics(directory, {**identity, "cloud_task_id": task.task_id})
    sender.send({"event": "ready", "pid": os.getpid(), "create_time": psutil.Process().create_time(),
                 "process_group": os.getpgrp() if os.name != "nt" else None,
                 "tree_token": os.environ["N2P3_CLOUD_TREE_TOKEN"]})
    if sender.recv() != "execute":
        return
    result = CloudResult(task.task_id, "completed")
    try:
        with diagnostic_task(task.output, {"task_id": task.task_id, "phase": task.resources.phase,
                "adapter": task.adapter, "resources": asdict(task.resources)}):
            if task.resources.interop_threads is not None:
                from train.runtime import configure_spawned_worker_threads

                # Inter-op is process-global and immutable after work begins.
                # Establish the explicit policy before importing the adapter.
                configure_spawned_worker_threads(task.resources.native_threads, interop_threads=task.resources.interop_threads)
            worker_diagnostic_phase("adapter_import")
            function = _adapter(task.adapter)
            if "torch" in sys.modules:
                from train.runtime import configure_spawned_worker_threads, cpu_thread_budget

                configure_spawned_worker_threads(task.resources.native_threads, interop_threads=task.resources.interop_threads)
                with cpu_thread_budget(task.resources.native_threads):
                    _CURRENT_WORKER_PHASES.verify_cuda_admission()
                    if (task.resources.defer_gpu and
                            sys.modules["torch"].get_num_threads() != task.resources.native_threads):
                        raise RuntimeError("worker cannot establish its declared target native thread budget")
                    worker_diagnostic_phase("adapter_execution")
                    result.value = function(task.arguments)
            else:
                worker_diagnostic_phase("adapter_execution")
                result.value = function(task.arguments)
            _CURRENT_WORKER_PHASES.verify_cuda_admission()
    except BaseException as error:
        result.status, result.failure = "failed", _failure(error, "adapter_execution")
    before = {"host": system_snapshot(), "device": _device_snapshot(task.resources.device)}
    worker_diagnostic_phase("cache_group_cleanup_started")
    try:
        if "torch" in sys.modules:
            from research.source_execution import release_source_worker

            release_source_worker(task.resources.device)
        gc.collect()
    except Exception as error:
        cleanup_failure = _failure(error, "worker_cleanup")
        if result.failure is None:
            result.status, result.failure = "failed", cleanup_failure
        else:
            result.supplementary_failures.append(cleanup_failure)
    after = {"host": system_snapshot(), "device": _device_snapshot(task.resources.device)}
    result.worker = {"pid": os.getpid(), "create_time": after["host"].get("create_time"),
                     "cleanup_before": before, "cleanup_after": after,
                     "target_gpu_claim_activated": task.resources.defer_gpu and _CURRENT_WORKER_PHASES.gpu_active}
    worker_diagnostic_phase("cache_group_cleanup_returned", status=result.status)
    try:
        sender.send(result)
    finally:
        sender.close()


def _make_process(context, task, sender, directory, identity, input_admission, admission, tree_token):
    # Encoding before creating a process makes a bad lightweight argument fail
    # before an unregistered child can be created by spawn's reduction step.
    encoded = pickle.dumps((task, input_admission, admission), protocol=pickle.HIGHEST_PROTOCOL)
    process_class = (_CloudSpawnProcess if _CloudSpawnProcess is not None
                     and context.get_start_method() == "spawn" else context.Process)
    process = process_class(target=_worker_entry, args=(encoded, sender, str(directory), identity,
        (task.resources.native_threads, task.task_id, tree_token)))
    if process_class is _CloudSpawnProcess:
        process._cloud_tree_token = tree_token
    return process


@dataclass
class _Running:
    task: CloudTask
    reservation: dict
    process: object
    receiver: object
    message: CloudResult | None = None
    receiving: bool = True
    containment: object = None
    bootstrap: dict | None = None
    retirement: dict | None = None
    tree_token: str | None = None
    owned_processes: dict = field(default_factory=dict)
    infrastructure_children: set = field(default_factory=set)
    launch_identity: dict | None = None


class _CloudOwnershipJournal:
    """Durable controller-owned process facts consumed by the next gate."""

    def __init__(self, directory, identity):
        self.directory = Path(directory).resolve()
        self.path = self.directory / f".cloud_controller.{uuid.uuid4().hex}.json"
        process = psutil.Process()
        self.record = {"schema": CLOUD_OWNERSHIP_SCHEMA, "status": "running",
                       "controller": _process_identity(process), "identity": dict(identity),
                       "started_unix": time.time(), "tasks": [], "run_status": None}

    def _write(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        # The caller normally already holds its programme lease.  This narrow
        # journal lock also makes controller updates atomic when a recovery
        # inspector reads the directory concurrently.
        with process_file_lock(self.directory / ".cloud_controller.lock"):
            atomic_write_json(self.path, self.record)

    def start(self):
        self._write()

    def _task(self, task_id):
        return next((row for row in self.record["tasks"] if row["task_id"] == task_id), None)

    def register_launch(self, task, tree_token):
        if self._task(task.task_id) is not None:
            raise CloudOwnershipError(f"duplicate cloud ownership task: {task.task_id}")
        self.record["tasks"].append({"task_id": task.task_id, "status": "launching",
            "tree_token": tree_token, "bootstrap": None,
            "containment": {"status": "pending", "backend": "windows_job" if os.name == "nt"
                            else "linux_subreaper_and_owned_process_groups"},
            "owned_processes": [], "retirement": None, "launch": None})
        self._write()

    def launched(self, task_id, process):
        row = self._task(task_id)
        if row is None:
            raise CloudOwnershipError(f"cloud process was launched without a journal row: {task_id}")
        launch = _process_identity(psutil.Process(process.pid))
        original = getattr(process, "_cloud_launch_identity", None)
        if original is not None and (original["pid"], original["create_time"]) != (
                launch["pid"], launch["create_time"]):
            raise CloudLaunchUnverifiedError("launch birth changed before controller registration")
        row["status"] = "spawned"
        row["launch"] = launch
        self._write()

    def record_item(self, item, *, status=None):
        row = self._task(item.task.task_id)
        if row is None:
            raise CloudOwnershipError(f"cloud task has no ownership journal row: {item.task.task_id}")
        row["status"] = status or ("retired" if item.retirement is not None else "running")
        row["tree_token"] = item.tree_token
        row["bootstrap"] = None if item.bootstrap is None else dict(item.bootstrap)
        if item.bootstrap is not None:
            launch = row.get("launch")
            if (not isinstance(launch, dict)
                    or launch.get("pid") != item.bootstrap.get("pid")
                    or launch.get("create_time") != item.bootstrap.get("create_time")):
                raise CloudOwnershipError(f"launch and bootstrap birth identities differ: {item.task.task_id}")
            row["containment"] = {"status": "admitted",
                "backend": "windows_job" if os.name == "nt" else "linux_subreaper_and_owned_process_groups",
                "process_group": item.bootstrap.get("process_group")}
        row["owned_processes"] = sorted((dict(value) for value in item.owned_processes.values()),
                                         key=lambda value: (value["pid"], value["create_time"]))
        row["retirement"] = None if item.retirement is None else dict(item.retirement)
        self._write()

    def spawn_failed(self, task_id, error, *, launch=None, retirement=None):
        row = self._task(task_id)
        if row is None:
            raise CloudOwnershipError(f"cloud spawn failure has no journal row: {task_id}")
        settled = retirement is not None and retirement.get("active_owned_processes_after_exit") == 0
        row.update(status="spawn_failed" if settled else "unverified",
                   failure={"type": type(error).__name__, "message": str(error)},
                   launch=launch,
                   containment={"status": "none" if settled and launch is None else
                                "retired" if settled else "unverified",
                                "backend": "process_not_started" if settled and launch is None else "launch_window"},
                   retirement=retirement or {"active_owned_processes_after_exit": None})
        self._write()

    def finish(self, *, run_status, settled):
        self.record.update(status="settled" if settled else "unverified", run_status=run_status,
                           finished_unix=time.time())
        self._write()


def _admit_bootstrap(item, message):
    """Science waits for its own containment; a rejection belongs to this task."""
    if item.bootstrap is not None or message.get("pid") != item.process.pid:
        raise ValueError("bootstrap PID differs from the owned process handle or repeated its admission")
    try:
        actual_create_time = psutil.Process(item.process.pid).create_time()
    except psutil.NoSuchProcess as error:
        raise ValueError("bootstrap process disappeared before birth identity admission") from error
    if actual_create_time != message.get("create_time"):
        raise ValueError("bootstrap birth identity differs from the actual process handle")
    if (item.launch_identity is None or item.launch_identity["pid"] != message["pid"]
            or item.launch_identity["create_time"] != actual_create_time):
        raise ValueError("bootstrap birth identity differs from the admitted launch ledger")
    if message.get("tree_token") != item.tree_token:
        raise ValueError("bootstrap process-tree ownership token differs from the owned task")
    if sys.platform == "linux":
        try:
            launch_token = psutil.Process(item.process.pid).environ().get("N2P3_CLOUD_TREE_TOKEN")
        except (psutil.AccessDenied, psutil.NoSuchProcess) as error:
            raise ValueError("cannot verify the Linux launch-time process-tree ownership token") from error
        if launch_token != item.tree_token:
            raise ValueError("Linux worker did not retain its launch-time process-tree ownership token")
    if os.name == "nt":
        item.containment = _WindowsJob(item.process)
    elif message.get("process_group") != item.process.pid:
        raise ValueError("worker science must start inside its own process session")
    item.bootstrap = message
    item.owned_processes[item.process.pid] = {
        "pid": item.process.pid,
        "create_time": message["create_time"],
        "process_group": message.get("process_group"),
    }
    item.receiver.send("execute")


class _WindowsJob:
    """Keep the entire nested process tree inside one OS-owned task lifetime."""

    def __init__(self, process):
        import ctypes
        from ctypes import wintypes

        size_t = ctypes.c_size_t

        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("min_ws", size_t), ("max_ws", size_t),
                ("active_limit", wintypes.DWORD), ("affinity", size_t),
                ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", Counters), ("process_memory", size_t),
                ("job_memory", size_t), ("peak_process_memory", size_t), ("peak_job_memory", size_t)]

        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_int64) for name in ("user", "kernel", "period_user", "period_kernel")]
            _fields_ += [(name, wintypes.DWORD) for name in ("faults", "processes", "active", "terminated")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes, kernel.CreateJobObjectW.restype = [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel, self.ctypes, self.accounting = kernel, ctypes, Accounting
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel.AssignProcessToJobObject(self.handle, process.sentinel):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            kernel.CloseHandle(self.handle)
            raise

    def retire(self):
        if self.handle is None:
            return self.retirement
        if not self.kernel.TerminateJobObject(self.handle, 0x10000):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        while True:
            info = self.accounting()
            if not self.kernel.QueryInformationJobObject(self.handle, 1, self.ctypes.byref(info), self.ctypes.sizeof(info), None):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            if info.active == 0:
                break
            time.sleep(.01)
        self.kernel.CloseHandle(self.handle)
        self.handle = None
        self.retirement = {"backend": "windows_job", "active_owned_processes_after_exit": 0}
        return self.retirement


def _enable_task_subreaper():
    global _SUBREAPER_USERS, _SUBREAPER_BASE_STATE

    if sys.platform != "linux":
        return None
    with _SUBREAPER_LOCK:
        if not _SUBREAPER_USERS:
            libc = ctypes.CDLL(None, use_errno=True)
            previous = ctypes.c_int()
            if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) or libc.prctl(36, 1, 0, 0, 0):
                raise OSError(ctypes.get_errno(), "cannot establish cloud descendant ownership")
            _SUBREAPER_BASE_STATE = previous.value
        _SUBREAPER_USERS += 1
        return _SUBREAPER_BASE_STATE


def _restore_task_subreaper(previous):
    global _SUBREAPER_USERS, _SUBREAPER_BASE_STATE

    if previous is not None:
        with _SUBREAPER_LOCK:
            _SUBREAPER_USERS -= 1
            if not _SUBREAPER_USERS:
                libc = ctypes.CDLL(None, use_errno=True)
                if libc.prctl(36, _SUBREAPER_BASE_STATE, 0, 0, 0):
                    raise OSError(ctypes.get_errno(), "cannot restore cloud descendant ownership")
                _SUBREAPER_BASE_STATE = None


def _remember_task_tree(item):
    def remember(process):
        try:
            item.owned_processes[process.pid] = {"pid": process.pid,
                "create_time": process.create_time(), "process_group": os.getpgid(process.pid)}
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass

    try:
        leader = psutil.Process(item.bootstrap["pid"])
        if leader.create_time() == item.bootstrap["create_time"]:
            remember(leader)
            for process in leader.children(recursive=True):
                remember(process)
    except psutil.NoSuchProcess:
        pass
    # Native leader exit adopts even setsid descendants into the subreaper.
    # Only the token written at the OS exec boundary proves that an adopted
    # process belongs to this launch.  Do not infer ownership from birth time:
    # a peer can fork after the retirement snapshot and have a newer PID.
    for process in psutil.Process().children():
        try:
            if ((process.pid, process.create_time()) in item.infrastructure_children
                    or not _owned_process_alive(process)
                    ):
                continue
            try:
                token = process.environ().get("N2P3_CLOUD_TREE_TOKEN")
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if token != item.tree_token:
                continue
            remember(process)
            for descendant in process.children(recursive=True):
                remember(descendant)
        except psutil.NoSuchProcess:
            pass


def _retire_posix_task_tree(item):
    groups = []
    while True:
        _remember_task_tree(item)
        live = []
        for identity in item.owned_processes.values():
            identity_key = (identity["pid"], identity["create_time"])
            if identity_key in item.infrastructure_children:
                continue
            try:
                process = psutil.Process(identity["pid"])
                if process.create_time() == identity["create_time"] and _owned_process_alive(process):
                    live.append(process)
            except psutil.NoSuchProcess:
                pass
        if not live:
            break
        process_groups = set()
        for process in live:
            try:
                process_groups.add(os.getpgid(process.pid))
            except ProcessLookupError:
                pass
        if os.getpgrp() in process_groups:
            raise RuntimeError("owned science descendant joined the coordinator process group")
        groups.extend(_retire_process_group(group) for group in sorted(process_groups))
    # Reap only this task's adopted children; never waitpid(-1) and consume an
    # independent multiprocessing handle's real OS exit.
    for identity in item.owned_processes.values():
        try:
            process = psutil.Process(identity["pid"])
            if process.create_time() == identity["create_time"]:
                process.wait(timeout=.5)
        except psutil.NoSuchProcess:
            pass
    return {"backend": "linux_subreaper_and_owned_process_groups",
        "process_group": item.bootstrap["process_group"], "groups": groups,
        "owned_processes": list(item.owned_processes.values()),
        "active_owned_processes_after_exit": 0}


def _retire_process_group(group):
    members = []
    for process in psutil.process_iter(["pid", "create_time", "status"]):
        try:
            if os.getpgid(process.pid) == group:
                members.append(process)
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    for process in members:
        if _owned_process_alive(process):
            process.terminate()
    _, alive = psutil.wait_procs(members, timeout=.5)
    for process in alive:
        if _owned_process_alive(process):
            process.kill()
    while any(_owned_process_alive(process) for process in alive):
        time.sleep(.01)
    return {"backend": "posix_session", "process_group": group,
            "retired_descendant_pids": [process.pid for process in members],
            "active_owned_processes_after_exit": 0, "orphan_zombies_may_await_init_reaping": True}


def _retire_tree(item):
    if item.retirement is not None:
        return item.retirement
    if item.containment is not None:
        item.retirement = item.containment.retire()
        return item.retirement
    if item.bootstrap is None:
        return {"backend": "bootstrap_not_admitted", "science_execution_started": False}
    item.retirement = (_retire_posix_task_tree(item) if sys.platform == "linux" and item.bootstrap.get("tree_token")
                       else _retire_process_group(item.bootstrap["process_group"]))
    return item.retirement


def _owned_process_alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class CloudWorkerExitError(RuntimeError):
    pass


class CloudExecutor:
    """Own process lifetime/ready work, while one existing owner admits bytes.

    ``on_finished`` admits the scientific output and can return newly unlocked
    tasks. A rejection changes a successful process result into a failed node;
    it never overrides an existing primary failure. ``on_observation`` is a
    non-scientific journal callback; its failures are retained separately.
    An I/O controller owns process handles, retirement, and ready dispatch.
    Scientific and observation callbacks stay serial on the calling thread.
    Only acknowledged scientific admission unlocks dependent work.
    """

    def __init__(self, admission, *, max_workers, diagnostic_directory, identity, poll_seconds=5.):
        if type(max_workers) is not int or max_workers < 1 or not 0 < poll_seconds < float("inf"):
            raise ValueError("cloud executor requires positive process capacity and polling interval")
        self.admission, self.max_workers = admission, max_workers
        self.poll_seconds = poll_seconds
        self.context = multiprocessing.get_context("spawn")
        self.diagnostic_root, self.identity = Path(diagnostic_directory), identity
        self._processes = {}

    @staticmethod
    def _add_tasks(requested, tasks):
        additions = tuple(tasks)
        seen = set(requested)
        for task in additions:
            if not task.task_id or task.task_id in seen:
                raise ValueError(f"duplicate or empty cloud task id: {task.task_id}")
            seen.add(task.task_id)
            if ":" not in task.adapter:
                raise ValueError("scientific adapters must use a module:function location")
            if (type(task.resources.native_threads) is not int or
                    not 1 <= task.resources.native_threads <= task.resources.cpu_threads):
                raise ValueError("native threads must fit the phase CPU reservation")
            if task.resources.interop_threads is not None and (
                    type(task.resources.interop_threads) is not int or task.resources.interop_threads < 1
                    or task.resources.native_threads + task.resources.interop_threads > task.resources.cpu_threads):
                raise ValueError("native and explicit inter-op threads must fit the phase CPU reservation")
            if type(task.resources.defer_gpu) is not bool or task.resources.defer_gpu and (
                    task.resources.phase != "calibration" or not task.resources.device.startswith("cuda")
                    or task.resources.interop_threads is None):
                raise ValueError("lazy GPU admission requires an explicit target native/inter-op policy")
        requested.update((task.task_id, task) for task in additions)

    def run(self, tasks, *, on_finished=None, on_observation=None):
        requested = {}
        self._add_tasks(requested, tasks)
        diagnostics = PoolFailureDiagnostics(self.diagnostic_root,
            {**self.identity, "execution_backend": "isolated_cache_group_processes"})
        assert_no_prior_owned_live(self.diagnostic_root)
        ownership_identity = {**self.identity, "pool_instance_id": diagnostics.identity["pool_instance_id"],
                              "python_version": sys.version, "fork_exec_abi": _fork_exec_abi()}
        ownership = _CloudOwnershipJournal(diagnostics.directory, ownership_identity)
        ownership.start()
        ownership_settled = False
        self._processes = {}
        messages, acknowledgements = Queue(), Queue()
        stopped = threading.Event()
        snapshot_lock, latest_snapshot, observation_notified = threading.Lock(), None, False
        wake_receiver, wake_sender = Pipe(duplex=False)
        controller_errors, observation_failures = [], []

        def admission_failure(outcome, error):
            failure = _failure(error, "artifact_admission")
            if outcome.failure is None:
                outcome.status, outcome.failure = "failed", failure
            else:
                outcome.supplementary_failures.append(failure)

        def observe(*, consume_notice=False):
            nonlocal latest_snapshot, observation_notified
            if on_observation is None:
                return
            with snapshot_lock:
                if consume_notice:
                    observation_notified = False
                snapshot, latest_snapshot = latest_snapshot, None
            if snapshot is not None:
                try:
                    on_observation(snapshot)
                except Exception as error:
                    observation_failures.append(_failure(error, "observation_callback"))

        def schedule():
            nonlocal latest_snapshot
            results, active, pending_admissions = {}, {}, {}
            in_admission = None

            def publish_snapshot():
                nonlocal latest_snapshot, observation_notified
                if on_observation is not None:
                    snapshot = {"observed_unix": time.time(), "running": list(active),
                        "admitting": list(pending_admissions),
                        "pending": [key for key in requested
                                    if key not in results and key not in active and key not in pending_admissions],
                        "completed": [key for key, result in results.items() if result.status == "completed"],
                        "failed": [key for key, result in results.items() if result.status != "completed"],
                        "resources": self.admission.record()}
                    with snapshot_lock:
                        latest_snapshot = snapshot
                        notify = not observation_notified
                        observation_notified = True
                    if notify:
                        messages.put(("observe", None))

            previous_subreaper = _enable_task_subreaper()
            try:
                while len(results) < len(requested) and not stopped.is_set():
                    while True:
                        try:
                            key, outcome, additions = acknowledgements.get_nowait()
                        except Empty:
                            break
                        try:
                            self._add_tasks(requested, additions)
                        except Exception as error:
                            admission_failure(outcome, error)
                        results[key] = outcome
                        del pending_admissions[key]
                        in_admission = None
                    pending = [task for key, task in requested.items()
                               if key not in results and key not in active and key not in pending_admissions]
                    pending.sort(key=lambda task: not (task.resources.device.startswith("cuda")
                        and task.resources.phase in {"source", "calibration"}
                        and all(key in results and results[key].status == "completed" for key in task.depends_on)))
                    for task in pending:
                        if stopped.is_set():
                            break
                        failed = [key for key in task.depends_on if key in results and results[key].status != "completed"]
                        if failed:
                            pending_admissions[task.task_id] = CloudResult(task.task_id, "blocked",
                                failure={"type": "FailedDependency", "message": "declared dependency failed",
                                         "dependencies": failed, "stage": "dependency"})
                            continue
                        if len(active) >= self.max_workers or any(key not in results for key in task.depends_on):
                            continue
                        reservation = self.admission.reserve_phase(task.resources.phase,
                            cpu_threads=task.resources.cpu_threads, memory_bytes=task.resources.memory_bytes,
                            gpu_workers=int(task.resources.device.startswith("cuda") and not task.resources.defer_gpu),
                            task_id=task.task_id,
                            gpu_capable=task.resources.phase == "calibration" and task.resources.device.startswith("cuda"))
                        if reservation is None:
                            continue
                        receiver, sender = None, None
                        process, started = None, False
                        try:
                            receiver, sender = self.context.Pipe(duplex=True)
                            tree_token = uuid.uuid4().hex
                            process = _make_process(self.context, task, sender, diagnostics.directory, diagnostics.identity,
                                self.admission.input_admission, self.admission, tree_token)
                            ownership.register_launch(task, tree_token)
                            process.start()
                            started = True
                            ownership.launched(task.task_id, process)
                        except Exception as error:
                            if receiver is not None:
                                receiver.close()
                            launch = getattr(process, "_cloud_launch_identity", None)
                            retirement = getattr(process, "_cloud_launch_retirement", None)
                            if started and process is not None:
                                if process.is_alive():
                                    process.terminate()
                                process.join()
                                process.close()
                                retirement = {"backend": "controller_spawn_failure",
                                              "owned_processes": [launch] if launch else [],
                                              "active_owned_processes_after_exit": 0}
                            if ownership._task(task.task_id) is not None:
                                if (launch is None and retirement is None and not started
                                        and getattr(process, "_cloud_spawn_pid", None) is None):
                                    retirement = {"backend": "process_not_started",
                                                  "active_owned_processes_after_exit": 0}
                                ownership.spawn_failed(task.task_id, error, launch=launch,
                                                        retirement=retirement)
                            if isinstance(error, CloudLaunchUnverifiedError):
                                raise
                            self.admission.release_phase(reservation)
                            pending_admissions[task.task_id] = CloudResult(task.task_id, "failed",
                                failure=_failure(error, "spawn"))
                        else:
                            infrastructure = {(child.pid, child.create_time()) for child in psutil.Process().children()
                                              if child.pid != process.pid}
                            active[task.task_id] = _Running(task, reservation, process, receiver,
                                tree_token=tree_token, infrastructure_children=infrastructure,
                                launch_identity=ownership._task(task.task_id)["launch"])
                            self._processes[process.pid] = process
                            diagnostics.remember_processes(self)
                        finally:
                            if sender is not None:
                                sender.close()
                    if not active and not pending_admissions:
                        for key, task in requested.items():
                            if key not in results:
                                pending_admissions[key] = CloudResult(key, "blocked",
                                    failure={"type": "UnreachableDependencyOrCapacity",
                                             "message": "no runnable node and no active worker",
                                             "dependencies": list(task.depends_on), "stage": "dependency_or_capacity"})
                    if in_admission is None and pending_admissions:
                        in_admission = next(iter(pending_admissions))
                        messages.put(("admit", (in_admission, pending_admissions[in_admission])))
                    publish_snapshot()
                    if len(results) == len(requested) or stopped.is_set():
                        break
                    # ACK and cancellation share the wait set with actual
                    # worker handles. Long main-thread science never stops
                    # retirement, containment handshakes, or ready dispatch.
                    handles = [wake_receiver]
                    handles.extend(item.process.sentinel for item in active.values())
                    handles.extend(item.receiver for item in active.values() if item.receiving)
                    ready = wait(handles, timeout=self.poll_seconds)
                    if wake_receiver in ready:
                        while wake_receiver.poll():
                            wake_receiver.recv()
                    for key, item in list(active.items()):
                        if sys.platform == "linux" and item.bootstrap is not None:
                            _remember_task_tree(item)
                        if item.receiving and item.receiver in ready:
                            try:
                                message = item.receiver.recv()
                                if isinstance(message, dict) and message.get("event") == "ready":
                                    _admit_bootstrap(item, message)
                                    ownership.record_item(item)
                                else:
                                    if not isinstance(message, CloudResult) or message.task_id != key:
                                        raise ValueError("worker returned an invalid terminal result for the owned task")
                                    item.message = message
                            except EOFError:
                                item.receiving = False
                            except Exception as error:
                                item.message = CloudResult(key, "failed", failure=_failure(error, "bootstrap_or_result"))
                                item.receiving = False
                                if item.process.is_alive():
                                    item.process.terminate()
                            if item.message is not None or not item.receiving:
                                item.receiver.close()
                                item.receiving = False
                        if item.process.sentinel not in ready and item.process.is_alive():
                            continue
                        item.process.join()
                        # Preserve every currently active peer identity before
                        # retiring this task. A peer may be adopted by the
                        # subreaper while this leader exits; it must never be
                        # mistaken for a late descendant of the retiring task.
                        for peer_key, peer in active.items():
                            if peer_key == key:
                                continue
                            try:
                                item.infrastructure_children.add(
                                    (peer.process.pid, psutil.Process(peer.process.pid).create_time())
                                )
                                peer_process = psutil.Process(peer.process.pid)
                                for descendant in peer_process.children(recursive=True):
                                    item.infrastructure_children.add(
                                        (descendant.pid, descendant.create_time())
                                    )
                            except psutil.NoSuchProcess:
                                pass
                        _retire_tree(item)
                        ownership.record_item(item, status="retired" if item.bootstrap is not None else "unverified")
                        if item.receiving:
                            try:
                                item.message = item.receiver.recv()
                            except EOFError:
                                pass
                            item.receiver.close()
                        if item.message is not None and item.message.failure is not None:
                            outcome = item.message
                        elif item.process.exitcode != 0 or item.message is None:
                            error = CloudWorkerExitError(
                                f"worker {item.process.pid} exited {item.process.exitcode} without a valid terminal result")
                            observation = diagnostics.capture(error, {"task_id": key, "phase": item.task.resources.phase},
                                                              stage="process_exit")
                            worker = dict(diagnostics.process_identities[item.process.pid])
                            if worker.get("create_time") is None:
                                state = next((row.get("worker", {}) for row in diagnostics.events[-1]["worker_states"]
                                              if row.get("worker", {}).get("pid") == item.process.pid), {})
                                worker["create_time"] = state.get("create_time")
                            outcome = CloudResult(key, "failed", failure={**_failure(error, "process_exit"),
                                                                          "diagnostic_observation": observation}, worker=worker)
                        else:
                            outcome = item.message
                        outcome.worker = outcome.worker or dict(diagnostics.process_identities[item.process.pid])
                        outcome.worker.update(pid=item.process.pid, exit_code=item.process.exitcode, alive=False,
                                              parent_after_exit=system_snapshot())
                        outcome.worker["process_tree_retirement"] = _retire_tree(item)
                        if self.admission.input_admission is not None:
                            try:
                                self.admission.input_admission.retire_task(key)
                            except Exception as error:
                                failure = _failure(error, "artifact_settlement")
                                if outcome.failure is None:
                                    outcome.status, outcome.failure = "failed", failure
                                else:
                                    outcome.supplementary_failures.append(failure)
                        self.admission.release_phase(item.reservation)
                        del active[key]
                        pending_admissions[key] = outcome
                    publish_snapshot()
            finally:
                settlement_failures = []
                for item in active.values():
                    if item.process.is_alive():
                        item.process.terminate()
                    item.process.join()
                    _retire_tree(item)
                    try:
                        ownership.record_item(item, status="retired" if item.bootstrap is not None else "unverified")
                    except Exception as error:
                        settlement_failures.append(error)
                    if self.admission.input_admission is not None:
                        try:
                            self.admission.input_admission.retire_task(item.task.task_id)
                        except Exception as error:
                            settlement_failures.append(error)
                    item.receiver.close()
                    self.admission.release_phase(item.reservation)
                for process in self._processes.values():
                    if process.is_alive():
                        process.terminate()
                        process.join()
                    try:
                        process.close()
                    except ValueError:
                        # A process can have reached its sentinel between the
                        # liveness check and close without the controller's
                        # normal retirement path joining it.
                        process.join(timeout=1.)
                        if process.is_alive():
                            process.terminate()
                            process.join()
                        try:
                            process.close()
                        except ValueError:
                            # The OS identity has already been retired; keep
                            # the primary controller result when multiprocessing
                            # refuses a late close on the wrapper object.
                            pass
                _restore_task_subreaper(previous_subreaper)
                if settlement_failures:
                    primary = sys.exception()
                    if primary is None:
                        raise settlement_failures[0]
                    for error in settlement_failures:
                        primary.add_note(f"artifact settlement failed during retirement: {error}")
            return results

        def control():
            try:
                results = schedule()
            except BaseException as error:
                controller_errors.append(error)
                messages.put(("error", error))
            else:
                messages.put(("complete", results))

        controller = threading.Thread(target=control, name="cloud-process-controller")
        try:
            controller.start()
            while True:
                try:
                    kind, value = messages.get(timeout=self.poll_seconds)
                except Empty:
                    observe()
                    continue
                if kind == "observe":
                    observe(consume_notice=True)
                    continue
                if kind == "complete":
                    observe()
                    run = CloudRun(value, {"directory": str(diagnostics.directory), "identity": diagnostics.identity},
                                   observation_failures)
                    ownership.finish(run_status=run.status, settled=True)
                    ownership_settled = True
                    return run
                if kind == "error":
                    raise value
                key, outcome = value
                additions = ()
                if on_finished is not None:
                    from train.runtime import cpu_thread_budget

                    before_wall, before_cpu = time.perf_counter(), time.process_time()
                    try:
                        with cpu_thread_budget(max(1, self.admission.coordinator_threads)):
                            additions = tuple(on_finished(outcome) or ())
                    except Exception as error:
                        admission_failure(outcome, error)
                    finally:
                        if outcome.worker is not None:
                            outcome.worker["artifact_admission"] = {
                                "wall_seconds": time.perf_counter() - before_wall,
                                "coordinator_cpu_seconds": time.process_time() - before_cpu}
                acknowledgements.put((key, outcome, additions))
                wake_sender.send(None)
                observe()
        finally:
            primary = sys.exception()
            stopped.set()
            if controller.ident is not None:
                wake_sender.send(None)
                controller.join()
            wake_receiver.close()
            wake_sender.close()
            if primary is not None:
                for error in controller_errors:
                    if error is not primary:
                        primary.add_note(f"cloud controller failed during retirement: {error}")
            if not ownership_settled:
                try:
                    ownership.finish(run_status="failed", settled=False)
                except BaseException as journal_error:
                    if primary is None:
                        raise
                    primary.add_note(f"cloud ownership journal could not be published: {journal_error}")
