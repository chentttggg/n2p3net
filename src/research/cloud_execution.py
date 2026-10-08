"""One resource-admitted ready queue for cache-group scientific adapters.

Adapters keep their recipes, RNG and result schemas. Arguments carry input
identities and lightweight configuration; results carry artifact paths/facts,
never a live model or an EEG matrix. Each cache group owns one fresh process,
so a returned result does not return its phase reservation until that process
has actually exited. Persistent input/artifact leases belong to the supplied
admission owner and are not released with a process reservation.
"""

from __future__ import annotations

import gc
import importlib
import multiprocessing
import os
import pickle
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import wait
from pathlib import Path

import psutil

from research.worker_diagnostics import (
    PoolFailureDiagnostics,
    diagnostic_task,
    initialize_worker_diagnostics,
    system_snapshot,
    worker_diagnostic_phase,
)


@dataclass(frozen=True)
class PhaseResources:
    phase: str
    cpu_threads: int
    memory_bytes: int
    native_threads: int
    device: str = "cpu"


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
def _native_environment(threads, task_id):
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "TORCHINDUCTOR_COMPILE_THREADS", "N2P3_CLOUD_TASK_ID")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = (task_id if name == "N2P3_CLOUD_TASK_ID" else
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


def _worker_entry(encoded, sender, directory, identity):
    if os.name != "nt":
        os.setsid()
    task, input_admission = pickle.loads(encoded)
    from research.source_execution import install_input_admission

    install_input_admission(input_admission)
    initialize_worker_diagnostics(directory, {**identity, "cloud_task_id": task.task_id})
    sender.send({"event": "ready", "pid": os.getpid(), "create_time": psutil.Process().create_time(),
                 "process_group": os.getpgrp() if os.name != "nt" else None})
    if sender.recv() != "execute":
        return
    result = CloudResult(task.task_id, "completed")
    try:
        with diagnostic_task(task.output, {"task_id": task.task_id, "phase": task.resources.phase,
                "adapter": task.adapter, "resources": asdict(task.resources)}):
            worker_diagnostic_phase("adapter_import")
            function = _adapter(task.adapter)
            if "torch" in sys.modules:
                from train.runtime import configure_spawned_worker_threads, cpu_thread_budget

                configure_spawned_worker_threads(task.resources.native_threads)
                with cpu_thread_budget(task.resources.native_threads):
                    worker_diagnostic_phase("adapter_execution")
                    result.value = function(task.arguments)
            else:
                worker_diagnostic_phase("adapter_execution")
                result.value = function(task.arguments)
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
                     "cleanup_before": before, "cleanup_after": after}
    worker_diagnostic_phase("cache_group_cleanup_returned", status=result.status)
    try:
        sender.send(result)
    finally:
        sender.close()


def _make_process(context, task, sender, directory, identity, input_admission):
    # Encoding before creating a process makes a bad lightweight argument fail
    # before an unregistered child can be created by spawn's reduction step.
    encoded = pickle.dumps((task, input_admission), protocol=pickle.HIGHEST_PROTOCOL)
    return context.Process(target=_worker_entry, args=(encoded, sender, str(directory), identity))


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


def _admit_bootstrap(item, message):
    """Science waits for its own containment; a rejection belongs to this task."""
    if item.bootstrap is not None or message.get("pid") != item.process.pid:
        raise ValueError("bootstrap PID differs from the owned process handle or repeated its admission")
    if os.name == "nt":
        item.containment = _WindowsJob(item.process)
    elif message.get("process_group") != item.process.pid:
        raise ValueError("worker science must start inside its own process session")
    item.bootstrap = message
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


def _retire_tree(item):
    if item.retirement is not None:
        return item.retirement
    if item.containment is not None:
        item.retirement = item.containment.retire()
        return item.retirement
    if item.bootstrap is None:
        return {"backend": "bootstrap_not_admitted", "science_execution_started": False}
    group = item.bootstrap["process_group"]
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
    item.retirement = {"backend": "posix_session", "process_group": group,
            "retired_descendant_pids": [process.pid for process in members],
            "active_owned_processes_after_exit": 0, "orphan_zombies_may_await_init_reaping": True}
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
        requested.update((task.task_id, task) for task in additions)

    def run(self, tasks, *, on_finished=None, on_observation=None):
        requested = {}
        self._add_tasks(requested, tasks)
        diagnostics = PoolFailureDiagnostics(self.diagnostic_root,
            {**self.identity, "execution_backend": "isolated_cache_group_processes"})
        results, active, observation_failures = {}, {}, []
        self._processes = {}

        def finish(outcome):
            if on_finished is not None:
                before_wall, before_cpu = time.perf_counter(), time.process_time()
                try:
                    additions = tuple(on_finished(outcome) or ())
                    self._add_tasks(requested, additions)
                except Exception as error:
                    failure = _failure(error, "artifact_admission")
                    if outcome.failure is None:
                        outcome.status, outcome.failure = "failed", failure
                    else:
                        outcome.supplementary_failures.append(failure)
                finally:
                    if outcome.worker is not None:
                        outcome.worker["artifact_admission"] = {
                            "wall_seconds": time.perf_counter() - before_wall,
                            "coordinator_cpu_seconds": time.process_time() - before_cpu}
            results[outcome.task_id] = outcome

        def observe():
            if on_observation is not None:
                snapshot = {"observed_unix": time.time(), "running": list(active),
                    "pending": [key for key in requested if key not in results and key not in active],
                    "completed": [key for key, result in results.items() if result.status == "completed"],
                    "failed": [key for key, result in results.items() if result.status != "completed"],
                    "resources": self.admission.record()}
                try:
                    on_observation(snapshot)
                except Exception as error:
                    observation_failures.append(_failure(error, "observation_callback"))

        try:
            while len(results) < len(requested):
                pending = [task for key, task in requested.items() if key not in results and key not in active]
                # Prioritize already-ready GPU fits over further CPU ingress.
                # Each adapter retains its scientific order internally.
                pending.sort(key=lambda task: not (task.resources.device.startswith("cuda")
                    and task.resources.phase in {"source", "calibration"}
                    and all(key in results and results[key].status == "completed" for key in task.depends_on)))
                for task in pending:
                    failed = [key for key in task.depends_on if key in results and results[key].status != "completed"]
                    if failed:
                        finish(CloudResult(task.task_id, "blocked", failure={"type": "FailedDependency",
                            "message": "declared dependency failed", "dependencies": failed, "stage": "dependency"}))
                        continue
                    if len(active) >= self.max_workers or any(key not in results for key in task.depends_on):
                        continue
                    reservation = self.admission.reserve_phase(task.resources.phase,
                        cpu_threads=task.resources.cpu_threads, memory_bytes=task.resources.memory_bytes,
                        gpu_workers=int(task.resources.device.startswith("cuda")), task_id=task.task_id)
                    if reservation is None:
                        continue
                    receiver, sender = None, None
                    try:
                        receiver, sender = self.context.Pipe(duplex=True)
                        with _native_environment(task.resources.native_threads, task.task_id):
                            process = _make_process(self.context, task, sender, diagnostics.directory, diagnostics.identity,
                                                    self.admission.input_admission)
                            process.start()
                    except Exception as error:
                        if receiver is not None:
                            receiver.close()
                        self.admission.release_phase(reservation)
                        finish(CloudResult(task.task_id, "failed", failure=_failure(error, "spawn")))
                    else:
                        active[task.task_id] = _Running(task, reservation, process, receiver)
                        self._processes[process.pid] = process
                        diagnostics.remember_processes(self)
                    finally:
                        if sender is not None:
                            sender.close()
                observe()
                if not active:
                    remaining = [task for key, task in requested.items() if key not in results]
                    for task in remaining:
                        finish(CloudResult(task.task_id, "blocked", failure={"type": "UnreachableDependencyOrCapacity",
                            "message": "no runnable node and no active worker", "dependencies": list(task.depends_on),
                            "stage": "dependency_or_capacity"}))
                    continue
                # Rebuild the complete handles list after every dispatch. No
                # manager thread can snapshot it before a new worker registers.
                handles = [item.process.sentinel for item in active.values()]
                handles.extend(item.receiver for item in active.values() if item.receiving)
                ready = wait(handles, timeout=self.poll_seconds)
                for key, item in list(active.items()):
                    if item.receiving and item.receiver in ready:
                        try:
                            message = item.receiver.recv()
                            if isinstance(message, dict) and message.get("event") == "ready":
                                _admit_bootstrap(item, message)
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
                    # A forked descendant may still own a copy of the result
                    # pipe. Retire the owned tree before waiting for its EOF.
                    _retire_tree(item)
                    if item.receiving:
                        try:
                            item.message = item.receiver.recv()
                        except EOFError:
                            pass
                        item.receiver.close()
                    if item.message is not None and item.message.failure is not None:
                        outcome = item.message
                    elif item.process.exitcode != 0 or item.message is None:
                        error = CloudWorkerExitError(f"worker {item.process.pid} exited {item.process.exitcode} without a valid terminal result")
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
                    # Source heap, native allocator and CUDA context are gone;
                    # artifact/tmpfs leases remain with the admission owner.
                    self.admission.release_phase(item.reservation)
                    del active[key]
                    finish(outcome)
                observe()
        finally:
            settlement_failures = []
            for item in active.values():
                if item.process.is_alive():
                    item.process.terminate()
                item.process.join()
                _retire_tree(item)
                if self.admission.input_admission is not None:
                    try:
                        self.admission.input_admission.retire_task(item.task.task_id)
                    except Exception as error:
                        settlement_failures.append(error)
                item.receiver.close()
                self.admission.release_phase(item.reservation)
            for process in self._processes.values():
                process.close()
            if settlement_failures:
                primary = sys.exception()
                if primary is None:
                    raise settlement_failures[0]
                for error in settlement_failures:
                    primary.add_note(f"artifact settlement failed during retirement: {error}")
        return CloudRun(results, {"directory": str(diagnostics.directory), "identity": diagnostics.identity}, observation_failures)
