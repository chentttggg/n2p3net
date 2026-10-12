"""Sequential cloud CLI orchestration; science and resources stay in the CLI.

An independent supervisor owns each science process and its durable OS exit.
The queue attaches to it after interruption. A result digest is a process
fact, never a replacement for the science owner's admission.
"""
from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import psutil

from research.cloud_execution import _owned_process_alive, _retire_tree
from research.execution import atomic_write_json
from research.source_execution import process_file_lock

QUEUE_SCHEMA = "n2p3_dataset_queue/1"
REQUEST_SCHEMA = "n2p3_dataset_queue_request/2"
LAUNCH_SCHEMA = "n2p3_dataset_queue_launch/2"
TERMINAL_SCHEMA = "n2p3_dataset_queue_terminal/2"
EXIT_SCHEMA = "n2p3_dataset_queue_exit/2"
INVOCATION_FAILURE_SCHEMA = "n2p3_dataset_queue_invocation_failure/1"


class DatasetQueueError(RuntimeError):
    """Queue binding or lifecycle evidence is incomplete or has changed."""


class QueueBusyError(DatasetQueueError):
    """An orphan science process remains alive without its supervisor."""


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise DatasetQueueError(f"queue binding requires a regular file: {target}")
    with target.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _locator(path: str | Path, expected: str | None = None) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    digest = sha256_file(target)
    if expected is not None and digest != expected:
        raise DatasetQueueError(f"bound input changed: {target}")
    return {"path": str(target), "sha256": digest}


def _json_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _read_bound(path: Path, expected: str) -> dict:
    _locator(path, expected)
    return json.loads(path.read_bytes())


def _identity(process: psutil.Process) -> dict:
    return {"pid": process.pid, "create_time": process.create_time()}


def _pid_alive(identity: Mapping[str, object]) -> bool:
    try:
        process = psutil.Process(int(identity["pid"]))
        return process.create_time() == float(identity["create_time"]) and _owned_process_alive(process)
    except psutil.NoSuchProcess:
        return False


def _original_group(identity: Mapping[str, object], group: int) -> int | None:
    """Check the leader generation after a live owned member proves the group."""
    try:
        process = psutil.Process(int(identity["pid"]))
        if process.create_time() != float(identity["create_time"]):
            return None
    except psutil.NoSuchProcess:
        pass
    return group


def _same_identity(first: Mapping[str, object], second: Mapping[str, object]) -> bool:
    return first["pid"] == second["pid"] and first["create_time"] == second["create_time"]


def _bound_record(path: Path, schema: str, request: dict, request_sha: str) -> dict:
    record = json.loads(path.read_bytes())
    if (record["schema"] != schema or record["request_sha256"] != request_sha
            or record["dataset_id"] != request["dataset_id"]):
        raise DatasetQueueError(f"dataset lifecycle binding changed: {path.name}")
    return record


def _runtime_files() -> list[dict]:
    return [_locator(path) for path in (
        __file__, Path(__file__).resolve().parents[2] / "experiments/run_dataset_queue.py",
        atomic_write_json.__code__.co_filename, _retire_tree.__code__.co_filename,
        inspect.unwrap(process_file_lock).__code__.co_filename)]


def _verify_inputs(request: Mapping[str, object]) -> None:
    for binding in (request["source"], request["plan"], *request["runtime_files"]):
        _locator(binding["path"], binding["sha256"])


def _artifact_facts(request: Mapping[str, object]) -> tuple[list[dict], list[str]]:
    root = Path(request["output_dir"]).resolve()
    facts, missing = [], []
    for relative in request["result_files"]:
        raw = root / relative
        path = raw.resolve()
        if not path.is_relative_to(root) or raw.is_symlink():
            raise DatasetQueueError("dataset artifact escapes its independent output directory")
        if not path.is_file():
            missing.append(relative)
        else:
            facts.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return facts, missing


@dataclass(frozen=True)
class DatasetQueueEntry:
    dataset_id: str
    command: tuple[str, ...]
    output_dir: Path
    source: Mapping[str, object]
    plan: Mapping[str, object]
    result_files: tuple[str, ...] = ()
    cwd: Path | None = None
    env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id or self.dataset_id.strip() != self.dataset_id:
            raise ValueError("dataset_id must be an exact nonempty identifier")
        if (isinstance(self.command, (str, bytes)) or not self.command
                or any(not isinstance(value, str) or not value or "\x00" in value for value in self.command)):
            raise ValueError("dataset command must be a nonempty structured argv")
        files = tuple(self.result_files)
        if (not files or len(set(files)) != len(files)
                or any(not isinstance(value, str) or not value or Path(value).is_absolute()
                       or ".." in Path(value).parts for value in files)):
            raise ValueError("result_files must be unique relative paths within the dataset output")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        object.__setattr__(self, "result_files", files)
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd).resolve())

    def request_record(self, queue_root: Path) -> dict:
        if self.output_dir == queue_root or not self.output_dir.is_relative_to(queue_root):
            raise DatasetQueueError("dataset output must be a distinct directory inside queue output root")
        control_root = queue_root / ".dataset_queue"
        if self.output_dir.is_relative_to(control_root) or control_root.is_relative_to(self.output_dir):
            raise DatasetQueueError("dataset output overlaps the queue control namespace")
        bindings = []
        for supplied in (self.source, self.plan):
            binding = dict(supplied)
            binding.update(_locator(binding["path"], binding["sha256"]))
            bindings.append(binding)
        cwd = self.cwd or queue_root
        if not cwd.is_dir():
            raise DatasetQueueError(f"dataset cwd does not exist: {cwd}")
        return {"schema": REQUEST_SCHEMA, "dataset_id": self.dataset_id, "argv": list(self.command),
            "cwd": str(cwd), "env": None if self.env is None else dict(self.env),
            "output_dir": str(self.output_dir), "source": bindings[0], "plan": bindings[1],
            "result_files": list(self.result_files), "runtime_files": _runtime_files()}


@dataclass(frozen=True)
class DatasetQueuePlan:
    entries: tuple[DatasetQueueEntry, ...]
    queue_id: str = "dataset-queue"

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        ids = [entry.dataset_id for entry in entries]
        if not entries or len(set(ids)) != len(ids):
            raise ValueError("queue requires an ordered unique dataset roster")
        outputs = [entry.output_dir for entry in entries]
        for index, output in enumerate(outputs):
            if any(output.is_relative_to(other) or other.is_relative_to(output) for other in outputs[:index]):
                raise ValueError("dataset output directories must not overlap")
        object.__setattr__(self, "entries", entries)


def _terminal_payload(request: dict, request_sha: str, directory: Path, exit_record: dict) -> dict:
    if (exit_record["schema"] != EXIT_SCHEMA or exit_record["request_sha256"] != request_sha
            or exit_record["dataset_id"] != request["dataset_id"]
            or exit_record["resource_tree"]["active_owned_processes_after_exit"] != 0):
        raise DatasetQueueError("dataset external exit or resource-tree binding changed")
    launch = _bound_record(directory / "queue_launch.json", LAUNCH_SCHEMA, request, request_sha)
    if exit_record["supervisor"] != launch["supervisor"]:
        raise DatasetQueueError("dataset exit supervisor differs from registered owner")
    child_path = directory / "queue_child.json"
    if exit_record["child"] is not None:
        child = _bound_record(child_path, LAUNCH_SCHEMA, request, request_sha)
        if (not _same_identity(exit_record["child"], child["child"])
                or exit_record["child"]["process_group"] != child["process_group"]):
            raise DatasetQueueError("dataset exit child differs from registered invocation")
        if not _same_identity(exit_record["child"], exit_record["delegated_process"]):
            raise DatasetQueueError("dataset OS wait process differs from registered invocation")
    elif child_path.exists():
        raise DatasetQueueError("dataset exit discarded its registered invocation")
    invocation_failure_path = directory / "queue_exec_failure.json"
    if invocation_failure_path.exists():
        invocation_failure = _bound_record(
            invocation_failure_path, INVOCATION_FAILURE_SCHEMA, request, request_sha)
        if (exit_record["invocation_failure_sha256"] != sha256_file(invocation_failure_path)
                or not _same_identity(invocation_failure["child"], exit_record["child"])
                or exit_record["science_execution_started"]
                or exit_record["failure"] != invocation_failure["failure"]):
            raise DatasetQueueError("dataset delegated exec failure binding changed")
    elif exit_record["invocation_failure_sha256"] is not None:
        raise DatasetQueueError("dataset delegated exec failure receipt missing")
    if (not exit_record["science_execution_started"] and exit_record["exit_code"] is not None
            or exit_record["science_execution_started"]
            and exit_record["exit_code"] != exit_record["delegated_process_exit_code"]):
        raise DatasetQueueError("dataset scientific exit differs from actual delegated invocation")
    facts, missing = _artifact_facts(request)
    failure = exit_record["failure"]
    if failure is None and missing:
        failure = {"stage": "artifact_observation", "type": "MissingResultArtifact",
                   "message": "process exited without all declared result files"}
    status = "completed" if exit_record["exit_code"] == 0 and failure is None else "failed"
    return {"schema": TERMINAL_SCHEMA, "dataset_id": request["dataset_id"],
        "request_sha256": request_sha, "process_status": status,
        "scientific_admission_status": "delegated_to_science_owner" if status == "completed" else "partial",
        "exit_code": exit_record["exit_code"], "child": exit_record["child"],
        "science_execution_started": exit_record["science_execution_started"],
        "delegated_process_exit_code": exit_record["delegated_process_exit_code"],
        "artifact_facts": facts, "missing_result_files": missing, "failure": failure,
        "resource_tree": exit_record["resource_tree"],
        "external_exit_receipt_sha256": sha256_file(directory / "queue_exit.json")}


def _terminal_from_exit(request: dict, request_sha: str, directory: Path, exit_record: dict) -> dict:
    terminal = _terminal_payload(request, request_sha, directory, exit_record)
    atomic_write_json(directory / "queue_terminal.json", terminal)
    return terminal


def _dispatch_failure_terminal(request: dict, request_sha: str, directory: Path) -> dict:
    path = directory / "queue_dispatch_failure.json"
    failure = _bound_record(path, INVOCATION_FAILURE_SCHEMA, request, request_sha)
    facts, missing = _artifact_facts(request)
    return {"schema": TERMINAL_SCHEMA, "dataset_id": request["dataset_id"],
        "request_sha256": request_sha, "process_status": "failed", "exit_code": None,
        "science_execution_started": False, "delegated_process_exit_code": None,
        "scientific_admission_status": "partial", "child": None,
        "artifact_facts": facts, "missing_result_files": missing, "failure": failure["failure"],
        "external_invocation_failure_sha256": sha256_file(path),
        "resource_tree": {"backend": "supervisor_not_spawned", "active_owned_processes_after_exit": 0}}


def _child_command(mode: str, request_path: Path, request_sha: str) -> list[str]:
    runner = Path(__file__).resolve().parents[2] / "experiments/run_dataset_queue.py"
    return [sys.executable, "-u", "-B", str(runner), mode, str(request_path),
            "--request-sha256", request_sha]


def execute_queue_entry(request_path: Path, request_sha: str) -> int:
    """Register before exec so parent death cannot leave an unbound science PID."""
    request = _read_bound(request_path, request_sha)
    _verify_inputs(request)
    directory = request_path.parent
    atomic_write_json(directory / "queue_child.json", {
        "schema": LAUNCH_SCHEMA, "dataset_id": request["dataset_id"], "request_sha256": request_sha,
        "child": _identity(psutil.Process()), "process_group": os.getpgrp(),
        "invocation_state": "registered_before_exec"})
    try:
        environment = dict(os.environ)
        if request["env"] is not None:
            environment.update(request["env"])
        os.chdir(request["cwd"])
        os.execvpe(request["argv"][0], request["argv"], environment)
    except OSError as error:
        atomic_write_json(directory / "queue_exec_failure.json", {
            "schema": INVOCATION_FAILURE_SCHEMA, "dataset_id": request["dataset_id"],
            "request_sha256": request_sha, "child": _identity(psutil.Process()),
            "science_execution_started": False,
            "failure": {"stage": "science_exec", "type": type(error).__name__,
                        "message": str(error), "errno": error.errno},
            "observed_unix": time.time(), "observed_by": "registered_guardian.exec_exception"})
        return 1


def _remember_children(owned: dict[int, dict]) -> None:
    for process in psutil.Process().children(recursive=True):
        try:
            owned[process.pid] = {**_identity(process), "process_group": os.getpgid(process.pid)}
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass


def _retire_owned_tree(owned: dict[int, dict]) -> dict:
    """Reuse CloudExecutor's retirement for every verified owned process group."""
    groups = set()
    for identity in owned.values():
        if _pid_alive(identity):
            try:
                group = os.getpgid(identity["pid"])
            except ProcessLookupError:
                continue
            if group in owned and _original_group(owned[group], group) is None:
                raise DatasetQueueError("owned process group leader identity changed")
            groups.add(group)
    retirements = []
    for group in sorted(groups):
        if group == os.getpgrp():
            raise DatasetQueueError("science descendant joined the supervisor's own process group")
        state = SimpleNamespace(retirement=None, containment=None, bootstrap={"process_group": group})
        retirements.append(_retire_tree(state))
    active = [identity for identity in owned.values() if _pid_alive(identity)]
    if active:
        raise DatasetQueueError("owned science descendants remain alive after retirement")
    return {"backend": "verified_owned_process_groups", "groups": retirements,
            "owned_processes": list(owned.values()), "active_owned_processes_after_exit": 0}


def _retire_supervised_tree(owned: dict[int, dict]) -> dict:
    retirements = []
    while True:
        _remember_children(owned)
        tree = _retire_owned_tree(owned)
        retirements.extend(tree["groups"])
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if not pid:
                break
        # A signal handler may fork after the retirement snapshot. Subreaper
        # ownership catches these adopted children before the next dataset.
        _remember_children(owned)
        if not any(_pid_alive(identity) for identity in owned.values()):
            return {"backend": "linux_subreaper_and_cloud_process_groups", "groups": retirements,
                    "owned_processes": list(owned.values()), "active_owned_processes_after_exit": 0}


def supervise_queue_entry(request_path: Path, request_sha: str) -> int:
    """Survive queue-parent loss, observe real wait(), retire tree, persist exit."""
    if sys.platform != "linux":
        raise DatasetQueueError("dataset process supervision requires the Linux cloud runtime")
    request = _read_bound(request_path, request_sha)
    _verify_inputs(request)
    directory = request_path.parent
    output = Path(request["output_dir"])
    output_lock = output.parent / ("." + output.name + ".dataset_queue.lock")
    with process_file_lock(output_lock):
        if (directory / "queue_exit.json").exists():
            return 0
        if (directory / "queue_launch.json").exists():
            raise DatasetQueueError("a registered dataset supervisor cannot be restarted")
        # Adopt setsid descendants after a native leader exit as well as its
        # ordinary group members. The science CLI keeps its original topology.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "cannot establish dataset descendant ownership")
        supervisor = _identity(psutil.Process())
        launch = {"schema": LAUNCH_SCHEMA, "dataset_id": request["dataset_id"],
                  "request_sha256": request_sha, "supervisor": supervisor, "started_unix": time.time()}
        atomic_write_json(directory / "queue_launch.json", launch)
        child = None
        failure = None
        code = None
        owned = {}
        tree = {"backend": "no_science_process", "active_owned_processes_after_exit": 0}
        cleanup_failure = None
        try:
            if output.exists():
                raise DatasetQueueError("unowned science output already exists")
            with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
                child = subprocess.Popen(_child_command("--execute-entry", request_path, request_sha),
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
                owned[child.pid] = {**_identity(psutil.Process(child.pid)), "process_group": child.pid}
                previous_tree = None
                while child.poll() is None:
                    _remember_children(owned)
                    if owned != previous_tree:
                        atomic_write_json(directory / "queue_tree.json", {
                            "schema": LAUNCH_SCHEMA, "dataset_id": request["dataset_id"],
                            "request_sha256": request_sha, "supervisor": supervisor,
                            "delegated_process": owned[child.pid],
                            "owned_processes": list(owned.values())})
                        previous_tree = dict(owned)
                    time.sleep(.05)
                code = child.wait()
            _verify_inputs(request)
        except Exception as error:
            failure = {"stage": "supervised_invocation", "type": type(error).__name__, "message": str(error)}
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                code = child.wait()
        if child is not None:
            try:
                tree = _retire_supervised_tree(owned)
            except Exception as error:
                cleanup_failure = {"stage": "resource_retirement", "type": type(error).__name__,
                                   "message": str(error)}
                tree = {"backend": "unverified_retirement", "owned_processes": list(owned.values()),
                        "active_owned_processes_after_exit": None}
        child_path = directory / "queue_child.json"
        identity = None
        if child_path.exists():
            registered = _bound_record(child_path, LAUNCH_SCHEMA, request, request_sha)
            identity = {**registered["child"], "process_group": registered["process_group"]}
        invocation_failure_path = directory / "queue_exec_failure.json"
        invocation_failure_sha = None
        science_started = identity is not None
        if invocation_failure_path.exists():
            invocation_failure = _bound_record(
                invocation_failure_path, INVOCATION_FAILURE_SCHEMA, request, request_sha)
            if identity is None or not _same_identity(invocation_failure["child"], identity):
                raise DatasetQueueError("delegated exec failure has no matching registered process")
            science_started = False
            failure = invocation_failure["failure"]
            invocation_failure_sha = sha256_file(invocation_failure_path)
        elif child is not None and identity is None:
            failure = {"stage": "guardian_bootstrap", "type": "ScienceBootstrapFailed",
                       "message": f"guardian exited with code {code} before registering the science invocation"}
        elif code and failure is None:
            failure = {"stage": "process_exit", "type": "ScienceProcessFailed",
                       "message": f"science process exited with code {code}"}
        exit_record = {"schema": EXIT_SCHEMA, "dataset_id": request["dataset_id"],
            "request_sha256": request_sha, "supervisor": supervisor, "child": identity,
            "delegated_process": owned.get(child.pid) if child is not None else None,
            "science_execution_started": science_started,
            "exit_code": code if science_started else None, "delegated_process_exit_code": code,
            "failure": failure, "cleanup_failure": cleanup_failure, "resource_tree": tree,
            "invocation_failure_sha256": invocation_failure_sha,
            "observed_unix": time.time(), "observed_by": "independent_dataset_supervisor.wait"
            if child is not None else "independent_dataset_supervisor.spawn_exception"}
        atomic_write_json(directory / "queue_exit.json", exit_record)
        if cleanup_failure is not None:
            raise DatasetQueueError("dataset exit preserved but resource-tree retirement is unverified")
        _terminal_from_exit(request, request_sha, directory, exit_record)
    return 0


@dataclass
class DatasetQueue:
    plan: DatasetQueuePlan
    output_root: Path
    poll_seconds: float = .25

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root).resolve()
        if not 0 < self.poll_seconds < float("inf"):
            raise ValueError("poll_seconds must be positive")
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pid_alive(identity: Mapping[str, object]) -> bool:
        return _pid_alive(identity)

    def _entry_dir(self, entry: DatasetQueueEntry) -> Path:
        index = self.plan.entries.index(entry)
        directory = self.output_root / ".dataset_queue" / f"entry_{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _bound_request(self, entry: DatasetQueueEntry, directory: Path) -> tuple[dict, str]:
        request = entry.request_record(self.output_root)
        path = directory / "queue_request.json"
        if path.exists():
            prior = json.loads(path.read_bytes())
            if _json_hash(prior) != _json_hash(request):
                raise DatasetQueueError(f"dataset request binding changed: {entry.dataset_id}")
        else:
            atomic_write_json(path, request)
        return request, sha256_file(path)

    def _recover_existing(self, entry: DatasetQueueEntry, directory: Path, request_sha: str) -> dict | None:
        terminal_path, exit_path = directory / "queue_terminal.json", directory / "queue_exit.json"
        request = _read_bound(directory / "queue_request.json", request_sha)
        launch_path = directory / "queue_launch.json"
        launch = json.loads(launch_path.read_bytes()) if launch_path.exists() else None
        if launch is not None and (launch["schema"] != LAUNCH_SCHEMA or launch["request_sha256"] != request_sha
                                   or launch["dataset_id"] != entry.dataset_id):
            raise DatasetQueueError("dataset supervisor binding changed")
        prior_terminal = terminal_path.exists()
        if prior_terminal:
            terminal = json.loads(terminal_path.read_bytes())
            if (terminal["schema"] != TERMINAL_SCHEMA or terminal["request_sha256"] != request_sha
                    or terminal["dataset_id"] != entry.dataset_id):
                raise DatasetQueueError("dataset terminal binding changed")
            if terminal["process_status"] == "unknown" and exit_path.exists():
                return _terminal_from_exit(request, request_sha, directory, json.loads(exit_path.read_bytes()))
            if terminal["process_status"] != "unknown":
                if "external_invocation_failure_sha256" in terminal:
                    expected = _dispatch_failure_terminal(request, request_sha, directory)
                    if terminal != expected:
                        raise DatasetQueueError("dataset terminal differs from its bound dispatch failure")
                    return terminal
                if terminal["external_exit_receipt_sha256"] != sha256_file(exit_path):
                    raise DatasetQueueError("dataset exit receipt missing or changed")
                expected = _terminal_payload(request, request_sha, directory, json.loads(exit_path.read_bytes()))
                if (expected["artifact_facts"] != terminal["artifact_facts"]
                        or expected["missing_result_files"] != terminal["missing_result_files"]):
                    raise DatasetQueueError("dataset artifacts changed after terminal publication")
                if terminal != expected:
                    raise DatasetQueueError("dataset terminal differs from its bound external exit")
                return terminal
            if launch is not None and _pid_alive(launch["supervisor"]):
                return None
        if launch is not None and _pid_alive(launch["supervisor"]):
            return None  # Attach and observe; the supervisor owns a late exit.
        if exit_path.exists():
            return _terminal_from_exit(request, request_sha, directory, json.loads(exit_path.read_bytes()))
        if (directory / "queue_dispatch_failure.json").exists():
            terminal = _dispatch_failure_terminal(request, request_sha, directory)
            atomic_write_json(terminal_path, terminal)
            return terminal
        child_path = directory / "queue_child.json"
        if child_path.exists():
            registered = _bound_record(child_path, LAUNCH_SCHEMA, request, request_sha)
            if _pid_alive(registered["child"]):
                raise QueueBusyError(f"science child remains alive without supervisor: {entry.dataset_id}")
        retirement = None
        tree_path = directory / "queue_tree.json"
        if tree_path.exists():
            tree = _bound_record(tree_path, LAUNCH_SCHEMA, request, request_sha)
            if launch is None or tree["supervisor"] != launch["supervisor"]:
                raise DatasetQueueError("dataset process-tree supervisor binding changed")
            if _pid_alive(tree["delegated_process"]):
                raise QueueBusyError("delegated science invocation remains alive without supervisor")
            retirement = _retire_owned_tree({identity["pid"]: identity for identity in tree["owned_processes"]})
        dispatch = directory / "queue_dispatch.json"
        if not prior_terminal and not launch_path.exists() and not dispatch.exists() and not child_path.exists():
            return None
        if not launch_path.exists() and dispatch.exists():
            command = _bound_record(dispatch, LAUNCH_SCHEMA, request, request_sha)["command"]
            for process in psutil.process_iter(["cmdline"]):
                if process.info["cmdline"] == command and _owned_process_alive(process):
                    return None  # Dispatch-to-registration gap; never spawn twice.
        facts, missing = _artifact_facts(request)
        terminal = {"schema": TERMINAL_SCHEMA, "dataset_id": entry.dataset_id,
            "request_sha256": request_sha, "process_status": "unknown", "exit_code": None,
            "scientific_admission_status": "unknown_missing_external_exit",
            "artifact_facts": facts, "missing_result_files": missing,
            "failure": {"stage": "recovery", "type": "MissingExitReceipt",
                        "message": "supervisor is absent without a durable OS exit receipt"}}
        if retirement is not None:
            terminal["resource_tree"] = retirement
        atomic_write_json(terminal_path, terminal)
        return terminal

    def _spawn(self, entry: DatasetQueueEntry, request: dict, request_sha: str, directory: Path) -> None:
        command = _child_command("--supervise-entry", directory / "queue_request.json", request_sha)
        atomic_write_json(directory / "queue_dispatch.json", {"schema": LAUNCH_SCHEMA, "command": command,
            "dataset_id": entry.dataset_id, "request_sha256": request_sha})
        try:
            with (directory / "supervisor.stdout.log").open("xb") as stdout, (
                    directory / "supervisor.stderr.log").open("xb") as stderr:
                subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 start_new_session=True)
        except OSError as error:
            atomic_write_json(directory / "queue_dispatch_failure.json", {
                "schema": INVOCATION_FAILURE_SCHEMA, "dataset_id": entry.dataset_id,
                "request_sha256": request_sha, "controller": _identity(psutil.Process()),
                "science_execution_started": False, "observed_by": "queue_controller.spawn_exception",
                "observed_unix": time.time(), "failure": {"stage": "supervisor_dispatch",
                    "type": type(error).__name__, "message": str(error), "errno": error.errno}})

    def _run_entry(self, entry: DatasetQueueEntry) -> dict:
        directory = self._entry_dir(entry)
        request, request_sha = self._bound_request(entry, directory)
        terminal = self._recover_existing(entry, directory, request_sha)
        if terminal is not None:
            return terminal
        if not (directory / "queue_dispatch.json").exists():
            self._spawn(entry, request, request_sha, directory)
        while True:
            time.sleep(self.poll_seconds)
            terminal = self._recover_existing(entry, directory, request_sha)
            if terminal is not None:
                return terminal

    def _bind_declaration(self) -> None:
        declaration = {"schema": QUEUE_SCHEMA, "queue_id": self.plan.queue_id,
            "entries": [entry.request_record(self.output_root) for entry in self.plan.entries]}
        control = self.output_root / ".dataset_queue"
        original_path = control / "queue.json"
        if not original_path.exists():
            atomic_write_json(original_path, {**declaration, "previous_declaration_sha256": None})
            return
        previous_path = original_path
        previous = json.loads(previous_path.read_bytes())
        extensions = sorted((control / "declarations").glob("extension_*.json"))
        for index, path in enumerate(extensions, start=1):
            extension = json.loads(path.read_bytes())
            if (path.name != f"extension_{index:06d}.json"
                    or extension["previous_declaration_sha256"] != sha256_file(previous_path)):
                raise DatasetQueueError("queue declaration extension chain changed")
            self._validate_append(previous, extension)
            previous_path, previous = path, extension
        self._validate_append(previous, declaration)
        if len(declaration["entries"]) > len(previous["entries"]):
            path = control / "declarations" / f"extension_{len(extensions) + 1:06d}.json"
            atomic_write_json(path, {**declaration,
                "previous_declaration_sha256": sha256_file(previous_path)})

    @staticmethod
    def _validate_append(previous: dict, current: dict) -> None:
        count = len(previous["entries"])
        if (previous["schema"] != QUEUE_SCHEMA or current["schema"] != QUEUE_SCHEMA
                or current["queue_id"] != previous["queue_id"]
                or len(current["entries"]) < count
                or _json_hash(current["entries"][:count]) != _json_hash(previous["entries"])):
            raise DatasetQueueError("ordered queue declaration prefix changed")

    def run(self) -> dict:
        if sys.platform != "linux":
            raise DatasetQueueError("dataset queue execution requires the Linux cloud runtime")
        with process_file_lock(self.output_root / ".dataset_queue/queue.lock"):
            self._bind_declaration()
            results = [self._run_entry(entry) for entry in self.plan.entries]
            summary = {"schema": QUEUE_SCHEMA, "queue_id": self.plan.queue_id,
                "orchestration_status": "all_entries_processed",
                "status": "completed" if all(row["process_status"] == "completed" for row in results) else "failed",
                "ordered_dataset_ids": [entry.dataset_id for entry in self.plan.entries], "results": results,
                "scientific_admission_scope": "delegated_to_existing_science_owner",
                "completed_scientific_results_are_not_inferred_by_queue": True}
            atomic_write_json(self.output_root / "queue_terminal.json", summary)
            return summary


def plan_from_record(record: Mapping[str, object], *, output_root: str | Path) -> DatasetQueuePlan:
    if record["schema"] != QUEUE_SCHEMA or not isinstance(record["datasets"], list):
        raise ValueError("dataset queue plan requires schema and ordered datasets")
    root = Path(output_root).resolve()
    entries = [DatasetQueueEntry(dataset_id=item["dataset_id"], command=item["argv"],
        output_dir=root / item["output_dir"], source=item["source"], plan=item["plan"],
        result_files=item["result_files"], cwd=item.get("cwd"), env=item.get("env"))
        for item in record["datasets"]]
    return DatasetQueuePlan(tuple(entries), queue_id=record.get("queue_id", "dataset-queue"))
