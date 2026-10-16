"""One original BNCI source fit, an independent observer, and read-only export.

The worker announces its own live identity before permission to enter science.
On Windows the observer holds that process's OS handle, independently of the
venv redirector, and contains both in the existing Windows Job owner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

PLAN_SCHEMA = "n2p3_original_continuous_source_stage_plan/1"
SCOPE = "software_correctness"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _json(path, record):
    path = Path(path)
    content = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    temporary = path.with_name(path.name+".writing")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _artifact(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def validate_source_stage_plan(plan):
    """Require the declared scientific inputs before any worker or fitter starts."""
    required = {"schema", "source_id", "source_evidence_scope", "actual_source_seed", "native_threads",
        "source_inputs", "target_input", "geometry", "authorization_preprocessing", "fit",
        "target_identity", "exclusion_scope", "input_manifest", "minimum_available_memory_bytes", "handshake_timeout_seconds"}
    if set(plan) != required or plan["schema"] != PLAN_SCHEMA:
        raise ValueError("source stage requires its complete explicit plan")
    if (plan["source_evidence_scope"] != SCOPE or plan["actual_source_seed"] is not None
            or type(plan["native_threads"]) is not int or plan["native_threads"] != 1
            or plan["exclusion_scope"] != "target_participant"
            or not isinstance(plan["source_id"], str) or not plan["source_id"]):
        raise ValueError("this deterministic software source stage requires null scientific seed and native1")
    if (set(plan["geometry"]) != {"channels", "sfreq", "unit", "reference", "operator_identity", "observation_rank"}
            or not plan["authorization_preprocessing"]):
        raise ValueError("source stage requires explicit geometry and preprocessing declarations")
    fit = plan["fit"]
    if set(fit) != {"basis", "ridge", "noise_order", "conditioning_samples", "degrees_of_freedom",
            "relative_noise_shape", "response_seconds", "pre_roll_seconds"}:
        raise ValueError("source fit parameters must all be explicit")
    basis = fit["basis"]
    if (not basis or not basis[0] or any(len(row) != len(basis[0]) for row in basis)
            or not all(math.isfinite(value) for row in basis for value in row)
            or type(fit["noise_order"]) is not int or fit["noise_order"] < 0
            or type(fit["conditioning_samples"]) is not int
            or fit["conditioning_samples"] < max(2, fit["noise_order"])):
        raise ValueError("source basis and actual startup/order are inconsistent")
    for name in ("ridge", "degrees_of_freedom", "relative_noise_shape", "response_seconds"):
        if not math.isfinite(fit[name]) or fit[name] <= 0:
            raise ValueError("source positive physical fit parameters must be finite")
    if not math.isfinite(fit["pre_roll_seconds"]) or fit["pre_roll_seconds"] < 0:
        raise ValueError("source pre-roll must be finite and nonnegative")
    inputs = plan["source_inputs"]
    if not inputs or not isinstance(inputs, list):
        raise ValueError("source stage needs a nonempty ordered original MAT cohort")
    for row in (*inputs, plan["target_input"]):
        if set(row) != {"path", "sha256", "bytes", "source_relative_path", "budget", "input_generation_seed"}:
            raise ValueError("each original MAT requires bytes, identity, dose and separate input-generation seed")
        relative = PurePosixPath(row["source_relative_path"])
        if (relative.name != str(relative) or relative.name not in {f"A{i:02d}.mat" for i in range(1, 9)}
                or len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"])
                or type(row["bytes"]) is not int or row["bytes"] < 1
                or type(row["budget"]) is not int or row["budget"] < 0
                or row["input_generation_seed"] is not None
                and (type(row["input_generation_seed"]) is not int or row["input_generation_seed"] < 0)):
            raise ValueError("source MAT identity, dose or generation seed is invalid")
    if (any(row["budget"] < 1 for row in inputs) or plan["target_input"]["budget"] != 0
            or len({row["sha256"] for row in inputs}) != len(inputs)
            or len({row["source_relative_path"] for row in inputs}) != len(inputs)
            or plan["target_input"]["source_relative_path"] in {row["source_relative_path"] for row in inputs}
            or plan["target_input"]["sha256"] in {row["sha256"] for row in inputs}):
        raise ValueError("source cohort must exclude the actual zero-label target")
    target = plan["target_identity"]
    if (set(target) != {"participant_key", "source_dataset_id", "source_relative_path", "source_sha256"}
            or not target["participant_key"] or target["source_dataset_id"] != "BNCI2014-008"
            or target["source_relative_path"] != plan["target_input"]["source_relative_path"]
            or target["source_sha256"] != plan["target_input"]["sha256"]):
        raise ValueError("source stage needs its explicit original BNCI target exclusion identity")
    if (type(plan["minimum_available_memory_bytes"]) is not int or plan["minimum_available_memory_bytes"] < 1
            or not math.isfinite(plan["handshake_timeout_seconds"]) or plan["handshake_timeout_seconds"] <= 0):
        raise ValueError("source stage needs explicit memory and startup budgets")
    return plan


def _checked_bytes(row):
    content = Path(row["path"]).read_bytes()
    if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError("original input bytes differ from the frozen stage plan")
    return content


def _cohort(plan):
    from data.epochs import parse_preprocessing_spec
    from research.joint_flat_authority import _geometry
    from transfer.ensemble_views import CalibrationPrefixCompiler

    preprocessing = parse_preprocessing_spec(plan["authorization_preprocessing"])
    manifest = json.loads(_checked_bytes(plan["input_manifest"]))
    declared = {(str(Path(row["path"]).resolve()), row["sha256"], row["bytes"]) for row in manifest["rows"]}
    if any((str(Path(row["path"]).resolve()), row["sha256"], row["bytes"]) not in declared
            for row in (*plan["source_inputs"], plan["target_input"])):
        raise ValueError("original MAT plan is absent from its bound software input manifest")
    fit = plan["fit"]
    compilers = []
    for row in (*plan["source_inputs"], plan["target_input"]):
        compiler = CalibrationPrefixCompiler(_checked_bytes(row), source_sha256=row["sha256"],
            source_relative_path=row["source_relative_path"], budgets=(row["budget"],), profiles=(),
            authorization_preprocessing=preprocessing, response_seconds=fit["response_seconds"],
            pre_roll_seconds=fit["pre_roll_seconds"])
        prefix = compiler.compile(row["budget"])
        # Source training and target exclusion must share the declared physical geometry.
        if _geometry(prefix).record() != plan["geometry"]:
            raise ValueError("actual original compiler geometry differs from the explicit source plan")
        compilers.append((compiler, prefix))
    target = compilers[-1][1].target_contract
    if any(target[key] != value for key, value in plan["target_identity"].items()):
        raise ValueError("actual zero-label target differs from the explicitly excluded target identity")
    return tuple(compilers[:-1]), target


def _identity(process):
    return {"pid": process.pid, "create_time": process.create_time()}


def _same_process(identity):
    import psutil

    try:
        process = psutil.Process(identity["pid"])
        return process if (process.create_time() == identity["create_time"] and process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE) else None
    except psutil.NoSuchProcess:
        return None


def _lineage(process):
    import psutil

    result = []
    while process is not None:
        try:
            result.append(_identity(process))
            process = process.parent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
    return result


def _imported_sources(root):
    result = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(root/"src"):
            result[path.relative_to(root).as_posix()] = _artifact(path)
    return result


class _ScientificExit:
    """Hold the scientific process itself until its OS exit is observed."""

    def __init__(self, identity, launcher):
        self.launcher, self.identity, self.handle = launcher, identity, None
        if _same_process(identity) is None:
            raise ValueError("scientific worker identity is no longer live")
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            self.kernel.WaitForSingleObject.restype = wintypes.DWORD
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.handle = self.kernel.OpenProcess(0x00101101, False, identity["pid"])
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if _same_process(identity) is None:
                self.close()
                raise ValueError("scientific PID birth changed while its OS handle was opened")
        elif identity["pid"] != launcher.pid:
            raise ValueError("non-Windows source stage requires its directly waitable scientific child")

    def poll(self):
        if self.handle is None:
            return self.launcher.poll()
        import ctypes
        from ctypes import wintypes

        state = self.kernel.WaitForSingleObject(self.handle, 0)
        if state == 258:  # WAIT_TIMEOUT: a live process can also return STILL_ACTIVE.
            return None
        if state != 0:  # WAIT_OBJECT_0: the process handle is signalled.
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(code.value)

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _retire_uncontained(owned, remember, launcher_pid):
    """Freeze before tree discovery, so a failed bootstrap cannot orphan a shim child."""
    import psutil

    suspended = set()
    while True:
        for identity in tuple(owned.values()):
            process = _same_process(identity)
            if process is not None and process.pid not in suspended:
                try:
                    process.suspend()
                    suspended.add(process.pid)
                except psutil.NoSuchProcess:
                    pass
        remember()
        live = [process for identity in owned.values() if (process := _same_process(identity)) is not None]
        if all(process.pid in suspended for process in live):
            break
    # All known roots and descendants are suspended before any parent is killed.
    # The caller reaps its launcher separately to retain Popen's actual OS exit.
    for process in reversed(live):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs([process for process in live if process.pid != launcher_pid], timeout=2)


def _event(directory, phase, **fields):
    import psutil

    record = {"phase": phase, "time": time.time(), "monotonic_ns": time.monotonic_ns(),
        **_identity(psutil.Process()), "source_evidence_scope": SCOPE, **fields}
    with (directory/"events.jsonl").open("ab") as stream:
        stream.write((json.dumps(record, sort_keys=True, allow_nan=False)+"\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return record


def _worker(plan_path, plan_sha256, directory):
    import psutil

    plan_bytes = plan_path.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != plan_sha256:
        raise ValueError("worker plan bytes changed before startup")
    plan = validate_source_stage_plan(json.loads(plan_bytes))
    for name in THREAD_VARIABLES:
        os.environ[name] = str(plan["native_threads"])
    identity = _identity(psutil.Process())
    ready = {"schema": "n2p3_source_stage_worker_ready/1", "identity": identity,
        "lineage": _lineage(psutil.Process()), "plan_sha256": plan_sha256,
        "sys_executable": sys.executable, "base_executable": sys._base_executable,
        "sys_prefix": sys.prefix, "native_threads": plan["native_threads"]}
    _json(directory/"worker_ready.json", ready)
    deadline = time.monotonic()+plan["handshake_timeout_seconds"]
    while not (directory/"observer_go.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("independent source observer did not issue its live identity gate")
        time.sleep(.05)
    gate = json.loads((directory/"observer_go.json").read_bytes())
    if (gate["scientific_identity"] != identity or gate["plan_sha256"] != plan_sha256
            or gate["ready_sha256"] != _artifact(directory/"worker_ready.json")["sha256"]):
        raise ValueError("observer gate belongs to another scientific process or plan")
    root = Path(__file__).resolve().parents[2]
    count = {"original_continuous_fit_calls": 0, "successful_source_fits": 0}
    source_before = {}
    failure = None
    try:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits

        from research.joint_flat_authority import build_source_parameters
        from transfer.continuous_decision import ContinuousDecisionModel
        from transfer.joint_flat_programme import FrozenJointFlatSource

        cohort, target = _cohort(plan)
        source_before = _imported_sources(root)
        original_fit = ContinuousDecisionModel.fit.__func__

        def counted_fit(cls, *args, **kwargs):
            if count["original_continuous_fit_calls"]:
                raise ValueError("one source genesis cannot invoke a second original fit")
            count["original_continuous_fit_calls"] += 1
            _event(directory, "source_fit_started", plan_sha256=plan_sha256,
                actual_public_source_fit_attempts=1, actual_source_fit_attempts=1,
                source_fit_scope="explicit_participant_excluded_software_programme", actual_source_seed=None)
            return original_fit(cls, *args, **kwargs)

        ContinuousDecisionModel.fit = classmethod(counted_fit)
        fit = plan["fit"]
        with threadpool_limits(1):
            parameters, model = build_source_parameters(cohort, target_participant_key=target["participant_key"],
                degrees_of_freedom=fit["degrees_of_freedom"], relative_noise_shape=fit["relative_noise_shape"],
                basis=np.asarray(fit["basis"], dtype=float), ridge=fit["ridge"], noise_order=fit["noise_order"],
                conditioning_samples=fit["conditioning_samples"], source_model_path=directory/"original-source.npz")
            original_bytes = (directory/"original-source.npz").read_bytes()
            if ContinuousDecisionModel.load(directory/"original-source.npz").digest() != model.digest():
                raise ValueError("saved original source NPZ changed its actual fitted model")
            frozen = FrozenJointFlatSource.from_parameters(parameters, original_model_content=original_bytes)
            with (directory/"frozen-source.json").open("xb") as stream:
                stream.write(frozen.content)
                stream.flush()
                os.fsync(stream.fileno())
            readback = FrozenJointFlatSource.from_bytes((directory/"frozen-source.json").read_bytes(),
                expected_sha256=frozen.digest)
            if readback.original_model_content != original_bytes or count["original_continuous_fit_calls"] != 1:
                raise ValueError("source stage must retain one actual original fit and its complete codec")
            count["successful_source_fits"] = 1
            _event(directory, "source_saved", plan_sha256=plan_sha256,
                actual_successful_public_source_fits=1, actual_successful_source_fits=1,
                noise_refits_after_original_source=0, source_model=_artifact(directory/"original-source.npz"),
                frozen_parameters=_artifact(directory/"frozen-source.json"), actual_source_seed=None)
            pools = threadpool_info()
            if any(pool["num_threads"] != 1 for pool in pools):
                raise ValueError("actual native source thread pools differ from the explicit native1 plan")
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        pools = []
    source_after = _imported_sources(root)
    _json(directory/"worker_terminal.json", {"schema": "n2p3_original_source_worker_terminal/1",
        "status": "failed" if failure else "passed", "identity": identity, "plan_sha256": plan_sha256,
        "counts": count, "source_before": source_before, "source_after": source_after,
        "native_thread_pools": pools, "failure": failure})
    return 1 if failure else 0


def admit_original_continuous_source_stage(directory, *, export_directory=None):
    """Reconstruct actual original issuers and admit through the real catalogue."""
    from research.joint_source_catalog import JointFlatSourceCatalog, issue_joint_source_entry
    from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

    directory = Path(directory).resolve()
    plan_bytes = (directory/"plan.json").read_bytes()
    plan = validate_source_stage_plan(json.loads(plan_bytes))
    terminal = json.loads((directory/"source_stage_terminal.json").read_bytes())
    if terminal["status"] != "passed" or hashlib.sha256(plan_bytes).hexdigest() != terminal["plan_sha256"]:
        raise ValueError("read-only source export requires its completed observed source stage")
    destination = directory if export_directory is None else Path(export_directory).resolve()
    if export_directory is not None:
        destination.mkdir(parents=True, exist_ok=False)
    if any((destination/name).exists() for name in ("SOURCE_ENTRY.json", "source_catalogue.json", "source_resolution.json")):
        raise FileExistsError("retain the original export; read-only recovery requires a fresh export directory")
    cohort, target = _cohort(plan)
    preparation = FrozenJointFlatSourcePreparation()
    entry = issue_joint_source_entry(plan["source_id"], codec_path=directory/"frozen-source.json",
        original_model_path=directory/"original-source.npz", source_cohort=cohort, target_contract=target,
        exclusion_scope=plan["exclusion_scope"], event_log_path=directory/"events.jsonl",
        started_event_index=0, completed_event_index=1, aggregate_terminal_path=directory/"source_stage_terminal.json",
        source_evidence_scope=SCOPE, _source_preparation=preparation)
    catalog = JointFlatSourceCatalog([entry], base_directory=directory, evidence_scope=SCOPE,
        _source_preparation=preparation)
    resolved = catalog.resolve(plan["source_id"], target_contract=target, exclusion_scope=plan["exclusion_scope"])
    _json(destination/"SOURCE_ENTRY.json", entry)
    _json(destination/"source_catalogue.json", catalog.record())
    _json(destination/"source_resolution.json", resolved.receipt)
    return {"source_entry": _artifact(destination/"SOURCE_ENTRY.json"), "catalogue": _artifact(destination/"source_catalogue.json"),
        "resolution": _artifact(destination/"source_resolution.json"), "readonly_source_refits": 0,
        "readonly_target_fits": 0, "data_admission_scope": SCOPE, "public_data_admission_executed": False}


def run_original_continuous_source_stage(plan, output_directory, *, python_executable):
    """Observe one isolated scientific child and preserve source/aggregate exits."""
    import psutil

    validate_source_stage_plan(plan)
    directory = Path(output_directory).resolve()
    resources = {"cpu_logical": psutil.cpu_count(), "affinity_logical": len(psutil.Process().cpu_affinity()),
        "memory_available_bytes": psutil.virtual_memory().available, "native_threads": 1,
        "observed_unix": time.time()}
    if resources["memory_available_bytes"] < plan["minimum_available_memory_bytes"]:
        raise MemoryError("source stage waits for its declared available-memory budget")
    directory.mkdir(parents=True, exist_ok=False)
    _json(directory/"resources_before.json", resources)
    _json(directory/"plan.json", plan)
    plan_sha = _artifact(directory/"plan.json")["sha256"]
    root = Path(__file__).resolve().parents[2]
    baseline = {path.relative_to(root).as_posix(): path.read_bytes() for path in (root/"src").rglob("*.py")}
    # One code-only archive covers both worker imports and the later observer
    # catalogue imports, without an environment or duplicate EEG/model copy.
    with zipfile.ZipFile(directory/"source_code.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(baseline.items()):
            archive.writestr(name, content)
    observer_before = _imported_sources(root)
    environment = dict(os.environ)
    environment.update({name: "1" for name in THREAD_VARIABLES})
    environment.update(PYTHONPATH=str(root/"src"), PYTHONDONTWRITEBYTECODE="1")
    argv = [str(python_executable), "-m", "research.original_continuous_source_stage", "--worker",
        "--plan", str(directory/"plan.json"), "--plan-sha256", plan_sha, "--output", str(directory)]
    launcher = job = handle = None
    owned = {}
    scientific = None
    scientific_exit = None
    launcher_exit = None
    peak_rss = 0
    scientific_peak_rss = 0
    failure = None
    retirement = {"backend": "not_launched", "active_owned_processes_after_exit": 0}

    def remember():
        nonlocal peak_rss, scientific_peak_rss
        for identity in list(owned.values()):
            process = _same_process(identity)
            if process is not None:
                try:
                    members = (process, *process.children(recursive=True))
                except psutil.NoSuchProcess:
                    members = ()
                for item in members:
                    try:
                        owned[item.pid] = _identity(item)
                        rss = item.memory_info().rss
                        peak_rss = max(peak_rss, rss)
                        if scientific is not None and item.pid == scientific["pid"]:
                            scientific_peak_rss = max(scientific_peak_rss, rss)
                    except psutil.NoSuchProcess:
                        pass

    try:
        with (directory/"worker.stdout.log").open("wb") as stdout, (directory/"worker.stderr.log").open("wb") as stderr:
            launcher = subprocess.Popen(argv, cwd=root, env=environment, stdout=stdout, stderr=stderr,
                creationflags=0x4 if os.name == "nt" else 0)  # CREATE_SUSPENDED before Job assignment.
            launch_identity = _identity(psutil.Process(launcher.pid))
            owned[launcher.pid] = launch_identity
            if os.name == "nt":
                from research.cloud_execution import _WindowsJob

                job = _WindowsJob(SimpleNamespace(sentinel=int(launcher._handle)))
                psutil.Process(launcher.pid).resume()
            deadline = time.monotonic()+plan["handshake_timeout_seconds"]
            while not (directory/"worker_ready.json").exists():
                remember()
                if launcher.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("source worker exited or did not issue its live ready identity")
                time.sleep(.05)
            ready = json.loads((directory/"worker_ready.json").read_bytes())
            scientific = ready["identity"]
            worker_process = _same_process(scientific)
            if worker_process is None or ready["plan_sha256"] != plan_sha or ready["native_threads"] != 1:
                raise ValueError("worker ready identity/plan is not the live requested science process")
            scientific_cmdline = worker_process.cmdline()
            if scientific_cmdline[1:] != argv[1:]:
                raise ValueError("actual scientific child command differs from the declared stage invocation")
            lineage = _lineage(worker_process)
            stop = next((i for i, identity in enumerate(lineage) if identity == launch_identity), None)
            if stop is None or lineage[:stop+1] != ready["lineage"][:stop+1]:
                raise ValueError("scientific child ancestry differs from the observed launcher/redirector")
            handle = _ScientificExit(scientific, launcher)
            if job is not None:
                import ctypes
                from ctypes import wintypes

                inside = wintypes.BOOL()
                job.kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
                if not job.kernel.IsProcessInJob(handle.handle, job.handle, ctypes.byref(inside)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not inside and not job.kernel.AssignProcessToJobObject(job.handle, handle.handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            owned[scientific["pid"]] = scientific
            _json(directory/"observer_identity.json", {"observer": _identity(psutil.Process()),
                "launcher": launch_identity, "scientific": scientific, "verified_lineage": lineage[:stop+1],
                "scientific_executable": worker_process.exe(), "scientific_cmdline": scientific_cmdline,
                "venv_redirector_is_scientific_process": scientific == launch_identity,
                "scientific_OS_exit_owner_before_fit": "Windows_process_handle" if os.name == "nt" else "waitable_child",
                "plan_sha256": plan_sha})
            _json(directory/"observer_go.json", {"scientific_identity": scientific, "plan_sha256": plan_sha,
                "ready_sha256": _artifact(directory/"worker_ready.json")["sha256"]})
            while (scientific_exit := handle.poll()) is None:
                remember()
                time.sleep(.05)
            launcher_exit = launcher.wait(timeout=plan["handshake_timeout_seconds"])
            remember()
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        try:
            if job is not None:
                retirement = job.retire()
            elif launcher is not None:
                _retire_uncontained(owned, remember, launcher.pid)
                retirement = {"backend": "suspended_verified_owned_process_births"}
            if launcher is not None:
                launcher_exit = launcher.wait(timeout=10)
                if job is None:
                    retirement["active_owned_processes_after_exit"] = sum(
                        _same_process(identity) is not None for identity in owned.values())
            if handle is not None and scientific_exit is None:
                scientific_exit = handle.poll()
        except BaseException as error:
            retirement = {"backend": "retirement_failed", "active_owned_processes_after_exit": None,
                "failure": {"type": type(error).__name__, "message": str(error)}}
            failure = failure or {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        finally:
            if handle is not None:
                handle.close()
            if job is not None and job.handle is not None:
                job.kernel.CloseHandle(job.handle)
                job.handle = None
    worker_path = directory/"worker_terminal.json"
    worker = json.loads(worker_path.read_bytes()) if worker_path.exists() else {}
    imported = set(worker.get("source_before", {})) | set(worker.get("source_after", {}))
    source_before = {name: {"sha256": hashlib.sha256(baseline[name]).hexdigest(), "bytes": len(baseline[name])}
        for name in imported if name in baseline}
    source_after = {name: ({key: value[key] for key in ("sha256", "bytes")}
        if (root/name).exists() and (value := _artifact(root/name)) else {"missing": True}) for name in imported}
    unchanged = bool(imported) and len(source_before) == len(imported) and source_before == source_after
    unchanged = unchanged and all(value["sha256"] == source_before[name]["sha256"]
        for phase in ("source_before", "source_after") for name, value in worker.get(phase, {}).items())
    passed = (failure is None and scientific_exit == 0 and worker.get("status") == "passed" and unchanged
        and retirement["active_owned_processes_after_exit"] == 0)
    count = worker.get("counts", {})
    terminal = {"schema": "n2p3_observed_original_continuous_source_stage/1", "status": "passed" if passed else "failed",
        "observed_original_fitter_pid": None if scientific is None else scientific["pid"],
        "observed_original_fitter_identity": scientific, "scientific_child_exit_code": scientific_exit,
        "launcher_exit_code": launcher_exit, "source_successful_fit_count": count.get("successful_source_fits", 0),
        "source_model_sha256": _artifact(directory/"original-source.npz")["sha256"] if (directory/"original-source.npz").exists() else None,
        "frozen_source_sha256": _artifact(directory/"frozen-source.json")["sha256"] if (directory/"frozen-source.json").exists() else None,
        "source_parameter_refits_after_original_source": 0, "source_unchanged": unchanged,
        "source_before": source_before, "source_after": source_after, "source_archive": _artifact(directory/"source_code.zip"),
        "owned_processes": list(owned.values()), "retirement": retirement,
        "scientific_peak_observed_RSS_bytes": scientific_peak_rss, "maximum_owned_process_RSS_bytes": peak_rss,
        "worker_terminal": _artifact(worker_path) if worker_path.exists() else None, "failure": failure,
        "source_evidence_scope": SCOPE, "actual_source_seed": None, "plan_sha256": plan_sha}
    _json(directory/"source_stage_terminal.json", terminal)
    aggregate = {"schema": "n2p3_original_source_stage_aggregate/1", "source_stage": _artifact(directory/"source_stage_terminal.json"),
        "source_stage_status": terminal["status"], "status": "failed", "observer_result_exit_code": 1, "admission": None}
    try:
        if not passed:
            raise RuntimeError("observed original source stage did not complete its scientific contract")
        aggregate["admission"] = admit_original_continuous_source_stage(directory)
        aggregate.update(status="passed", observer_result_exit_code=0)
    except BaseException as error:
        aggregate["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    observer_after = _imported_sources(root)
    aggregate["observer_imported_sources_before"] = observer_before
    aggregate["observer_imported_sources_after"] = observer_after
    aggregate["observer_imported_sources_match_prelaunch_archive"] = all(name in baseline
        and value["sha256"] == hashlib.sha256(baseline[name]).hexdigest()
        for phase in (observer_before, observer_after) for name, value in phase.items())
    if not aggregate["observer_imported_sources_match_prelaunch_archive"]:
        aggregate.update(status="failed", observer_result_exit_code=1,
            failure={"type": "SourceIdentityChanged", "message": "observer imports differ from prelaunch code archive"})
    _json(directory/"aggregate_terminal.json", aggregate)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--admit-only", action="store_true")
    parser.add_argument("--export-output", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.admit_only:
        if args.export_output is None:
            parser.error("--admit-only requires a fresh --export-output directory")
        result = admit_original_continuous_source_stage(args.output, export_directory=args.export_output)
        _json(args.export_output/"readonly_recovery.json", {"source_stage": _artifact(args.output/"source_stage_terminal.json"),
            "admission": result, "consumer_source_sha256": _artifact(__file__)["sha256"], "source_refits": 0})
        print(json.dumps(result))
        return 0
    if args.plan is None:
        parser.error("--plan is required for a new original source stage")
    if args.worker:
        return _worker(args.plan, args.plan_sha256, args.output)
    result = run_original_continuous_source_stage(json.loads(args.plan.read_bytes()), args.output,
        python_executable=args.python)
    print(json.dumps(result))
    return result["observer_result_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
