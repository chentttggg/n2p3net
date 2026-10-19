"""Resolve pinned Lean once and admit two independent checks or one exclusive job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import psutil
from lean_process import managed_process
from verify import ROOT, run

LEAN_VARIABLES = ("LEAN_PATH", "LEAN_SRC_PATH", "LEAN_SYSROOT")
SECOND_SLOT_MIN_AVAILABLE = 1280 * 1024**2


class EnvironmentChangedError(RuntimeError):
    """Resolve again outside the admission gate before running a changed project."""


class LeanEnvironment(dict):
    """A normal subprocess environment plus in-memory provenance for revalidation."""

    def __init__(self, values, *, root, lake, fingerprint, raw, inputs):
        super().__init__(values)
        self.root, self.lake = root, lake
        self.fingerprint, self.raw, self.inputs = fingerprint, dict(raw), dict(inputs)


class _FileLock:
    """OS byte/file locks release on process exit, including an abnormal exit."""

    def __init__(self, path: Path, *, shared: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.shared = shared
        self.held = False
        self.overlapped = None

    def acquire(self, *, blocking: bool = True) -> bool:
        while True:
            if os.name == "nt":
                import ctypes
                import msvcrt
                from ctypes import wintypes

                class Overlapped(ctypes.Structure):
                    _fields_ = [
                        ("Internal", ctypes.c_size_t),
                        ("InternalHigh", ctypes.c_size_t),
                        ("Offset", wintypes.DWORD),
                        ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", wintypes.HANDLE),
                    ]

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                lock = kernel.LockFileEx
                lock.argtypes = [
                    wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                    wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                ]
                lock.restype = wintypes.BOOL
                self.overlapped = Overlapped()
                flags = 1 | (0 if self.shared else 2)  # FAIL_IMMEDIATELY / EXCLUSIVE
                acquired = bool(lock(msvcrt.get_osfhandle(self.stream.fileno()), flags, 0,
                                     1, 0, ctypes.byref(self.overlapped)))
                if not acquired and ctypes.get_last_error() not in (32, 33):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                import fcntl

                flags = (fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX) | fcntl.LOCK_NB
                try:
                    fcntl.flock(self.stream.fileno(), flags)
                    acquired = True
                except BlockingIOError:
                    acquired = False
            if acquired:
                self.held = True
                return True
            if not blocking:
                return False
            time.sleep(0.05)

    def close(self) -> None:
        if self.held:
            if os.name == "nt":
                import ctypes
                import msvcrt
                from ctypes import wintypes

                unlock = ctypes.WinDLL("kernel32", use_last_error=True).UnlockFileEx
                unlock.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.DWORD, ctypes.c_void_p]
                unlock.restype = wintypes.BOOL
                if not unlock(msvcrt.get_osfhandle(self.stream.fileno()), 0, 1, 0,
                              ctypes.byref(self.overlapped)):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.held = False
        self.stream.close()


@contextmanager
def _locked(path: Path, *, shared: bool = False):
    lock = _FileLock(path, shared=shared)
    try:
        lock.acquire()
        yield
    finally:
        lock.close()


def _journal_state(path: Path) -> tuple[int, dict[int, dict]]:
    sequence, entries = 0, {}
    if not path.exists():
        return sequence, entries
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
            ticket = int(event["ticket"])
        except (ValueError, TypeError, KeyError):
            # A process can die during its last append. OS locks still bound capacity.
            continue
        sequence = max(sequence, ticket)
        if event.get("event") == "queued":
            entries[ticket] = {**event, "status": "waiting"}
        elif event.get("event") == "entered" and ticket in entries:
            entries[ticket]["status"] = "active"
        elif event.get("event") == "finished":
            entries.pop(ticket, None)
    return sequence, entries


def _append_event(path: Path, event: dict) -> None:
    """Keep tickets as a durable journal; recovery never deletes arbitrary files."""
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end:
            stream.seek(end - 1)
            needs_separator = stream.read(1) != b"\n"
            stream.seek(0, os.SEEK_END)
            if needs_separator:
                stream.write(b"\n")
        stream.write((json.dumps(event, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _owner_alive(pid: int, created: float) -> bool:
    try:
        process = psutil.Process(pid)
        return (abs(process.create_time() - created) < 0.001
                and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # Never retire a live owner merely because inspection is denied.


@contextmanager
def compile_slot(*, shared: bool = False, root: Path | None = None,
                 minimum_available_bytes: int = SECOND_SLOT_MIN_AVAILABLE):
    """FIFO admission: up to two independent checks, or one exclusive build/cache.

    The second check requires live available memory above 1.25 GiB by default;
    the observed extra private commitment was below 0.3 GiB in the paired probe.
    A queued writer prevents later readers from overtaking it. The legacy lock
    is held shared/exclusive throughout execution, so existing exclusive users
    still exclude these checks. Resolve environments before entering this gate.
    """
    root = (ROOT if root is None else root).resolve()
    state_dir = root / ".lake" / "verification-scheduler"
    state_dir.mkdir(parents=True, exist_ok=True)
    coordinator = state_dir / "coordinator.lock"
    journal = state_dir / "tickets.jsonl"
    process = psutil.Process()
    owner = {"pid": process.pid, "created": process.create_time(), "shared": shared}
    legacy = _FileLock(root / ".lake" / "verification-compile.lock", shared=shared)
    worker = None
    ticket = None
    started = time.perf_counter()
    try:
        with _locked(coordinator):
            sequence, _ = _journal_state(journal)
            ticket = sequence + 1
            _append_event(journal, {"event": "queued", "ticket": ticket, **owner})
        while True:
            with _locked(coordinator):
                _, entries = _journal_state(journal)
                for number, entry in list(entries.items()):
                    if not _owner_alive(entry["pid"], entry["created"]):
                        _append_event(journal, {"event": "finished", "ticket": number,
                                                "reason": "owner_exited_or_pid_reused"})
                        entries.pop(number)
                if ticket not in entries:
                    raise RuntimeError("Lean admission ticket disappeared")
                waiting = sorted(k for k, v in entries.items() if v["status"] == "waiting")
                active = [v for v in entries.values() if v["status"] == "active"]
                available = psutil.virtual_memory().available
                allowed = waiting[0] == ticket and (
                    (not active) or (shared and len(active) < 2
                                    and all(v["shared"] for v in active)
                                    and available >= minimum_available_bytes)
                )
                if allowed and legacy.acquire(blocking=False):
                    if shared:
                        for slot in range(2):
                            candidate = _FileLock(state_dir / f"check-{slot}.lock")
                            if candidate.acquire(blocking=False):
                                worker = candidate
                                break
                            candidate.close()
                        if worker is None:
                            legacy.close()
                            legacy = _FileLock(root / ".lake" / "verification-compile.lock",
                                               shared=shared)
                    if not shared or worker is not None:
                        _append_event(journal, {"event": "entered", "ticket": ticket,
                                                "at_unix_seconds": time.time()})
                        break
            time.sleep(0.1)
        admission = {"ticket": ticket, "shared": shared,
                     "available_ram_at_admission_bytes": available,
                     "queue_seconds": round(time.perf_counter() - started, 4),
                     "admitted_at_unix_seconds": time.time()}
        try:
            yield admission
        finally:
            admission["released_at_unix_seconds"] = time.time()
    finally:
        # Hold the coordinator through release so the next waiter sees one transition.
        with _locked(coordinator):
            if worker is not None:
                worker.close()
            legacy.close()
            if ticket is not None:
                _append_event(journal, {"event": "finished", "ticket": ticket})


def _environment_fingerprint(lake: Path, root: Path, env: dict) -> str:
    lean = lake.with_name("lean.exe" if os.name == "nt" else "lean")
    paths = [root / "lean-toolchain", root / "lakefile.toml", root / "lakefile.lean",
             root / "lake-manifest.json", lake, lean]
    packages = root / ".lake" / "packages"
    if packages.is_dir():
        for package in sorted(packages.iterdir()):
            if package.is_dir():
                paths.extend(package / name for name in
                             ("lakefile.toml", "lakefile.lean", "lake-manifest.json"))
    identity = {}
    for path in paths:
        identity[str(path.resolve())] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    identity["input_lean_variables"] = {key: env.get(key) for key in LEAN_VARIABLES}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _resolve_environment(lake: Path, root: Path, env: dict) -> dict:
    python = root.parents[1] / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError(f"Project virtual environment Python is missing: {python}")
    script = ("import os,json;print(json.dumps({k:os.environ.get(k) for k in "
              "['LEAN_PATH','LEAN_SRC_PATH','LEAN_SYSROOT']}))")
    command = [str(lake), "env", str(python), "-c", script]
    with managed_process(command, cwd=root, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT) as process:
        output, _ = process.communicate()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output=output)
    result = output.decode("utf-8")
    values = json.loads(result)
    if set(values) != set(LEAN_VARIABLES) or not all(
        value is None or isinstance(value, str) for value in values.values()
    ):
        raise ValueError("Lake returned an invalid Lean environment")
    return values


def _ordered_lean_path(value: str, root: Path) -> str:
    """Reorder only namespace-disjoint existing roots; unknown roots are barriers."""
    paths = value.split(os.pathsep)
    counts, owners = {}, {}
    for index, raw in enumerate(paths):
        path = Path(raw)
        path = path if path.is_absolute() else root / path
        if not raw or not path.is_dir():
            continue
        try:
            names = {child.name.casefold() if child.is_dir() else child.stem.casefold()
                     for child in path.iterdir()
                     if child.is_dir() or child.suffix.casefold() == ".olean"}
            for name in names:
                if name in owners:
                    return value
                owners[name] = index
            counts[index] = sum(1 for _ in path.rglob("*.olean"))
        except OSError:
            return value
    reordered = list(paths)
    start = 0
    while start < len(paths):
        if start not in counts:
            start += 1
            continue
        stop = start
        while stop < len(paths) and stop in counts:
            stop += 1
        indices = sorted(range(start, stop), key=lambda i: -counts[i])
        reordered[start:stop] = [paths[i] for i in indices]
        start = stop
    return os.pathsep.join(reordered)


def _include_core_path(value: str, sysroot: str | None, lake: Path, root: Path) -> str:
    """Expose Lean's implicit final core root to the same disjointness check."""
    core = Path(sysroot) / "lib" / "lean" if sysroot else lake.parent.parent / "lib" / "lean"
    core = (core if core.is_absolute() else root / core).resolve()
    if not core.is_dir():
        return value
    for raw in value.split(os.pathsep):
        if raw:
            path = Path(raw)
            if (path if path.is_absolute() else root / path).resolve() == core:
                return value
    return value + (os.pathsep if value else "") + str(core)


def lean_environment(lake: Path, env: dict | None = None, *, root: Path | None = None) -> dict:
    """Return a fresh environment; persist only the three Lake-produced Lean paths.

    Cache identity includes project/toolchain files, both pinned executables and
    the three input Lean variables. Path namespaces are checked live on every
    use so a newly built namespace cannot make a cached ordering change imports.
    Call outside compile_slot: cache misses use an exclusive FIFO slot.
    """
    root = (ROOT if root is None else root).resolve()
    lake = lake.resolve()
    result = dict(os.environ if env is None else env)
    result["PATH"] = str(lake.parent) + os.pathsep + result.get("PATH", "")
    cache = root / ".lake" / "verification-environment.json"

    def read_cache(fingerprint):
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            values = cached["environment"]
            if (cached["fingerprint"] == fingerprint and set(values) == set(LEAN_VARIABLES)
                    and all(v is None or isinstance(v, str) for v in values.values())):
                return values
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    fingerprint = _environment_fingerprint(lake, root, result)
    values = read_cache(fingerprint)
    if values is None:
        with compile_slot(root=root):
            fingerprint = _environment_fingerprint(lake, root, result)
            values = read_cache(fingerprint)
            if values is None:
                values = _resolve_environment(lake, root, result)
                cache.write_text(json.dumps({"fingerprint": fingerprint, "environment": values},
                                            indent=2) + "\n", encoding="utf-8")
    prepared = LeanEnvironment(result, root=root, lake=lake, fingerprint=fingerprint,
                               raw=values, inputs=result)
    prepared.update(revalidate_environment(prepared))
    return prepared


def revalidate_environment(env: LeanEnvironment) -> dict:
    """Pure filesystem recheck after admission; never invokes Lake or another gate.

    Preserve the LeanEnvironment object returned by lean_environment until this
    call. A queued writer may add namespaces: always re-sort the original raw
    roots, not a previously sorted path. Configuration drift fails closed; the
    caller can leave its slot, resolve again, then requeue.
    """
    if not isinstance(env, LeanEnvironment):
        raise TypeError("Use the environment returned by lean_environment")
    if _environment_fingerprint(env.lake, env.root, env.inputs) != env.fingerprint:
        raise EnvironmentChangedError("Lean configuration changed while queued; resolve outside the slot")
    result = dict(env)
    values, root, lake = env.raw, env.root, env.lake
    for key, value in values.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    if "LEAN_PATH" in result:
        effective = _include_core_path(result["LEAN_PATH"], result.get("LEAN_SYSROOT"), lake, root)
        result["LEAN_PATH"] = _ordered_lean_path(effective, root)
    return result



def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lake", type=Path, default=Path("D:/BCI/lean-tools/lean-4.24.0-windows/bin/lake.exe")
    )
    parser.add_argument("--file", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--cache-module", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sum([bool(args.file), bool(args.target), bool(args.cache_module)]) != 1:
        parser.error("supply one of --file, --target, or --cache-module")
    env = dict(os.environ)
    env["PATH"] = str(args.lake.resolve().parent) + os.pathsep + env.get("PATH", "")
    env["LEAN_NUM_THREADS"] = "1"
    env["MATHLIB_CACHE_DIR"] = "D:/BCI/lean-tools/mathlib-cache"
    if args.file:
        env = lean_environment(args.lake, env)
        command = [str(args.lake.resolve().with_name("lean.exe" if os.name == "nt" else "lean")),
                   "-j1", args.file]
    elif args.target:
        command = [str(args.lake.resolve()), "build", args.target]
    else:
        command = [str(args.lake.resolve()), "exe", "cache", "get", *args.cache_module]
    args.output.mkdir(parents=True, exist_ok=False)
    queued = time.perf_counter()
    with compile_slot(shared=bool(args.file)) as admission:
        queue_seconds = time.perf_counter() - queued
        if args.file:
            env = revalidate_environment(env)
        result = run(command, args.output, "lean", env)
    result["compile_queue_seconds"] = round(queue_seconds, 4)
    result["admission"] = admission
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    if result["exit_code"]:
        print((args.output / "lean.log").read_text(encoding="utf-8"), flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
