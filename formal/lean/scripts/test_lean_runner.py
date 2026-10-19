"""Behavioral scheduler/cache tests; no proof or toolchain is modified."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import sys
import time
from pathlib import Path

import psutil
import pytest
import run_lean
from verify import run


def _worker(root, shared, label, events, release, reserve=0):
    try:
        with run_lean.compile_slot(shared=shared, root=Path(root),
                                   minimum_available_bytes=reserve) as admission:
            events.put(("entered", label, admission))
            if not release.wait(20):
                raise TimeoutError(label)
        events.put(("finished", label, None))
    except BaseException as error:
        events.put(("error", label, repr(error)))
        raise


def _start(root, shared, label, events, reserve=0):
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    process = context.Process(target=_worker,
                              args=(str(root), shared, label, events, release, reserve))
    process.start()
    return process, release


def _entered(events, expected):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        event, label, payload = events.get(timeout=max(0.1, deadline - time.monotonic()))
        assert event != "error", payload
        if event == "entered":
            assert label == expected
            return payload
    raise AssertionError(f"Admission missing: {expected}")


def _not_entered(events):
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        try:
            event, label, payload = events.get(timeout=max(0.01, deadline - time.monotonic()))
        except queue.Empty:
            return
        assert event == "finished", (event, label, payload)


def _queued(root, process):
    deadline = time.monotonic() + 15
    path = root / ".lake" / "verification-scheduler" / "tickets.jsonl"
    while time.monotonic() < deadline:
        _, entries = run_lean._journal_state(path)
        if any(entry["pid"] == process.pid for entry in entries.values()):
            return
        time.sleep(0.02)
    raise AssertionError("Worker did not enqueue")


def _finish(*workers):
    for _, release in workers:
        release.set()
    for process, _ in workers:
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert process.exitcode == 0


def test_two_independent_checks_are_the_hard_cap(tmp_path):
    events = multiprocessing.get_context("spawn").Queue()
    workers = []
    try:
        workers.append(_start(tmp_path, True, "a", events))
        _entered(events, "a")
        workers.append(_start(tmp_path, True, "b", events))
        _entered(events, "b")
        workers.append(_start(tmp_path, True, "c", events))
        _queued(tmp_path, workers[2][0])
        _not_entered(events)
        workers[0][1].set()
        _entered(events, "c")
    finally:
        _finish(*workers)


def test_fifo_writer_blocks_later_checks_and_excludes_them(tmp_path):
    events = multiprocessing.get_context("spawn").Queue()
    workers = []
    try:
        workers.append(_start(tmp_path, True, "first", events))
        _entered(events, "first")
        workers.append(_start(tmp_path, False, "writer", events))
        _queued(tmp_path, workers[1][0])
        workers.append(_start(tmp_path, True, "later", events))
        _queued(tmp_path, workers[2][0])
        _not_entered(events)
        workers[0][1].set()
        _entered(events, "writer")
        _not_entered(events)
        workers[1][1].set()
        _entered(events, "later")
    finally:
        _finish(*workers)


def test_second_check_obeys_live_memory_gate(tmp_path):
    events = multiprocessing.get_context("spawn").Queue()
    workers = []
    try:
        workers.append(_start(tmp_path, True, "first", events))
        _entered(events, "first")
        workers.append(_start(tmp_path, True, "memory_waiter", events, reserve=2**63))
        _queued(tmp_path, workers[1][0])
        _not_entered(events)
        workers[0][1].set()
        _entered(events, "memory_waiter")  # A first check does not require the second-slot reserve.
    finally:
        _finish(*workers)


def test_crashed_owner_and_pid_reuse_are_recovered(tmp_path):
    events = multiprocessing.get_context("spawn").Queue()
    worker = _start(tmp_path, True, "crash", events)
    _entered(events, "crash")
    worker[0].terminate()
    worker[0].join(10)
    assert not worker[0].is_alive()
    journal = tmp_path / ".lake" / "verification-scheduler" / "tickets.jsonl"
    sequence, _ = run_lean._journal_state(journal)
    current = psutil.Process()
    run_lean._append_event(journal, {
        "event": "queued", "ticket": sequence + 1, "pid": current.pid,
        "created": current.create_time() + 60, "shared": False,
    })
    assert run_lean._owner_alive(current.pid, current.create_time())
    with run_lean.compile_slot(root=tmp_path):
        _, entries = run_lean._journal_state(journal)
        assert len(entries) == 1
    assert not run_lean._journal_state(journal)[1]
    assert journal.read_text().count("owner_exited_or_pid_reused") == 2


def test_environment_cache_tracks_pinned_inputs_and_keeps_only_lean_keys(tmp_path, monkeypatch):
    root = tmp_path / "project" / "formal" / "lean"
    root.mkdir(parents=True)
    files = [root / "lean-toolchain", root / "lakefile.toml", root / "lake-manifest.json"]
    package_file = root / ".lake" / "packages" / "demo" / "lakefile.toml"
    package_file.parent.mkdir(parents=True)
    files.append(package_file)
    lake = tmp_path / "bin" / ("lake.exe" if os.name == "nt" else "lake")
    lean = lake.with_name("lean.exe" if os.name == "nt" else "lean")
    lake.parent.mkdir()
    files += [lake, lean]
    for path in files:
        path.write_text("initial")
    calls = []
    core = tmp_path / "sys" / "lib" / "lean"
    (core / "Init").mkdir(parents=True)
    (core / "Init" / "M.olean").write_bytes(b"test")
    values = {"LEAN_PATH": str(tmp_path / "unknown"), "LEAN_SRC_PATH": "source",
              "LEAN_SYSROOT": str(tmp_path / "sys")}

    def resolver(*_):
        calls.append(True)
        return values

    monkeypatch.setattr(run_lean, "_resolve_environment", resolver)
    base = {"PATH": "original", "SECRET_TOKEN": "must-not-be-cached"}
    first = run_lean.lean_environment(lake, base, root=root)
    assert first["SECRET_TOKEN"] == base["SECRET_TOKEN"]
    assert str(core) in first["LEAN_PATH"].split(os.pathsep)
    assert run_lean.lean_environment(lake, {**base, "PATH": "new"}, root=root)["PATH"].endswith("new")
    assert len(calls) == 1
    for count, path in enumerate(files, 2):
        path.write_text("changed")
        run_lean.lean_environment(lake, base, root=root)
        assert len(calls) == count
    raw = (root / ".lake" / "verification-environment.json").read_text()
    assert "SECRET_TOKEN" not in raw and base["SECRET_TOKEN"] not in raw
    assert set(json.loads(raw)["environment"]) == set(run_lean.LEAN_VARIABLES)


def test_path_ordering_revalidates_namespaces_and_preserves_unknown_barriers(tmp_path):
    a, b, missing = tmp_path / "a", tmp_path / "b", tmp_path / "not-built"
    for root, namespace, count in [(a, "Alpha", 1), (b, "Beta", 3)]:
        (root / namespace).mkdir(parents=True)
        for index in range(count):
            (root / namespace / f"M{index}.olean").write_bytes(b"test")
    original = os.pathsep.join(map(str, [missing, a, b]))
    assert run_lean._ordered_lean_path(original, tmp_path) == os.pathsep.join(map(str, [missing, b, a]))
    barrier = os.pathsep.join(map(str, [a, missing, b]))
    assert run_lean._ordered_lean_path(barrier, tmp_path) == barrier
    (missing / "Beta").mkdir(parents=True)
    assert run_lean._ordered_lean_path(original, tmp_path) == original


def _command_tree_worker(root, pid_file, marker):
    grandchild = (
        "import time\nfrom pathlib import Path\n"
        f"p=Path({marker!r})\n"
        "while True:\n"
        " with p.open('a') as stream:\n"
        "  stream.write('write\\n')\n"
        " time.sleep(.02)\n"
    )
    child = (
        "import os,sys,json,time,subprocess,psutil; from pathlib import Path; "
        f"g=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        f"Path({pid_file!r}).write_text(json.dumps([[p,psutil.Process(p).create_time()] "
        "for p in [os.getpid(),g.pid]])); time.sleep(30)"
    )
    with run_lean.compile_slot(shared=True, root=Path(root), minimum_available_bytes=0):
        run([sys.executable, "-c", child], Path(root), "managed_child", dict(os.environ))


@pytest.mark.skipif(os.name != "nt", reason="Controller-death JobObject binding is Windows-specific")
def test_real_child_tree_dies_before_exclusive_writes_can_overlap(tmp_path):
    context = multiprocessing.get_context("spawn")
    pid_file, marker = tmp_path / "children.json", tmp_path / "child_writes.txt"
    controller = context.Process(target=_command_tree_worker,
                                 args=(str(tmp_path), str(pid_file), str(marker)))
    controller.start()
    try:
        deadline = time.monotonic() + 15
        while (not pid_file.exists() or not marker.exists()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.exists() and marker.exists()
        identities = json.loads(pid_file.read_text())
        controller.terminate()
        controller.join(10)
        assert not controller.is_alive()
        with run_lean.compile_slot(root=tmp_path):
            for pid, created in identities:
                if run_lean._owner_alive(pid, created):
                    psutil.Process(pid).wait(timeout=5)
                assert not run_lean._owner_alive(pid, created)
            before = marker.read_bytes()
            time.sleep(0.3)
            assert marker.read_bytes() == before
            (tmp_path / "exclusive_writer.txt").write_text("no child overlapped this writer")
    finally:
        if controller.is_alive():
            controller.terminate()
            controller.join(5)


def test_admission_revalidation_uses_raw_paths_and_rejects_configuration_drift(tmp_path, monkeypatch):
    root = tmp_path / "formal" / "lean"
    root.mkdir(parents=True)
    lake = tmp_path / ("lake.exe" if os.name == "nt" else "lake")
    lake.write_text("lake")
    lake.with_name("lean.exe" if os.name == "nt" else "lean").write_text("lean")
    a, b = tmp_path / "a", tmp_path / "b"
    for path, name, count in [(a, "Alpha", 1), (b, "Beta", 3)]:
        (path / name).mkdir(parents=True)
        for index in range(count):
            (path / name / f"M{index}.olean").write_bytes(b"test")
    raw = os.pathsep.join(map(str, [a, b]))
    calls = []

    def resolver(*_):
        calls.append(True)
        return {"LEAN_PATH": raw, "LEAN_SRC_PATH": None, "LEAN_SYSROOT": None}

    monkeypatch.setattr(run_lean, "_resolve_environment", resolver)
    env = run_lean.lean_environment(lake, {}, root=root)
    assert env["LEAN_PATH"] == os.pathsep.join(map(str, [b, a]))
    with run_lean.compile_slot(root=root):  # Writer runs after environment preparation.
        (a / "Beta").mkdir()
    with run_lean.compile_slot(shared=True, root=root):
        assert run_lean.revalidate_environment(env)["LEAN_PATH"] == raw
    (root / "lakefile.toml").write_text("changed while queued")
    with run_lean.compile_slot(shared=True, root=root):
        with pytest.raises(run_lean.EnvironmentChangedError):
            run_lean.revalidate_environment(env)
    assert len(calls) == 1  # Revalidation neither invokes Lake nor recursively requests a slot.
    run_lean.lean_environment(lake, {}, root=root)
    assert len(calls) == 2
