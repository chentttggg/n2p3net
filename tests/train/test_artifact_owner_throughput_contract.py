"""Strict named-output admission remains live across independent output trees."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace

import pytest

from research import source_execution as source
from research.source_execution import SourceInputAdmission


def _owner(monkeypatch, capacity=0):
    monkeypatch.setattr(source, "filesystem_uses_host_memory", lambda path: capacity > 0)
    return SourceInputAdmission(SimpleNamespace(SourceInputState=source._SourceInputState),
                                capacity_bytes=capacity, load_workers=4)


def _named(state, path, nbytes, identity, *, directory=False, temporary_paths=()):
    return state.reserve_artifact(str(path), [str(path), *map(str, temporary_paths)], directory,
                                  str(path.parent.stat().st_dev), nbytes, 0, False, identity)


@contextmanager
def _blocked_scan(monkeypatch, path):
    original = source._named_output_bytes
    entered, release = Event(), Event()

    def scan(item, directory=False):
        if Path(item) == path:
            entered.set()
            assert release.wait(5), "owner scan was not released by its counterexample"
        return original(item, directory)

    monkeypatch.setattr(source, "_named_output_bytes", scan)
    try:
        yield entered, release
    finally:
        release.set()


def _pool_matches_leases(state):
    record = state.record()
    assert sum(pool["reserved_bytes"] for pool in record["filesystem_pools"].values()) == sum(
        row.get("disk_bytes", 0) for row in record["active_leases"])


def test_scanning_one_output_does_not_block_independent_named_writer_or_phase(tmp_path, monkeypatch):
    owner = _owner(monkeypatch, capacity=16384)
    original = source._named_output_bytes
    slow_root, peer_root = tmp_path / "slow", tmp_path / "peer"
    scan_entered, release_scan, peer_finished = Event(), Event(), Event()
    enabled = False

    def slow_scan(path, directory=False):
        if enabled and Path(path) == slow_root and directory:
            scan_entered.set()
            if not release_scan.wait(10):
                raise AssertionError("counterexample did not release its exact filesystem scan")
        return original(path, directory)

    monkeypatch.setattr(source, "_named_output_bytes", slow_scan)

    def independent_work():
        with owner.reserve("load", 1024):
            with owner.artifact_directory(peer_root, 4096):
                owner.write_json(peer_root / "completed.json", {"peer": "completed while the other scan was pending"})
        peer_finished.set()

    with owner.artifact_directory(slow_root, 4096), ThreadPoolExecutor(max_workers=2) as pool:
        slow_root.mkdir()
        enabled = True
        writer = pool.submit(owner.write_json, slow_root / "first.json", {"same": "strict admission"})
        assert scan_entered.wait(3)
        peer = pool.submit(independent_work)
        try:
            assert peer_finished.wait(2), "unrelated output/staging transaction blocked behind filesystem scan"
        finally:
            release_scan.set()
            writer.result(timeout=5)
            peer.result(timeout=5)
    record = owner.record()
    assert record["load_active"] == 0 and not record["pending_nested_artifacts"]
    assert all(row["committed"] for row in record["active_leases"])


@pytest.mark.parametrize("preexisting", [False, True])
def test_unmanaged_file_or_existing_file_growth_is_seen_before_next_nested_write(tmp_path, monkeypatch, preexisting):
    owner = _owner(monkeypatch, capacity=100)
    root = tmp_path / "output"
    root.mkdir()
    unmanaged = root / "unmanaged.bin"
    if preexisting:
        unmanaged.write_bytes(b"old" * 10)
    with pytest.raises(ValueError, match="remaining shared filesystem/RAM budget"):
        with owner.artifact_directory(root, 10):
            unmanaged.write_bytes(b"x" * (120 if preexisting else 90))
            with owner.artifact_writer(root / "must_not_open.bin", 20):
                pytest.fail("unmanaged surviving growth was silently ignored")
    assert not (root / "must_not_open.bin").exists()
    row = owner.record()["active_leases"][0]
    assert row["committed"] and row["disk_bytes"] == 90
    assert owner.record()["free_staging_bytes"] == 10


def test_strict_scan_preserves_hardlinks_file_symlinks_and_excludes_symlink_directories(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    child = root / "nested"
    child.mkdir()
    original = child / "data.bin"
    original.write_bytes(b"bounded")
    os.link(original, root / "alias.bin")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "not_covered.bin").write_bytes(b"outside" * 100)
    (root / "file-link").symlink_to(original)
    (root / "directory-link").symlink_to(outside, target_is_directory=True)
    (root / "dangling-link").symlink_to(tmp_path / "absent")
    expected = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert expected == 3 * len(b"bounded")
    assert source._named_output_bytes(root, True) == expected


def test_retirement_marks_task_before_waiting_for_its_output_scan_and_rejects_late_admission(tmp_path, monkeypatch):
    state = source._SourceInputState(1024, 4)
    root = tmp_path / "retiring"
    identity = {"pid": os.getpid(), "task_id": "retiring"}
    key = str(tmp_path.stat().st_dev)
    outer = state.reserve_artifact(str(root), [str(root)], True, key, 100, 0, False, identity)
    root.mkdir()
    (root / "unmanaged.bin").write_bytes(b"surviving")
    original = source._named_output_bytes
    scan_entered, release_scan, retired_mark_seen = Event(), Event(), Event()

    def slow_scan(path, directory=False):
        if Path(path) == root and directory and not release_scan.is_set():
            scan_entered.set()
            if not release_scan.wait(10):
                raise AssertionError("counterexample did not release retirement scan")
        return original(path, directory)

    monkeypatch.setattr(source, "_named_output_bytes", slow_scan)
    class ObservedRetirements(set):
        def add(self, value):
            super().add(value)
            retired_mark_seen.set()

    state.retired_tasks = ObservedRetirements()
    with ThreadPoolExecutor(max_workers=3) as pool:
        admission = pool.submit(state.reserve_artifact, str(root / "late.bin"), [str(root / "late.bin")],
                                False, key, 32, 0, False, identity)
        assert scan_entered.wait(3)
        retirement = pool.submit(state.retire_task, "retiring")

        def probe():
            # This must reject while retirement is awaiting the per-output scan;
            # returning False or allocating late would reopen the retired task.
            with pytest.raises(RuntimeError, match="retired"):
                state.try_resident_input(0, "after-retirement", identity)

        try:
            assert retired_mark_seen.wait(2), "retirement could not close allocation authority before scan completion"
            pool.submit(probe).result(timeout=2)
        finally:
            release_scan.set()
            with pytest.raises(RuntimeError, match="retired"):
                admission.result(timeout=5)
            retirement.result(timeout=5)
    row = state.leases[outer["token"]]
    assert row["committed"] and row["disk_bytes"] == len(b"surviving")
    assert not state.nested_artifacts


def test_retirement_cannot_settle_a_peer_retry_of_a_completed_owned_output(tmp_path, monkeypatch):
    state = source._SourceInputState(0, 4)
    first, shared = tmp_path / "first", tmp_path / "shared"
    a, b = {"pid": 100, "task_id": "A"}, {"pid": 200, "task_id": "B"}
    _named(state, first, 100, a, directory=True)
    original = _named(state, shared, 100, a, directory=True)
    first.mkdir()
    shared.mkdir()
    payload = shared / "data.bin"
    payload.write_bytes(b"x" * 10)
    state.settle_artifact(original["token"])
    peer_ready, finish_peer = Event(), Event()

    def retry():
        lease = _named(state, shared, 100, b, directory=True)
        peer_ready.set()
        assert finish_peer.wait(5)
        payload.write_bytes(b"y" * 100)
        try:
            state.settle_artifact(lease["token"])
        except ValueError as error:
            return str(error)
        return None

    with _blocked_scan(monkeypatch, first) as (entered, release), ThreadPoolExecutor(2) as pool:
        retirement = pool.submit(state.retire_task, "A")
        assert entered.wait(3)
        peer = pool.submit(retry)
        try:
            assert peer_ready.wait(2), "independent retry blocked behind an unrelated retirement scan"
            release.set()
            retirement.result(timeout=3)
            row = dict(state.leases[original["token"]])
        finally:
            release.set()
            finish_peer.set()
        settlement_error = peer.result(timeout=3)
    assert row["task_id"] == "B" and not row["committed"]
    assert row["disk_bytes"] == 110
    assert settlement_error is None
    _pool_matches_leases(state)


def test_retirement_rechecks_the_actual_lease_after_waiting_for_output_scope(tmp_path, monkeypatch):
    state = source._SourceInputState(0, 4)
    shared = tmp_path / "shared"
    a, b = {"pid": 100, "task_id": "A"}, {"pid": 200, "task_id": "B"}
    original = _named(state, shared, 100, a, directory=True)
    shared.mkdir()
    payload = shared / "data.bin"
    payload.write_bytes(b"x" * 10)
    guard_reached, release_retirement = Event(), Event()
    original_guard = state._artifact_guard
    retiring_thread = None

    @contextmanager
    def delayed_guard(path, *temporary_paths):
        if get_ident() == retiring_thread:
            guard_reached.set()
            assert release_retirement.wait(5)
        with original_guard(path, *temporary_paths) as scope:
            yield scope

    def retire():
        nonlocal retiring_thread
        retiring_thread = get_ident()
        state.retire_task("A")

    monkeypatch.setattr(state, "_artifact_guard", delayed_guard)
    with ThreadPoolExecutor(1) as pool:
        retirement = pool.submit(retire)
        try:
            assert guard_reached.wait(3)
            state.settle_artifact(original["token"])
            peer = _named(state, shared, 100, b, directory=True)
        finally:
            release_retirement.set()
        retirement.result(timeout=3)
    row = state.leases[peer["token"]]
    assert row["task_id"] == "B" and not row["committed"]
    assert row["disk_bytes"] == 110
    payload.write_bytes(b"y" * 100)
    state.settle_artifact(peer["token"])
    _pool_matches_leases(state)


def test_ancestor_registration_cannot_admit_two_writers_for_the_same_path(tmp_path, monkeypatch):
    state = source._SourceInputState(0, 4)
    root = tmp_path / "output"
    root.mkdir()
    target, temporary = root / "shared.bin", root / ".a.tmp"
    initial, a, b, parent = ({"pid": index, "task_id": name} for index, name in enumerate(
        ("initial", "A", "B", "parent"), start=100))
    lease = _named(state, target, 10, initial)
    target.write_bytes(b"x" * 10)
    state.settle_artifact(lease["token"])
    controller_started, controller_done = Event(), Event()

    def parent_and_peer():
        controller_started.set()
        covering = _named(state, root, 100, parent, directory=True)
        try:
            try:
                _named(state, target, 80, b)
            except ValueError as error:
                assert "already owns" in str(error)
                return "rejected duplicate writer"
            return "admitted duplicate writer"
        finally:
            state.settle_artifact(covering["token"])
            controller_done.set()

    with _blocked_scan(monkeypatch, temporary) as (entered, release), ThreadPoolExecutor(2) as pool:
        writer = pool.submit(_named, state, target, 100, a, temporary_paths=(temporary,))
        assert entered.wait(3)
        controller = pool.submit(parent_and_peer)
        try:
            assert controller_started.wait(2)
            controller_done.wait(.3)
        finally:
            release.set()
        writer.result(timeout=3)
        outcome = controller.result(timeout=3)
    assert outcome == "rejected duplicate writer"
    assert state.leases[lease["token"]]["task_id"] == "A"
    _pool_matches_leases(state)


def test_ancestor_registration_cannot_hide_a_concurrent_pending_nested_peak(tmp_path, monkeypatch):
    state = source._SourceInputState(0, 4)
    parent, root = tmp_path / "parent", tmp_path / "parent" / "outer"
    root.mkdir(parents=True)
    a, b = root / "a.bin", root / "b.bin"
    identity, parent_identity = {"pid": 100, "task_id": "writers"}, {"pid": 200, "task_id": "parent"}
    outer = _named(state, root, 100, identity, directory=True)
    controller_started, controller_done = Event(), Event()

    def parent_and_peer():
        controller_started.set()
        covering = _named(state, parent, 100, parent_identity, directory=True)
        try:
            return _named(state, b, 80, identity)
        finally:
            state.settle_artifact(covering["token"])
            controller_done.set()

    with _blocked_scan(monkeypatch, a) as (entered, release), ThreadPoolExecutor(2) as pool:
        writer = pool.submit(_named, state, a, 80, identity)
        assert entered.wait(3)
        controller = pool.submit(parent_and_peer)
        try:
            assert controller_started.wait(2)
            controller_done.wait(.3)
        finally:
            release.set()
        left, right = writer.result(timeout=3), controller.result(timeout=3)
    assert not left["owns_transaction"] and not right["owns_transaction"]
    assert sum(row["disk_bytes"] for row in state.nested_artifacts.values()) == 160
    assert state.leases[outer["token"]]["disk_bytes"] == 160
    _pool_matches_leases(state)


def test_atomic_temporary_scope_blocks_an_overlapping_directory_scan(tmp_path, monkeypatch):
    state = source._SourceInputState(0, 4)
    target = tmp_path / "target.bin"
    temporary_root = tmp_path / "temporary"
    temporary_root.mkdir()
    temporary = temporary_root / ".atomic.tmp"
    identity = {"pid": 100, "task_id": "atomic"}
    peer_identity = {"pid": 200, "task_id": "peer"}
    original_scan = source._named_output_bytes
    scan_entered, release_scan, peer_started, peer_scan = (Event() for _ in range(4))

    def scan(path, directory=False):
        if Path(path) == temporary:
            scan_entered.set()
            assert release_scan.wait(5)
        if Path(path) == temporary_root:
            peer_scan.set()
        return original_scan(path, directory)

    def peer():
        peer_started.set()
        return _named(state, temporary_root, 100, peer_identity, directory=True)

    monkeypatch.setattr(source, "_named_output_bytes", scan)
    with ThreadPoolExecutor(2) as pool:
        writer = pool.submit(_named, state, target, 100, identity, temporary_paths=(temporary,))
        assert scan_entered.wait(3)
        other = pool.submit(peer)
        try:
            assert peer_started.wait(2)
            overlapped = peer_scan.wait(.3)
        finally:
            release_scan.set()
        writer.result(timeout=3)
        other.result(timeout=3)
    assert not overlapped, "directory scan overlapped the atomic writer's registered temporary scope"
    _pool_matches_leases(state)


def test_four_independent_named_writers_can_enter_their_filesystem_scans_together(tmp_path, monkeypatch):
    state = source._SourceInputState(0, 4)
    roots = {tmp_path / f"writer_{index}" for index in range(4)}
    original_scan = source._named_output_bytes
    barrier, seen, seen_lock = Barrier(4), set(), Lock()

    def scan(path, directory=False):
        path = Path(path)
        with seen_lock:
            first = directory and path in roots and path not in seen
            if first:
                seen.add(path)
        if first:
            barrier.wait(timeout=3)
        return original_scan(path, directory)

    def writer(root):
        identity = {"pid": 100, "task_id": root.name}
        lease = _named(state, root, 100, identity, directory=True)
        root.mkdir()
        (root / "data.bin").write_bytes(b"actual")
        state.settle_artifact(lease["token"])

    monkeypatch.setattr(source, "_named_output_bytes", scan)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(writer, roots))
    assert seen == roots
    assert all(row["committed"] and row["disk_bytes"] == 6 for row in state.leases.values())
    _pool_matches_leases(state)


def test_output_inspection_reenters_held_scope_and_wakes_peer_after_failure(tmp_path):
    state = source._SourceInputState(0, 4)
    root, nested = tmp_path / "output", tmp_path / "output" / "nested"
    peer_started, peer_entered = Event(), Event()

    def peer():
        peer_started.set()
        with state._artifact_guard(root):
            peer_entered.set()

    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(OSError, match="filesystem probe failed"), state._artifact_guard(root):
            with state._artifact_guard(nested):
                with pytest.raises(RuntimeError, match="cannot expand"):
                    with state._artifact_guard(tmp_path):
                        pytest.fail("reentrant scope expansion must fail before waiting")
            future = pool.submit(peer)
            assert peer_started.wait(2)
            assert not peer_entered.wait(.1)
            raise OSError("filesystem probe failed")
        future.result(timeout=3)
    assert peer_entered.is_set()
