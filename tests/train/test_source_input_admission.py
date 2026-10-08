"""Byte staging preserves ready NN priority, resident transfer and failure cleanup."""
from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from research.source_execution import (
    HostMemoryAdmission,
    PreparedSourceThreadMemory,
    SourceInputAdmission,
    install_input_admission,
)


def test_four_large_cpu_fallbacks_transfer_to_retained_group_reservations():
    gib = 2**30
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=10 * gib, load_workers=1)
        groups = HostMemoryAdmission(manager, {
            "host_memory_budget_bytes": 82 * gib, "shared_staging_memory_bytes": 10 * gib,
            "source_worker_memory_bytes": 18 * gib, "target_worker_memory_bytes": 4 * gib,
            "source_workers": 4, "target_workers": 4, "gpu_slots": 4,
            "cpu_budget": 25, "coordinator_cpu_threads": 1, "native_threads_per_worker": 6})
        reservations = [groups.reserve_phase("source", cpu_threads=6, memory_bytes=18 * gib) for _ in range(4)]
        assert all(reservations)
        entered = [Event() for _ in range(4)]
        release = Event()

        def stage(index):
            # No large allocation: the actual 6GiB matrices are represented by
            # their exact lease size. Each 18GiB source group remains reserved.
            with owner.reserve("nn", 6 * gib) as lease:
                lease.release_unused()  # runtime admitted retained RSS to group
                entered[index].set()
                assert release.wait(5)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(stage, index) for index in range(4)]
            try:
                assert all(event.wait(5) for event in entered)
                assert owner.record()["free_staging_bytes"] == 10 * gib
                assert groups.record()["source_groups"] == 4
                assert groups.record()["free_worker_memory_bytes"] == 0
            finally:
                release.set()
            for future in futures:
                future.result()
        assert owner.record()["peak_staging_bytes"] == 6 * gib
        for reservation in reservations:
            groups.release_phase(reservation)
        assert groups.record()["free_worker_memory_bytes"] == 72 * gib


def test_ready_nn_priority_does_not_hold_cache_io_through_gpu_fit():
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=10, load_workers=1)
        nn_entered, io_entered, finish = Event(), Event(), Event()

        def nn():
            with owner.reserve("nn", 8):
                nn_entered.set()
                assert finish.wait(5)

        def io():
            with owner.reserve("load"):
                assert owner.record()["nn_active"] == 1
                io_entered.set()

        with owner.reserve("nn", 9) as first, ThreadPoolExecutor(max_workers=2) as pool:
            waiting = pool.submit(nn)
            # Observe the actual queue under the shared lock; no timing-only
            # assumption is used to claim that the NN was ready before I/O.
            deadline = time.monotonic() + 5
            while owner.record()["nn_waiting"] != 1 and time.monotonic() < deadline:
                time.sleep(.01)
            assert owner.record()["nn_waiting"] == 1
            loading = pool.submit(io)
            first.release_unused()
            try:
                assert nn_entered.wait(5) and io_entered.wait(5)
            finally:
                finish.set()
            waiting.result()
            loading.result()
        assert owner.record()["free_staging_bytes"] == 10


def test_input_phase_failure_and_partial_retention_release_all_bytes():
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=10, load_workers=1)
        with pytest.raises(RuntimeError):
            with owner.reserve("nn", 8) as lease:
                lease.release_unused(3)
                assert owner.record()["free_staging_bytes"] == 7
                raise RuntimeError("failure after CPU fallback")
        assert owner.record()["free_staging_bytes"] == 10
        assert owner.record()["nn_active"] == 0
        with pytest.raises(ValueError, match="capacity"):
            with owner.reserve("nn", 11):
                pytest.fail("oversized requests must fail before waiting")
        with pytest.raises(RuntimeError):
            with owner.reserve("load"):
                raise RuntimeError("cache load failed")
        assert owner.record()["load_active"] == 0
        parallel = SourceInputAdmission(manager, capacity_bytes=10, load_workers=2)
        with parallel.reserve("load"), parallel.reserve("load"):
            assert parallel.record()["load_active"] == 2
        assert parallel.record()["load_active"] == 0


def test_named_writer_retains_final_partial_bytes_and_does_not_recharge_baseline(tmp_path):
    final, temporary = tmp_path / "cache.npz", tmp_path / "cache.tmp.npz"
    final.write_bytes(b"old!")
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=0, load_workers=2)
        with owner.artifact_writer(final, 8, temporary_paths=(temporary,)):
            temporary.write_bytes(b"newstate")
            temporary.replace(final)
        with pytest.raises(RuntimeError, match="actual producer failure"):
            with owner.artifact_writer(final, 3, temporary_paths=(temporary,)):
                temporary.write_bytes(b"tmp")
                raise RuntimeError("actual producer failure")
        record = owner.record()
    lease = record["active_leases"][0]
    assert lease["baseline_bytes"] == 4
    assert lease["disk_bytes"] == 7  # final 8 + partial 3 - original 4
    assert sum(row["reserved_bytes"] for row in record["filesystem_pools"].values()) == 7
    assert final.read_bytes() == b"newstate" and temporary.read_bytes() == b"tmp"


def test_prewrite_failure_releases_empty_named_output_reservation(tmp_path):
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=0, load_workers=1)
        with pytest.raises(RuntimeError):
            with owner.artifact_writer(tmp_path / "not_written.npz", 128):
                raise RuntimeError("failed before write")
        record = owner.record()
    assert all(row["reserved_bytes"] == 0 for row in record["filesystem_pools"].values())
    assert all(row["disk_bytes"] == 0 for row in record["active_leases"])


def test_snapshot_permanent_ram_capacity_rejects_after_persistent_writer(tmp_path, monkeypatch):
    import research.source_execution as module

    monkeypatch.setattr(module, "filesystem_uses_host_memory", lambda path: True)
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=10, load_workers=1)
        with owner.artifact_writer(tmp_path / "retained.npz", 6):
            (tmp_path / "retained.npz").write_bytes(b"retain")
        assert owner.record()["free_staging_bytes"] == 4
        before = time.monotonic()
        with pytest.raises(ValueError, match="persistent files"):
            with owner.snapshot(tmp_path, 5):
                pytest.fail("permanent capacity shortage must reject before entering")
        assert time.monotonic() - before < 1
        assert owner.record()["peak_staging_bytes"] == 6
        for phase in ("nn", "load"):
            with pytest.raises(ValueError, match="persistent files"):
                with owner.reserve(phase, 5):
                    pytest.fail("persistent RAM shortage must not park an input waiter")
        assert owner.record()["nn_waiting"] == 0


def test_active_task_output_writer_cannot_wait_for_its_own_ram_reservation(tmp_path, monkeypatch):
    import research.source_execution as module

    monkeypatch.setattr(module, "filesystem_uses_host_memory", lambda path: True)
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=10, load_workers=1)
        with owner.artifact_directory(tmp_path / "output", 6):
            for phase in ("nn", "load"):
                with pytest.raises(ValueError, match="persistent files"):
                    with owner.reserve(phase, 8):
                        pytest.fail("a task cannot wait for its own outer writer to settle")
            with pytest.raises(ValueError, match="persistent files"):
                with owner.snapshot(tmp_path, 8):
                    pytest.fail("a task cannot wait for its own output RAM to become free")
        assert owner.record()["free_staging_bytes"] == 10
        assert owner.record()["nn_waiting"] == 0


def test_zero_byte_ram_writer_does_not_erase_later_actual_ram_settlement(tmp_path, monkeypatch):
    import research.source_execution as module

    monkeypatch.setattr(module, "filesystem_uses_host_memory", lambda path: True)
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=10, load_workers=1)
        path = tmp_path / "ram.npz"
        with owner.artifact_writer(path, 0):
            path.write_bytes(b"")
        with owner.artifact_writer(path, 6):
            path.write_bytes(b"ram")
        record = owner.record()
        assert record["free_staging_bytes"] == 7
        assert record["active_leases"][0]["nbytes"] == 3
        assert next(iter(record["filesystem_pools"].values()))["uses_host_memory"] is True


def test_atomic_json_inside_admitted_directory_is_charged_once(tmp_path):
    import json

    payload = {"subject": "参与者", "probability": [.25, .75]}
    expected = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=0, load_workers=2)
        with owner.artifact_directory(tmp_path / "outputs", 1024):
            owner.write_json(tmp_path / "outputs" / "results.json", payload)
        record = owner.record()
    assert (tmp_path / "outputs" / "results.json").read_bytes() == expected
    assert len(record["active_leases"]) == 1
    assert record["active_leases"][0]["disk_bytes"] == len(expected)
    assert sum(row["reserved_bytes"] for row in record["filesystem_pools"].values()) == len(expected)


def test_atomic_json_rewrite_tracks_unique_temporary_paths_without_double_charge(tmp_path):
    import json

    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=0, load_workers=1)
        owner.write_json(tmp_path / "progress.json", {"progress": 1})
        owner.write_json(tmp_path / "progress.json", {"progress": 12})
        record = owner.record()
    expected = (json.dumps({"progress": 12}, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    assert (tmp_path / "progress.json").read_bytes() == expected
    assert record["active_leases"][0]["paths"] == [str((tmp_path / "progress.json").resolve())]
    assert record["active_leases"][0]["disk_bytes"] == len(expected)
    assert len(list(tmp_path.iterdir())) == 1


def _snapshot_worker(owner, directory, marker, release):
    install_input_admission(owner, f"snapshot-{marker}")
    marker.with_suffix(".ready").write_text("waiting")
    with owner.snapshot(directory, 7):
        marker.write_text("entered")
        if not release.wait(10):
            raise TimeoutError("snapshot behavior probe was not released")


def test_spawned_snapshot_workers_share_one_loader_and_filesystem_pool(tmp_path):
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=16, load_workers=1)
        first = owner.snapshot(tmp_path, 7)
        first.__enter__()
        release = context.Event()
        marker = tmp_path / "entered.txt"
        child = context.Process(target=_snapshot_worker, args=(owner, tmp_path, marker, release))
        child.start()
        deadline = time.monotonic() + 5
        try:
            while not marker.with_suffix(".ready").exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.with_suffix(".ready").exists() and not marker.exists()
            assert owner.record()["load_active"] == 1
            assert len(owner.record()["filesystem_pools"]) == 1
            assert next(iter(owner.record()["filesystem_pools"].values()))["reserved_bytes"] == 7
            first.__exit__(None, None, None)
            first = None
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists() and owner.record()["load_active"] == 1
        finally:
            if first is not None:
                first.__exit__(None, None, None)
            release.set()
            child.join(10)
            if child.is_alive():
                child.terminate()
                child.join()
        assert child.exitcode == 0
        assert owner.record()["free_staging_bytes"] == 16 and owner.record()["load_active"] == 0
        assert all(row["reserved_bytes"] == 0 for row in owner.record()["filesystem_pools"].values())


def test_prepared_source_temporary_and_snapshot_leases_drain_after_failure():
    owner = PreparedSourceThreadMemory(budget_bytes=20, resident_bytes=10,
        temporary_bytes=[8], disk_bytes=12, snapshot_bytes=[9])
    with pytest.raises(RuntimeError):
        with owner.reserve(8, 9):
            assert owner.free == 2 and owner.disk_free == 3
            raise RuntimeError("adaptation failed")
    assert owner.free == 10 and owner.disk_free == 12
    with owner.reserve(8, 9):
        pass
    assert owner.peak_reserved == 8 and owner.peak_snapshot == 9
