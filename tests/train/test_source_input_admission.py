"""Byte staging preserves ready NN priority, resident transfer and failure cleanup."""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from research.source_execution import (
    HostMemoryAdmission,
    PreparedSourceThreadMemory,
    SourceInputAdmission,
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
            with owner.condition:
                assert owner.condition.wait_for(lambda: owner.state.nn_waiting == 1, timeout=5)
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
        with pytest.raises(ValueError, match="shared snapshot filesystem budget"):
            SourceInputAdmission(manager, capacity_bytes=10, load_workers=2)


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
