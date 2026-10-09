"""Verified external cleanup restores disk capacity without recrediting owned bytes."""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from research import source_execution
from research.source_execution import SourceInputAdmission


@pytest.fixture
def reclaim_owner(monkeypatch):
    manager = SimpleNamespace(SourceInputState=source_execution._SourceInputState)
    monkeypatch.setattr(source_execution, "filesystem_uses_host_memory", lambda path: False)
    return SourceInputAdmission(manager, capacity_bytes=10, load_workers=4)


def _physical_file(monkeypatch, path, *, blocks=16, links=1, freed_bytes=8192):
    # Windows has no st_blocks. Supply the observed Linux allocation metadata,
    # retaining the real file identity, while actually deleting the tiny file.
    path.write_bytes(b"retained exact original bytes")
    expected = path.lstat()
    free = {"bytes": 64 * 2**20 + 8192}
    original_lstat, original_unlink = Path.lstat, Path.unlink

    def lstat(target, *args, **kwargs):
        info = original_lstat(target, *args, **kwargs)
        if target != path:
            return info
        values = {key: getattr(info, key) for key in (
            "st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}
        values["st_nlink"] = links
        if blocks is not None:
            values["st_blocks"] = blocks
        return SimpleNamespace(**values)

    def unlink(target, *args, **kwargs):
        original_unlink(target, *args, **kwargs)
        if target == path:
            free["bytes"] += freed_bytes

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(source_execution.shutil, "disk_usage", lambda target:
        SimpleNamespace(total=2**30, used=2**30 - free["bytes"], free=free["bytes"]))
    return expected, free


def _filesystem(owner):
    return next(iter(owner.record()["filesystem_pools"].values()))


def test_deleted_old_cache_enables_later_pool_in_the_same_owner(tmp_path, monkeypatch, reclaim_owner):
    owner = reclaim_owner
    original = tmp_path / "old-cache.npz"
    expected, _ = _physical_file(monkeypatch, original)
    with owner.artifact_writer(tmp_path / "first-result.json", 0):
        pass
    initial = _filesystem(owner)["baseline_free_bytes"]
    with pytest.raises(ValueError, match="remaining shared filesystem"):
        with owner.artifact_writer(tmp_path / "later-pool.npz", 12288):
            pytest.fail("the original baseline cannot hold the next pool")
    receipt = owner.remove_external_file(original, expected_stat=expected)
    assert receipt["credited_bytes"] == 8192 and not original.exists()
    assert _filesystem(owner)["baseline_free_bytes"] == initial
    assert _filesystem(owner)["reclaimed_bytes"] == 8192
    with owner.snapshot(tmp_path, 12288):
        assert _filesystem(owner)["reserved_bytes"] == 12288
    assert _filesystem(owner)["reserved_bytes"] == 0
    with owner.artifact_writer(tmp_path / "later-pool.npz", 12288):
        (tmp_path / "later-pool.npz").write_bytes(b"x" * 12288)
    assert _filesystem(owner)["reserved_bytes"] == 12288


@pytest.mark.parametrize("coverage", ["active_directory", "committed_directory", "file", "temporary", "overlap"])
def test_owned_or_overlapping_output_is_preserved_without_any_credit(
    tmp_path, monkeypatch, reclaim_owner, coverage,
):
    owner = reclaim_owner
    directory = tmp_path / "results"
    directory.mkdir()
    original = directory / "owned-or-original.npz"
    expected, _ = _physical_file(monkeypatch, original)
    active = None
    if coverage in {"active_directory", "committed_directory", "overlap"}:
        active = owner.artifact_directory(directory, 8192)
        active.__enter__()
        if coverage != "active_directory":
            active.__exit__(None, None, None)
            active = None
    if coverage in {"file", "overlap"}:
        with owner.artifact_writer(original, 8):
            pass
    if coverage == "temporary":
        with owner.artifact_writer(directory / "final.npz", 8, temporary_paths=(original,)):
            pass
    before = owner.record()
    try:
        with pytest.raises(ValueError, match="named artifact"):
            owner.remove_external_file(original, expected_stat=expected)
        assert original.read_bytes() == b"retained exact original bytes"
        assert owner.record() == before
    finally:
        if active is not None:
            active.__exit__(None, None, None)


def test_deleted_path_cannot_be_recreated_and_recredited(tmp_path, monkeypatch, reclaim_owner):
    original = tmp_path / "old.npz"
    expected, _ = _physical_file(monkeypatch, original)
    reclaim_owner.remove_external_file(original, expected_stat=expected)
    before = reclaim_owner.record()
    with pytest.raises(ValueError, match="already reclaimed"):
        reclaim_owner.remove_external_file(original, expected_stat=expected)
    original.write_bytes(b"new external write at the same name")
    with pytest.raises(ValueError, match="already reclaimed"):
        reclaim_owner.remove_external_file(original, expected_stat=original.lstat())
    assert original.exists() and reclaim_owner.record() == before


def test_owned_growth_deletion_returns_its_lease_without_external_credit(tmp_path, reclaim_owner):
    path = tmp_path / "owned.npz"
    with reclaim_owner.artifact_writer(path, 8):
        path.write_bytes(b"newstate")
    before = reclaim_owner.record()
    assert _filesystem(reclaim_owner)["reserved_bytes"] == 8
    with pytest.raises(ValueError, match="named artifact"):
        reclaim_owner.remove_external_file(path, expected_stat=path.lstat())
    assert reclaim_owner.record() == before and path.read_bytes() == b"newstate"
    # Existing named settlement owns deletion of its own growth. It earns no
    # second external capacity credit and does not change the original baseline.
    with reclaim_owner.artifact_writer(path, 0):
        path.unlink()
    assert _filesystem(reclaim_owner)["reserved_bytes"] == 0
    assert _filesystem(reclaim_owner)["reclaimed_bytes"] == 0
    assert not reclaim_owner.record()["external_file_reclaims"]
    assert _filesystem(reclaim_owner)["baseline_free_bytes"] == next(
        iter(before["filesystem_pools"].values()))["baseline_free_bytes"]


@pytest.mark.parametrize("blocks,links,freed,credit,deferred", [
    (0, 1, 2**20, 0, 0), (8, 1, 2**20, 4096, 0), (16, 1, 2048, 2048, 6144),
    (16, 2, 0, 0, 0), (None, 1, 8192, 0, 0), (16, 1, 0, 0, 8192),
])
def test_credit_uses_deleted_allocation_separately_from_observed_free_space(
    tmp_path, monkeypatch, reclaim_owner, blocks, links, freed, credit, deferred,
):
    original = tmp_path / "external.npz"
    expected, _ = _physical_file(monkeypatch, original, blocks=blocks, links=links, freed_bytes=freed)
    receipt = reclaim_owner.remove_external_file(original, expected_stat=expected)
    assert receipt["credited_bytes"] == credit
    assert receipt["deferred_allocation_upper_bound_bytes"] == deferred
    assert receipt["observed_free_delta_bytes"] == freed
    assert _filesystem(reclaim_owner)["reclaimed_bytes"] == credit
    assert _filesystem(reclaim_owner)["deferred_reclaim_upper_bound_bytes"] == deferred
    assert not original.exists()


def test_actual_diskfree_remains_a_gate_after_credit(tmp_path, monkeypatch, reclaim_owner):
    original = tmp_path / "old.npz"
    expected, free = _physical_file(monkeypatch, original)
    reclaim_owner.remove_external_file(original, expected_stat=expected)
    free["bytes"] = 64 * 2**20 + 1024
    with pytest.raises(ValueError, match="remaining shared filesystem"):
        with reclaim_owner.artifact_writer(tmp_path / "new.npz", 2048):
            pytest.fail("credited capacity cannot bypass actual physical pressure")
    assert not (tmp_path / "new.npz").exists()


def test_pending_writers_existing_before_delayed_reclaim_do_not_overbook_free(
    tmp_path, monkeypatch, reclaim_owner,
):
    gib = 2**30
    original = tmp_path / "old.npz"
    expected, free = _physical_file(monkeypatch, original, blocks=10 * gib // 512, freed_bytes=0)
    free["bytes"] = 45 * gib
    barrier, release = Barrier(3), Event()

    def pending(index):
        with reclaim_owner.artifact_writer(tmp_path / f"pending{index}.npz", 10 * gib):
            barrier.wait(timeout=5)
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(pending, index) for index in range(2)]
        try:
            barrier.wait(timeout=5)
            assert _filesystem(reclaim_owner)["reserved_bytes"] == 20 * gib
            reclaim_owner.remove_external_file(original, expected_stat=expected)
            with pytest.raises(ValueError, match="remaining shared filesystem"):
                with reclaim_owner.artifact_writer(tmp_path / "overbook.npz", 30 * gib):
                    pytest.fail("two pending 10GiB writers plus 30GiB exceed actual 45GiB")
            free["bytes"] += 10 * gib
            with reclaim_owner.artifact_writer(tmp_path / "now-admissible.npz", 30 * gib):
                assert _filesystem(reclaim_owner)["reserved_bytes"] == 50 * gib
        finally:
            release.set()
        for future in futures:
            future.result()


@pytest.mark.parametrize("phase", ["named", "snapshot"])
def test_delayed_reclaim_protects_existing_snapshots_and_partial_writers_until_settlement(
    tmp_path, monkeypatch, reclaim_owner, phase,
):
    gib = 2**30
    original = tmp_path / "old.npz"
    expected, free = _physical_file(monkeypatch, original, blocks=10 * gib // 512, freed_bytes=0)
    free["bytes"] = 45 * gib
    with ExitStack() as stack:
        stack.enter_context(reclaim_owner.snapshot(tmp_path, 10 * gib))
        partial = tmp_path / "partial.npz"
        stack.enter_context(reclaim_owner.artifact_writer(partial, 10 * gib))
        partial.write_bytes(b"surviving partial")
        reclaim_owner.remove_external_file(original, expected_stat=expected)
        if phase == "named":
            with pytest.raises(ValueError, match="remaining shared filesystem"):
                with reclaim_owner.artifact_writer(tmp_path / "blocked.npz", 30 * gib):
                    pytest.fail("partly written output is still reserved until settlement")
        else:
            assert not reclaim_owner._ledger.try_snapshot(str(tmp_path), str(tmp_path.stat().st_dev),
                30 * gib, 0, False, "probe", reclaim_owner._identity())
    assert _filesystem(reclaim_owner)["reserved_bytes"] == len(b"surviving partial")
    assert partial.read_bytes() == b"surviving partial"
    with reclaim_owner.artifact_writer(tmp_path / "after-settlement.npz", 30 * gib):
        pass


def test_delayed_physical_release_is_usable_when_actual_free_arrives(tmp_path, monkeypatch, reclaim_owner):
    original = tmp_path / "old.npz"
    expected, free = _physical_file(monkeypatch, original, freed_bytes=0)
    receipt = reclaim_owner.remove_external_file(original, expected_stat=expected)
    assert receipt["credited_bytes"] == 0
    assert receipt["deferred_allocation_upper_bound_bytes"] == 8192
    assert receipt["observed_free_delta_bytes"] == 0
    with pytest.raises(ValueError, match="remaining shared filesystem"):
        with reclaim_owner.artifact_writer(tmp_path / "new.npz", 12288):
            pytest.fail("an unlink with delayed release cannot bypass the actual free-space gate")
    free["bytes"] += 8192  # statvfs refresh or the last external open FD closes
    with reclaim_owner.artifact_writer(tmp_path / "new.npz", 12288):
        (tmp_path / "new.npz").write_bytes(b"x" * 12288)
    assert _filesystem(reclaim_owner)["reserved_bytes"] == 12288


def test_no_deferred_credit_preserves_the_original_materialized_writer_fast_path(
    tmp_path, monkeypatch, reclaim_owner,
):
    gib = 2**30
    original = tmp_path / "old.npz"
    _, free = _physical_file(monkeypatch, original)
    free["bytes"] = 45 * gib
    with reclaim_owner.artifact_writer(tmp_path / "materialized.npz", 20 * gib):
        free["bytes"] -= 20 * gib  # this writer's file now occupies its reserved disk space
        assert _filesystem(reclaim_owner)["reclaimed_bytes"] == 0
        assert _filesystem(reclaim_owner)["deferred_reclaim_upper_bound_bytes"] == 0
        with reclaim_owner.artifact_writer(tmp_path / "next.npz", 10 * gib):
            assert _filesystem(reclaim_owner)["reserved_bytes"] == 30 * gib


def test_atomic_filesystem_record_observes_current_unsettled_reservations(tmp_path, reclaim_owner):
    path = tmp_path / "partial.npz"
    with reclaim_owner.artifact_writer(path, 6), reclaim_owner.snapshot(tmp_path, 4), reclaim_owner.reserve("nn", 3):
        record = reclaim_owner.record()
        filesystem = next(iter(record["filesystem_pools"].values()))
        assert filesystem["reserved_bytes"] == filesystem["live_unsettled_reserved_bytes"] == 10
        assert sum(row["disk_bytes"] for row in record["active_leases"]
            if row["kind"] in {"anonymous_snapshot", "persistent_artifact"}) == 10
        path.write_bytes(b"abc")
        assert _filesystem(reclaim_owner)["live_unsettled_reserved_bytes"] == 10
    record = reclaim_owner.record()
    filesystem = next(iter(record["filesystem_pools"].values()))
    assert filesystem["reserved_bytes"] == 3 and filesystem["live_unsettled_reserved_bytes"] == 0
    assert record["free_staging_bytes"] == 10 and path.read_bytes() == b"abc"


@pytest.mark.parametrize("freed_bytes", [8192, 0])
def test_tmpfs_reclaim_does_not_expand_fixed_ram_staging(tmp_path, monkeypatch, reclaim_owner, freed_bytes):
    monkeypatch.setattr(source_execution, "filesystem_uses_host_memory", lambda path: True)
    original = tmp_path / "old.npz"
    expected, _ = _physical_file(monkeypatch, original, freed_bytes=freed_bytes)
    with reclaim_owner.artifact_writer(tmp_path / "retained.npz", 6):
        (tmp_path / "retained.npz").write_bytes(b"retain")
    reclaim_owner.remove_external_file(original, expected_stat=expected)
    assert reclaim_owner.record()["free_staging_bytes"] == 4
    with pytest.raises(ValueError, match="persistent files"):
        with reclaim_owner.snapshot(tmp_path, 5):
            pytest.fail("disk reclamation does not raise the fixed RAM capacity")


def test_changed_original_and_retired_task_are_not_deleted(tmp_path, monkeypatch, reclaim_owner):
    original = tmp_path / "old.npz"
    expected, _ = _physical_file(monkeypatch, original)
    original.write_bytes(b"changed after exact archive verification")
    with pytest.raises(ValueError, match="identity"):
        reclaim_owner.remove_external_file(original, expected_stat=expected)
    assert original.exists() and not reclaim_owner.record()["filesystem_pools"]
    monkeypatch.setenv("N2P3_CLOUD_TASK_ID", "completed-cleanup")
    reclaim_owner.retire_task("completed-cleanup")
    with pytest.raises(RuntimeError, match="retired"):
        reclaim_owner.remove_external_file(original, expected_stat=original.lstat())
    assert original.exists()


def test_nonregular_target_is_rejected_before_unlink(tmp_path, reclaim_owner):
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        reclaim_owner.remove_external_file(directory, expected_stat=directory.lstat())
    assert directory.is_dir()
    if os.name != "nt":
        link = tmp_path / "link"
        link.symlink_to(directory)
        with pytest.raises(ValueError, match="regular file"):
            reclaim_owner.remove_external_file(link, expected_stat=link.lstat())
        assert link.is_symlink()


def test_real_manager_consumes_parent_stat_identity_and_records_removal(tmp_path):
    original = tmp_path / "archived.npz"
    original.write_bytes(b"exact archived original")
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=0, load_workers=1)
        receipt = owner.remove_external_file(original, expected_stat=original.lstat())
        record = owner.record()
    assert not original.exists() and receipt["removed"]
    assert record["external_file_reclaims"] == [receipt]
    assert record["free_staging_bytes"] == 0
