"""Shared replay bytes remain admitted without retaining decode worker slots."""
import multiprocessing

import pytest

from research.source_execution import SourceInputAdmission


def test_resident_inputs_leave_decode_slots_free_and_retry_without_waiting():
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=10, load_workers=1)
        lease = owner.try_resident_input(8)
        assert lease is not None
        assert owner.record()["resident_input_bytes"] == 8
        assert owner.record()["free_staging_bytes"] == 2
        assert owner.record()["load_active"] == owner.record()["nn_active"] == 0
        with owner.reserve("load", 0), owner.reserve("nn", 2):
            assert owner.record()["load_active"] == owner.record()["nn_active"] == 1
            assert owner.try_resident_input(1) is None
        assert owner.try_resident_input(3) is None
        lease.release_unused()
        again = owner.try_resident_input(10)
        assert again is not None
        again.release_unused()
        assert owner.record()["free_staging_bytes"] == 10
        assert owner.record()["resident_input_bytes"] == 0
        assert owner.record()["load_active"] == owner.record()["nn_active"] == 0


def test_retired_holder_and_partial_release_preserve_exact_byte_accounting(monkeypatch):
    monkeypatch.setenv("N2P3_CLOUD_TASK_ID", "shared-holder")
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=10, load_workers=1)
        lease = owner.try_resident_input(8)
        lease.release_unused(3)
        assert lease.active and lease.nbytes == 3
        assert owner.record()["resident_input_bytes"] == 3
        assert owner.record()["free_staging_bytes"] == 7
        owner.retire_task("shared-holder")
        assert owner.record()["free_staging_bytes"] == 10
        assert owner.record()["resident_input_bytes"] == 0
        with pytest.raises(RuntimeError, match="retired"):
            owner.try_resident_input(1)
        lease.release_unused()
        assert owner.record()["free_staging_bytes"] == 10


@pytest.mark.parametrize("amount", [False, -1, 11, 1.5])
def test_resident_input_claim_rejects_invalid_and_impossible_sizes(amount):
    with multiprocessing.get_context("spawn").Manager() as manager:
        owner = SourceInputAdmission(manager, capacity_bytes=10, load_workers=1)
        with pytest.raises(ValueError):
            owner.try_resident_input(amount)
        assert owner.record()["free_staging_bytes"] == 10
