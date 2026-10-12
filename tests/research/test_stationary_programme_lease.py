"""The scientific entry owns one attempt; shared cloud code owns prior trees."""
from __future__ import annotations

import threading

import psutil
import pytest

from experiments import run_stationary_cloud_panel as entry
from research.local_program_control import atomic_record


def test_live_controller_and_live_prior_local_worker_reject_before_executor(tmp_path, monkeypatch):
    plan = tmp_path/"plan.json"
    atomic_record(plan, {})
    own = psutil.Process()
    lease = tmp_path/"controller.lease.json"
    atomic_record(lease, {"run_id": "live", "controller_pid": own.pid, "controller_created": own.create_time()})
    with pytest.raises(RuntimeError, match="another live controller"), entry.programme_attempt(plan):
        pytest.fail("live controller entered")
    lease.unlink()
    atomic_record(tmp_path/"worker.A01.old.identity.json", {
        "actual_worker_pid": own.pid, "actual_worker_created": own.create_time()})
    with pytest.raises(RuntimeError, match="descendant remains live"), entry.programme_attempt(plan):
        pytest.fail("live old local worker entered")


def test_missing_or_unverified_shared_history_api_is_hold(tmp_path, monkeypatch):
    plan = tmp_path/"plan.json"
    atomic_record(plan, {})
    monkeypatch.delattr(entry.cloud_execution, "assert_no_prior_owned_live", raising=False)
    with pytest.raises(RuntimeError, match="historical owned-tree"), entry.programme_attempt(plan):
        pytest.fail("missing API entered")
    def unknown(directory):
        assert directory == tmp_path/"cloud_diagnostics"
        raise RuntimeError("prior cloud owned descendant remains live")
    monkeypatch.setattr(entry.cloud_execution, "assert_no_prior_owned_live", unknown, raising=False)
    with pytest.raises(RuntimeError, match="owned descendant remains live"), entry.programme_attempt(plan):
        pytest.fail("unverified tree entered")
    assert not (tmp_path/"controller.lease.json").exists()


def test_retired_attempt_archives_lease_and_same_output_rejects_second_live_attempt(tmp_path, monkeypatch):
    plan = tmp_path/"plan.json"
    atomic_record(plan, {})
    own = psutil.Process()
    atomic_record(tmp_path/"controller.lease.json", {
        "run_id": "old", "controller_pid": own.pid, "controller_created": own.create_time()-1.})
    histories = []
    monkeypatch.setattr(entry.cloud_execution, "assert_no_prior_owned_live", histories.append, raising=False)
    with entry.programme_attempt(plan) as attempt:
        assert attempt["plan_file_sha256"] == entry.sha(plan)
        assert (tmp_path/"controller.expired.old.json").exists()
        failures = []
        def compete():
            try:
                with entry.programme_attempt(plan):
                    failures.append("incorrect_entry")
            except RuntimeError as error:
                failures.append(str(error))
        peer = threading.Thread(target=compete)
        peer.start()
        peer.join(timeout=3.)
        assert not peer.is_alive()
        assert len(failures) == 1 and "another live controller" in failures[0]
    assert not (tmp_path/"controller.lease.json").exists()
    assert (tmp_path/f"controller.finished.{attempt['run_id']}.json").exists()
    assert histories == [tmp_path/"cloud_diagnostics"]
