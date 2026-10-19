"""Narrow probes for the controller's durable prior-tree admission gate."""
from __future__ import annotations

import os

import psutil
import pytest

from research.cloud_execution import (
    CLOUD_OWNERSHIP_SCHEMA,
    CloudOwnershipError,
    assert_no_prior_owned_live,
)
from research.execution import atomic_write_json


def _settled_record(*, pid=2_000_000_000, create_time=1.0, status="settled", pool_instance_id="old-pool"):
    group = os.getpgrp() if os.name != "nt" else None
    identity = {"pid": pid, "create_time": create_time, "process_group": group}
    token = "tree-token"
    return {"schema": CLOUD_OWNERSHIP_SCHEMA, "status": status,
            "controller": identity, "identity": {"run_id": "old", "pool_instance_id": pool_instance_id},
            "run_status": "failed", "tasks": [{"task_id": "old-task", "status": "retired",
                "tree_token": token, "bootstrap": {**identity, "tree_token": token},
                "containment": {"status": "admitted", "backend": "linux_subreaper_and_owned_process_groups"},
                "owned_processes": [identity],
                "retirement": {"active_owned_processes_after_exit": 0}}]}


def test_settled_dead_pid_birth_receipt_allows_a_new_controller(tmp_path):
    atomic_write_json(tmp_path / "old-pool" / ".cloud_controller.old.json",
                      _settled_record(pool_instance_id="old-pool"))
    assert assert_no_prior_owned_live(tmp_path) is None


def test_unverified_history_blocks_even_when_recorded_pid_is_gone(tmp_path):
    atomic_write_json(tmp_path / "old-pool" / ".cloud_controller.old.json",
                      _settled_record(status="running", pool_instance_id="old-pool"))
    with pytest.raises(CloudOwnershipError, match="unverified"):
        assert_no_prior_owned_live(tmp_path)


def test_live_same_birth_owned_pid_blocks_resume(tmp_path):
    process = psutil.Process()
    atomic_write_json(tmp_path / "old-pool" / ".cloud_controller.old.json",
                      _settled_record(pid=process.pid, create_time=process.create_time(),
                                      pool_instance_id="old-pool"))
    with pytest.raises(CloudOwnershipError, match="remains live"):
        assert_no_prior_owned_live(tmp_path)


def test_legacy_diagnostic_without_controller_receipt_is_unverified(tmp_path):
    (tmp_path / "old" ).mkdir()
    (tmp_path / "old" / "worker_1.state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CloudOwnershipError, match="no ownership receipt"):
        assert_no_prior_owned_live(tmp_path)


def test_settled_pool_cannot_cover_another_pool_without_a_receipt(tmp_path):
    atomic_write_json(tmp_path / "pool_a" / ".cloud_controller.old.json",
                      _settled_record(pool_instance_id="pool_a"))
    (tmp_path / "pool_b").mkdir()
    (tmp_path / "pool_b" / "worker_2.state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CloudOwnershipError, match="pool_b"):
        assert_no_prior_owned_live(tmp_path)


def test_foreign_pool_receipt_does_not_authorize_its_directory(tmp_path):
    atomic_write_json(tmp_path / "pool_a" / ".cloud_controller.old.json",
                      _settled_record(pool_instance_id="pool_a"))
    atomic_write_json(tmp_path / "pool_b" / ".cloud_controller.foreign.json",
                      _settled_record(pool_instance_id="pool_a"))
    with pytest.raises(CloudOwnershipError, match="pool_b"):
        assert_no_prior_owned_live(tmp_path)


def test_spawn_failure_receipt_cannot_hide_an_actual_live_created_process(tmp_path):
    process = psutil.Process()
    identity = {"pid": process.pid, "create_time": process.create_time(),
                "process_group": os.getpgrp() if os.name != "nt" else None}
    record = _settled_record()
    record["tasks"] = [{"task_id": "failed-launch", "status": "spawn_failed", "launch": identity,
                        "retirement": {"active_owned_processes_after_exit": 0,
                                       "owned_processes": [identity]}}]
    atomic_write_json(tmp_path / "old-pool" / ".cloud_controller.old.json", record)
    with pytest.raises(CloudOwnershipError, match="remains live"):
        assert_no_prior_owned_live(tmp_path)


def test_legacy_worker_state_must_match_the_receipt_pool(tmp_path):
    pool = tmp_path / "old-pool"
    atomic_write_json(pool / ".cloud_controller.old.json", _settled_record())
    atomic_write_json(pool / "worker_1.state.json", {"identity": {"pool_instance_id": "foreign-pool"}})
    with pytest.raises(CloudOwnershipError, match="worker state differs"):
        assert_no_prior_owned_live(tmp_path)
