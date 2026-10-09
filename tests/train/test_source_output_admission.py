"""Source outputs consume their own bytes, independently of resident EEG RAM."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research import source_execution
from research.source_execution import (
    HostMemoryAdmission,
    SourceInputAdmission,
    install_input_admission,
)
from tests.support.source_programs import (
    source_early_stopping_config,
    source_early_stopping_data,
)


@pytest.fixture
def owner(monkeypatch):
    # Use the real atomic state locally: no large data, GPU or extra manager.
    manager = SimpleNamespace(SourceInputState=source_execution._SourceInputState,
                              HostMemoryState=source_execution._HostMemoryState)
    admission = SourceInputAdmission(manager, capacity_bytes=0, load_workers=4)
    monkeypatch.setattr(source_execution, "filesystem_uses_host_memory", lambda path: False)
    previous = source_execution.current_input_admission()
    install_input_admission(admission)
    try:
        yield admission, manager
    finally:
        install_input_admission(previous)


def test_four_source_groups_keep_18gib_ram_on_50gib_output_disk(tmp_path, monkeypatch, owner):
    import experiments.run_source_training_chain as runner
    import train.runtime as runtime

    admission, manager = owner
    gib = 2**30
    monkeypatch.setattr(source_execution.shutil, "disk_usage", lambda path:
        SimpleNamespace(total=50 * gib, used=126 * gib // 10, free=374 * gib // 10))
    host = HostMemoryAdmission(manager, {
        "host_memory_budget_bytes": 82 * gib, "shared_staging_memory_bytes": 10 * gib,
        "source_worker_memory_bytes": 18 * gib, "target_worker_memory_bytes": 4 * gib,
        "source_workers": 4, "target_workers": 4, "gpu_slots": 4,
        "cpu_budget": 25, "coordinator_cpu_threads": 1, "native_threads_per_worker": 6})
    reservations = [host.reserve_phase("source", cpu_threads=6, memory_bytes=18 * gib) for _ in range(4)]
    assert all(reservations)
    barrier = Barrier(4)
    monkeypatch.setattr(runtime, "configure_spawned_worker_threads", lambda value: None)
    monkeypatch.setattr(runtime, "cpu_thread_budget", lambda value: nullcontext())
    monkeypatch.setattr(runner, "_assert_source_bytes", lambda *args: None)

    def execute(plan, *, output, **kwargs):
        assert plan["_source_worker_memory_bytes"] == 18 * gib
        assert host.record()["reserved_cpu_threads"] == 25
        assert host.record()["active_gpu_groups"] == 4
        barrier.wait(timeout=3)
        admission.write_json(output / "source_chain_record.json", {"source_snapshot": {}})

    monkeypatch.setattr(runner, "execute_source_chain", execute)
    plan = {"_source_input_admission": admission, "_source_worker_memory_bytes": 18 * gib,
            "_native_threads_per_worker": 6, "_phase_reserved": True,
            "source_snapshot_manifest": "frozen.json", "outer_folds": [{"name": "fixed"}]}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(runner._source_group_worker,
            (plan, tmp_path / f"group{index}", "cuda:0", ("P2_source",))) for index in range(4)]
    errors = [future.exception() for future in futures]
    for reservation in reservations:
        host.release_phase(reservation)
    assert errors == [None] * 4
    assert all(future.result().endswith("source_chain_record.json") for future in futures)
    assert sum(row["reserved_bytes"] for row in admission.record()["filesystem_pools"].values()) < 1024


@pytest.mark.parametrize("selecting", [False, True])
def test_actual_source_checkpoints_and_reuse_are_owned_with_identical_model_state(tmp_path, owner, selecting):
    from dataclasses import replace

    from experiments.run_source_training_chain import fit_or_reuse_source_trajectory
    from train.runtime import cpu_thread_budget

    admission, _ = owner
    config = source_early_stopping_config()
    if not selecting:
        config = replace(config, global_early_stopping=None, fixed_updates=1)
    arguments = dict(registry={}, dataset=source_early_stopping_data(), rows=np.arange(32), config=config,
        source_cache_sha256="a" * 64, source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["outer"], device="cpu", evidence_scope="software_correctness")
    with cpu_thread_budget(1):
        original = fit_or_reuse_source_trajectory(**arguments, destination=tmp_path / "first", task_id="first")
        reused = fit_or_reuse_source_trajectory(**arguments, destination=tmp_path / "reuse", task_id="reuse")
    assert original["physical_optimizer_updates"] > 0 and reused["physical_optimizer_updates"] == 0
    paths = [Path(path) for result in (original, reused) for path in result["checkpoints"].values()]
    leases = {row["path"]: row for row in admission.record()["active_leases"]}
    assert all(str(path) in leases for path in paths)
    assert all(leases[str(path)]["disk_bytes"] == path.stat().st_size for path in paths)
    first = torch.load(paths[0], weights_only=False)
    second = torch.load(paths[-1], weights_only=False)
    for key, value in first["trunk_state_dict"].items():
        torch.testing.assert_close(value, second["trunk_state_dict"][key], rtol=0, atol=0)
    if selecting:
        assert paths[0].read_bytes() == paths[-1].read_bytes()
    assert first["source_fit_ledger"] == second["source_fit_ledger"]
    assert admission.record()["free_staging_bytes"] == 0


def test_progress_append_charges_utf8_bytes_once_and_preserves_events(tmp_path, owner):
    from baselines.progress import make_epoch_progress_callback

    admission, _ = owner
    callback = make_epoch_progress_callback(tmp_path, 0)
    callback({"stage": "选轮", "epoch": 1})
    callback({"stage": "refit", "epoch": 2})
    path = tmp_path / "fold_0.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in rows] == ["选轮", "refit"]
    leases = {row["path"]: row for row in admission.record()["active_leases"]}
    assert leases[str(path)]["disk_bytes"] == path.stat().st_size
    assert sum(row["reserved_bytes"] for row in admission.record()["filesystem_pools"].values()) == path.stat().st_size


def test_checkpoint_space_failure_rejects_before_overwriting_original(tmp_path, monkeypatch, owner):
    admission, _ = owner
    path = tmp_path / "source.pt"
    path.write_bytes(b"original")
    monkeypatch.setattr(source_execution.shutil, "disk_usage", lambda path:
        SimpleNamespace(total=50 * 2**30, used=50 * 2**30 - 64 * 2**20, free=64 * 2**20))
    with pytest.raises(ValueError, match="remaining shared filesystem"):
        source_execution.save_source_checkpoint(path, {"tensor": torch.arange(8)})
    assert path.read_bytes() == b"original"
    assert not admission.record()["active_leases"]


def test_failed_checkpoint_partial_remains_charged_and_is_not_erased(tmp_path, monkeypatch, owner):
    admission, _ = owner
    original_save = torch.save

    def fail_on_file(payload, stream):
        if hasattr(stream, "fileno"):
            stream.write(b"partial-checkpoint")
            stream.flush()
            raise OSError("actual checkpoint write failure")
        return original_save(payload, stream)

    monkeypatch.setattr(torch, "save", fail_on_file)
    path = tmp_path / "source.pt"
    with pytest.raises(OSError, match="actual checkpoint write failure"):
        source_execution.save_source_checkpoint(path, {"tensor": torch.arange(8)})
    assert path.read_bytes() == b"partial-checkpoint"
    lease = admission.record()["active_leases"][0]
    assert lease["committed"] and lease["disk_bytes"] == len(b"partial-checkpoint")


@pytest.mark.parametrize("failure", ["partial", "insufficient"])
def test_selected_checkpoint_copy_failure_keeps_original_and_settles_partial(tmp_path, monkeypatch, owner, failure):
    from data.source_records import file_sha256
    from experiments import run_source_training_chain as runner

    admission, _ = owner
    origin = tmp_path / "original.pt"
    origin.write_bytes(b"original producer bytes")
    ledger = {"training_participant_keys": []}
    contract = SimpleNamespace(architecture={}, training_participant_keys=[], source_cache_sha256="cache",
        source_snapshot_sha256="snapshot", validation={"fit_ledger": ledger}, digest=lambda: "contract")
    monkeypatch.setattr("data.study.dataset_study_membership", lambda dataset: {})
    monkeypatch.setattr("transfer.checkpoint.load_checkpoint_payload", lambda path: {})
    monkeypatch.setattr("transfer.checkpoint.checkpoint_training_contract", lambda payload: contract)
    monkeypatch.setattr("research.source_selection.source_architecture_record", lambda *args: {})
    if failure == "partial":
        def fail_copy(source, target):
            Path(target).write_bytes(b"partial")
            raise OSError("actual copy failure")
        monkeypatch.setattr(runner.shutil, "copyfile", fail_copy)
        expected = (OSError, "actual copy failure")
    else:
        monkeypatch.setattr(source_execution.shutil, "disk_usage", lambda path:
            SimpleNamespace(total=50 * 2**30, used=50 * 2**30 - 64 * 2**20, free=64 * 2**20))
        expected = (ValueError, "remaining shared filesystem")
    with pytest.raises(expected[0], match=expected[1]):
        runner._materialize_source_trajectory(
            registry={"signature": [{"checkpoints": {1: {"path": str(origin), "sha256": file_sha256(origin)}},
                "task_id": "original", "source_cache_sha256": "cache"}]},
            dataset=None, rows=None, config=SimpleNamespace(global_early_stopping=True),
            destination=tmp_path / "reuse", task_id="reuse", source_cache_sha256="cache",
            source_snapshot_sha256="snapshot", excluded_source_datasets=[], device="cpu",
            evidence_scope="software_correctness", signature="signature", signature_record={},
            fit_ledger=ledger, gpu_slots=None, architecture=None)
    assert origin.read_bytes() == b"original producer bytes"
    path = tmp_path / "reuse/source.u1.pt"
    if failure == "partial":
        assert path.read_bytes() == b"partial"
        lease = admission.record()["active_leases"][0]
        assert lease["committed"] and lease["disk_bytes"] == len(b"partial")
    else:
        assert not path.exists() and not admission.record()["active_leases"]


def test_sibling_eeg_cache_uses_payload_budget_separate_from_group_json(tmp_path, owner):
    from data.epochs import save_epoch_dataset

    admission, _ = owner
    cache = tmp_path / "source_views/fold/source.npz"
    save_epoch_dataset(cache, source_early_stopping_data(), compressed=False)
    record = tmp_path / "groups/fold/source_chain_record.json"
    admission.write_json(record, {"source_cache": str(cache)})
    leases = {row["path"]: row for row in admission.record()["active_leases"]}
    paths = (cache, cache.with_suffix(".record.json"), record)
    assert set(leases) == {str(path) for path in paths}
    assert all(leases[str(path)]["disk_bytes"] == path.stat().st_size for path in paths)
    assert sum(row["reserved_bytes"] for row in admission.record()["filesystem_pools"].values()) == sum(
        path.stat().st_size for path in paths)
