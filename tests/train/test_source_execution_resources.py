from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from research.source_execution import (
    HostMemoryAdmission,
    SharedTrajectoryRegistry,
    resolve_source_resources,
)
from tests.support.datasets import epoch_dataset, source_dataset
from tests.support.io import read_json, write_json


def _append_registry_entry(directory):
    registry = SharedTrajectoryRegistry(directory)
    with registry.transaction("a" * 64) as local:
        entries = local["a" * 64]
        if not entries:
            entries.append({"checkpoints": {500: {"path": "source.pt", "sha256": "b" * 64}}})
            return True
        assert 500 in entries[0]["checkpoints"]
        return False


def test_registry_transaction_serializes_competing_publishers(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_append_registry_entry, [tmp_path] * 16))
    assert sum(outcomes) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_registry_lock_releases_after_failure(tmp_path):
    registry = SharedTrajectoryRegistry(tmp_path)
    with pytest.raises(RuntimeError):
        with registry.transaction("a" * 64):
            raise RuntimeError("failure before publication")
    assert _append_registry_entry(tmp_path)


def test_auto_resources_use_quota_and_memory_instead_of_one_worker():
    result = resolve_source_resources({}, 9, "cuda", cpu_count=25, free_device_bytes=32 * 2**30,
                                      free_host_bytes=256 * 2**30)
    assert result["source_workers"] == 9
    assert result["gpu_slots"] == result["target_workers"] == 8
    assert result["inner_workers_per_group"] == 4
    assert result["compiler_threads_per_worker"] == 1
    limited = resolve_source_resources({"cpu_budget": 3}, 9, "cuda",
                                       cpu_count=25, free_device_bytes=6 * 2**30, free_host_bytes=256 * 2**30)
    assert limited["source_workers"] == 3
    assert limited["gpu_slots"] == 1
    explicit = resolve_source_resources({"source_workers": 2, "target_workers": 3}, 9,
                                         "cuda", cpu_count=25, free_device_bytes=32 * 2**30,
                                         free_host_bytes=256 * 2**30)
    assert explicit["source_workers"] == 2 and explicit["target_workers"] == 3
    for value in (0, -1, True):
        with pytest.raises(ValueError, match="CPU budget"):
            resolve_source_resources({"cpu_budget": value}, 9, "cpu", cpu_count=25)


def test_memory_budget_limits_groups_and_rejects_unaffordable_worker():
    result = resolve_source_resources({}, 9, "cpu", cpu_count=25, free_host_bytes=16 * 2**30)
    assert result["source_workers"] == 2
    assert result["target_workers"] == 3
    with pytest.raises(ValueError, match="available host memory"):
        resolve_source_resources({}, 9, "cpu", cpu_count=25, free_host_bytes=4 * 2**30)


def test_memory_admission_releases_failed_children_and_expands_after_group_exit():
    import multiprocessing

    resources = resolve_source_resources({}, 9, "cpu", cpu_count=25, free_host_bytes=32 * 2**30)
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)
        original = admission.state.free
        with admission.source_group(), admission.source_group(), admission.source_group():
            with admission.target_workers(6) as workers:
                assert workers == 2
                with admission.target_workers(6) as other_workers:
                    assert other_workers == 1
            with pytest.raises(RuntimeError):
                with admission.target_workers(6) as workers:
                    assert workers == 2
                    raise RuntimeError("worker failed")
            assert admission.state.free == original - 3 * admission.group_bytes
        assert admission.state.free == original
        with admission.source_group(), admission.target_workers(6) as workers:
            assert workers == 5
        assert admission.state.free == original


def test_nested_workers_share_cpu_quota_across_groups():
    import multiprocessing

    resources = resolve_source_resources({}, 2, "cpu", cpu_count=4, free_host_bytes=256 * 2**30)
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)
        with admission.source_group(), admission.source_group():
            with admission.target_workers(4) as first:
                assert first == 3
                with admission.target_workers(4) as second:
                    assert second == 1
            assert admission.state.child_cpus == 0


def test_source_groups_preserve_outer_library_and_layout_boundaries():
    from experiments.run_source_training_chain import source_execution_groups

    outers = [{"name": name} for name in ("a", "b", "c", "d")]
    outers.append({"name": "family", "family_exclusion": True})
    groups = source_execution_groups({"outer_folds": outers}, ("S1", "E2", "S2"))
    assert len(groups) == 9
    assert groups[-1] == (outers[-1], "common_three")
    assert all(layout == "common_three" for _, layout in source_execution_groups(
        {"outer_folds": outers}, ("E2", "S2")))


def test_shared_registry_reuses_real_source_fit_across_independent_instances(tmp_path):
    import numpy as np

    from baselines.deep import DeepConfig
    from experiments.run_source_training_chain import fit_or_reuse_source_trajectory

    dataset = source_dataset()
    config = DeepConfig(fixed_updates=1, batch_size=4, sampling="uniform_rows", compile_mode=None)
    args = dict(dataset=dataset, rows=np.arange(4), config=config, source_cache_sha256="a" * 64,
                source_snapshot_sha256="b" * 64, excluded_source_datasets=["outer"],
                device="cpu", evidence_scope="software_correctness")
    first = fit_or_reuse_source_trajectory(registry=SharedTrajectoryRegistry(tmp_path / "registry"),
        destination=tmp_path / "first", task_id="first", **args)
    second = fit_or_reuse_source_trajectory(registry=SharedTrajectoryRegistry(tmp_path / "registry"),
        destination=tmp_path / "second", task_id="second", **args)
    assert first["physical_optimizer_updates"] == 1
    assert second["physical_optimizer_updates"] == 0
    assert second["source_training_reuse"]["origin_task"] == "first"
    assert Path(first["checkpoints"][1]) != Path(second["checkpoints"][1])


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_source_cli_executes_layouts_from_its_snapshot(tmp_path, workers):
    import subprocess
    import sys

    import numpy as np
    import pandas as pd

    from data.channel import build_channel_identity
    from data.contract import SOURCE_COHORT_DATA_CONTRACTS
    from data.domain import annotate_source_domain, namespace_epoch_dataset
    from data.epochs import (
        concatenate_epoch_datasets,
        preprocessing_spec_from_contract,
        save_epoch_dataset,
    )
    from data.events import observed_only_timeline
    from data.identity import DatasetIdentityTable
    from experiments.freeze_source_snapshot import freeze_source
    from transfer.target_experiments import user_fixed_target_batch_rule

    layouts = ("six", "common_three")
    views = []
    for library in ("a", "b", "c", "held"):
        source = epoch_dataset()
        identity = build_channel_identity(("Fz", "P3", "Pz", "P4", "PO8", "Oz"), allow_missing_positions=False)
        source.channel_names, source.channel_positions_m = identity.names, identity.coords
        source.channel_mask = np.ones(6, bool)
        source.preprocessing = preprocessing_spec_from_contract(SOURCE_COHORT_DATA_CONTRACTS["causal"])
        source.subject_ids = np.repeat(["s1", "s2", "s3", "s4"], 12)
        source.y = np.tile([0, 1], 24)
        source.metadata = pd.DataFrame({"subject": source.subject_ids, "run": ["1"] * 48})
        source.event_timeline = observed_only_timeline(dataset_id=library, subject_ids=source.subject_ids,
            stimulus_ids=np.arange(48), onset_times_s=np.arange(48) * 2.,
            evidence_available_times_s=np.arange(48) * 2. + 1.2,
            group_ids=source.subject_ids, online_causal=True, timing_source="synthetic_schedule")
        source.X = np.zeros((48, 6, source.preprocessing.n_times), np.float32)
        source.identity_table = DatasetIdentityTable.from_source_rows(source.subject_ids.tolist(), [library] * 48)
        source.provenance["source_reference"] = "measured_channel:CZ"
        views.append(annotate_source_domain(namespace_epoch_dataset(source, library), library))
    source = concatenate_epoch_datasets(views[:3], name="parallel_source", provenance={
        "source_reference": "measured_channel:CZ",
        "source_domain_axis": {**views[0].provenance["source_domain_axis"], "domains": ["a", "b", "c"]}})
    cache = save_epoch_dataset(tmp_path / "source.npz", source)
    target = save_epoch_dataset(tmp_path / "target.npz", views[3])
    manifest = freeze_source(tmp_path / "snapshot", materialize=True)
    plan = {"schema": "n2p3_source_training_chain/1", "source_snapshot_manifest": str(manifest),
            "evidence_scope": "software_correctness", "correctness_source_updates": 1,
            "correctness_candidate_updates": [1], "compile_mode": None, "cpu_budget": 2,
            "source_layouts": layouts, "source_workers": workers,
            "source_worker_memory_gib": 0.25, "target_worker_memory_gib": 0.25,
            "batch_rule": user_fixed_target_batch_rule(), "outer_folds": [{"name": "held", "source_cache": str(cache),
                "excluded_source_datasets": ["held"], "source_protocols": {}, "target_cache": str(target),
                "target_study": "held", "target_protocol": {"kind": "single_recording_prefix"}}]}
    path = tmp_path / "plan.json"
    write_json(path, plan)
    completed = subprocess.run([sys.executable, str(manifest.parent / "checkout/experiments/run_source_training_chain.py"),
        "--plan", str(path), "--output", str(tmp_path / "run"), "--stages", "S1", "--device", "cpu"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = read_json(tmp_path / "run/source_chain_record.json")
    resources = report["resource_policy"]
    effective_workers = resources["source_workers"]
    assert 1 <= effective_workers <= workers
    if effective_workers < workers:
        assert resources["host_memory_budget_bytes"] < workers * resources["source_worker_memory_bytes"]
    assert len({row["progress"]["pid"] for row in report["groups"]}) == effective_workers
    assert report["task_count"] == 4 * len(layouts)
    assert report["source_failure_count"] == 0
    assert report["completed_physical_optimizer_updates"] == 4 * len(layouts)


def test_cloud_floor_routes_25_threads_and_four_gpu_workers_without_silent_clamp():
    import json

    plan = json.loads((Path(__file__).resolve().parents[2] / "configs/cloud_training_resources.json").read_text())
    resources = resolve_source_resources(plan, 5, "cuda", cpu_count=25,
        free_device_bytes=32 * 2**30, free_host_bytes=88 * 2**30)
    assert resources["cpu_budget"] == 25
    assert resources["source_workers"] == resources["target_workers"] == resources["gpu_slots"] == 4
    assert resources["native_threads_per_worker"] == 6
    for cpu, memory in ((24, 88), (25, 32)):
        with pytest.raises(ValueError, match="floor"):
            resolve_source_resources(plan, 5, "cuda", cpu_count=cpu,
                free_device_bytes=32 * 2**30, free_host_bytes=memory * 2**30)
