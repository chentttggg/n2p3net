import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest
import torch
from threadpoolctl import threadpool_info

from research.cloud_execution import (
    CloudExecutor,
    CloudTask,
    PhaseResources,
    target_phase_resources,
)
from research.source_execution import (
    HostMemoryAdmission,
    SourceInputAdmission,
    _HostMemoryState,
    resolve_source_resources,
)


def _resources():
    return resolve_source_resources(
        {"cpu_budget": 25, "source_workers": 4, "target_workers": 4,
         "gpu_workers": 4, "target_native_threads_per_worker": 2,
         "host_memory_fraction": .9, "source_worker_memory_gib": 1,
         "target_worker_memory_gib": .25, "gpu_task_memory_gib": 4},
        4, "cuda:0", cpu_count=25, free_device_bytes=32 * 2**30,
        free_host_bytes=8 * 2**30,
    )


def _record_worker_budget(arguments):
    return {"native": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
            "native_pools": [row["num_threads"] for row in threadpool_info()],
            "output": arguments["output"]}


def test_target_policy_counts_immutable_interop_in_the_total_cpu_budget():
    resources = _resources()
    phase = target_phase_resources(resources, "cuda:0")
    assert phase.cpu_threads == phase.native_threads + phase.interop_threads == 3
    assert phase.defer_gpu
    assert 4 * phase.cpu_threads + resources["coordinator_cpu_threads"] <= 25


def test_gpu_actor_capacity_is_retained_until_retirement_and_cpu_tasks_remain_ready():
    resources = _resources()
    ledger = _HostMemoryState(resources)
    actors = [ledger.reserve_phase("calibration", 3, 2**20, 0, f"p{i}", True) for i in range(4)]
    assert all(actors)
    assert ledger.reserve_phase("calibration", 3, 2**20, 0, "waiting", True) is None
    cpu = ledger.reserve_phase("evaluation", 3, 2**20, 0, "cpu-replay", False)
    assert cpu is not None
    for i in range(4):
        assert ledger.try_activate_target_gpu(f"p{i}")["gpu_workers"] == 1
    assert ledger.record()["active_gpu_groups"] == 4
    ledger.release_phase(actors[0]["token"])
    next_actor = ledger.reserve_phase("calibration", 3, 2**20, 0, "waiting", True)
    assert next_actor is not None
    assert ledger.try_activate_target_gpu("waiting")["gpu_workers"] == 1
    for reservation in [*actors[1:], cpu, next_actor]:
        ledger.release_phase(reservation["token"])
    assert ledger.record()["active_gpu_groups"] == 0
    assert ledger.record()["reserved_cpu_threads"] == 1


def test_spawned_budget_and_scientific_parent_callback_use_their_own_claims(tmp_path):
    import multiprocessing

    resources = _resources()
    with multiprocessing.get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, resources)
        task = CloudTask("score", "tests.research.test_cloud_phase_policy:_record_worker_budget",
                         {"output": str(tmp_path / "score")}, str(tmp_path / "score"),
                         PhaseResources("evaluation", 3, 2**20, 2, interop_threads=1))
        callbacks = []

        def admit(outcome):
            callbacks.append((torch.get_num_threads(), [row["num_threads"] for row in threadpool_info()]))
            assert outcome.value["native"] == 2
            assert outcome.value["interop"] == 1
            assert all(count <= 2 for count in outcome.value["native_pools"])
            assert Path(outcome.value["output"]).name == "score"

        result = CloudExecutor(admission, max_workers=4, diagnostic_directory=tmp_path / "diagnostics",
                               identity={"kind": "resource-budget-counterexample"}, poll_seconds=.01).run(
                                   [task], on_finished=admit)
        assert result.status == "completed"
        assert callbacks and callbacks[0][0] == 1
        assert all(count == 1 for count in callbacks[0][1])
        assert admission.record()["reserved_cpu_threads"] == 1


def _escaped_native_exit(arguments):
    from research.cloud_execution import worker_compute_phase

    with worker_compute_phase("cuda:0"):
        pass
    receipt = Path(arguments["receipt"])
    code = ("import json,pathlib,sys,time,psutil; p=psutil.Process(); "
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid,'create_time':p.create_time()})); "
            "time.sleep(60)")
    subprocess.Popen([sys.executable, "-c", code, str(receipt)], start_new_session=True)
    deadline = time.monotonic() + 10
    while not receipt.is_file():
        if time.monotonic() > deadline:
            raise RuntimeError("escaped descendant failed to publish its actual identity")
        time.sleep(.01)
    os._exit(7)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux escaped process ownership")
def test_gpu_claim_is_released_only_after_native_exit_escaped_descendant_is_dead(tmp_path):
    import multiprocessing

    from research.cloud_execution import _owned_process_alive

    receipt = tmp_path / "escaped.json"
    observations = []
    resources = _resources()
    try:
        with multiprocessing.get_context("spawn").Manager() as manager:
            admission = HostMemoryAdmission(manager, resources)
            task = CloudTask("escaped", "tests.research.test_cloud_phase_policy:_escaped_native_exit",
                             {"receipt": str(receipt)}, str(tmp_path / "escaped-output"),
                             target_phase_resources(resources, "cuda:0"))

            def admit(outcome):
                identity = json.loads(receipt.read_bytes())
                try:
                    process = psutil.Process(identity["pid"])
                    alive = process.create_time() == identity["create_time"] and _owned_process_alive(process)
                except psutil.NoSuchProcess:
                    alive = False
                observations.append({"alive": alive, "resources": admission.record(), "worker": outcome.worker})

            result = CloudExecutor(admission, max_workers=4, diagnostic_directory=tmp_path / "diagnostics",
                                   identity={"kind": "escaped-tree-counterexample"}, poll_seconds=.01).run(
                                       [task], on_finished=admit)
        assert result.results["escaped"].worker["exit_code"] == 7
        assert observations and observations[0]["alive"] is False
        assert observations[0]["resources"]["active_gpu_groups"] == 0
        assert observations[0]["worker"]["process_tree_retirement"]["active_owned_processes_after_exit"] == 0
    finally:
        if receipt.exists():
            identity = json.loads(receipt.read_bytes())
            try:
                process = psutil.Process(identity["pid"])
                if process.create_time() == identity["create_time"] and _owned_process_alive(process):
                    process.kill()
                    process.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="actual tmpfs/RAM admission")
def test_parallel_nested_pending_artifacts_cannot_exceed_shared_ram_before_writing():
    import multiprocessing

    with tempfile.TemporaryDirectory(prefix="n2p3-nested-peak-", dir="/dev/shm") as scratch:
        root = Path(scratch) / "result"
        with multiprocessing.get_context("spawn").Manager() as manager:
            owner = SourceInputAdmission(manager, capacity_bytes=12 * 2**20, load_workers=1)
            entered, release = threading.Event(), threading.Event()

            def pending_writer():
                with owner.artifact_writer(root / "a.bin", 8 * 2**20):
                    entered.set()
                    if not release.wait(10):
                        raise RuntimeError("test did not release its pending writer")

            with owner.artifact_directory(root, 2**20), ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(pending_writer)
                try:
                    assert entered.wait(10)
                    with pytest.raises(ValueError, match="actual peak"):
                        with owner.artifact_writer(root / "b.bin", 8 * 2**20):
                            pass
                    assert owner.record()["peak_staging_bytes"] <= 12 * 2**20
                finally:
                    release.set()
                    future.result(timeout=10)
            assert owner.record()["free_staging_bytes"] == 12 * 2**20


def test_nested_gpu_upgrade_respects_already_waiting_target_fifo():
    ledger = _HostMemoryState(_resources())
    sources = [ledger.reserve_phase("source", 1, 2**20, 1, f"s{i}", False) for i in range(4)]
    target = ledger.reserve_phase("calibration", 3, 2**20, 0, "waiting-target", True)
    assert all(sources) and target is not None
    assert ledger.try_activate_target_gpu("waiting-target") is None
    ledger.release_phase(sources[-1]["token"])
    nested = ledger.reserve_nested(2, True, "s0")
    assert nested["workers"] == 1 and nested["gpu_workers"] == 0
    assert ledger.try_activate_target_gpu("waiting-target")["gpu_workers"] == 1
    ledger.release_nested(nested["token"])
    for reservation in [*sources[:-1], target]:
        ledger.release_phase(reservation["token"])
    assert ledger.record()["active_gpu_groups"] == 0
    assert ledger.record()["reserved_cpu_threads"] == 1


@pytest.mark.parametrize("entry", ["grouped", "prefix_selection"])
def test_actual_neural_fits_enter_compute_including_nested_prefix_callers(tmp_path, monkeypatch, entry):
    from dataclasses import replace

    import research.cloud_execution as cloud
    import transfer.ensemble_fit as fit_owner
    from tests.support.ensemble_fixtures import programs, selections
    from train.runtime import cpu_thread_budget
    from transfer.ensemble import EnsembleProgram

    class ComputeObserver:
        pending = False

        def enter_compute(self, device):
            assert device == "cpu"
            self.pending = True

    observer = ComputeObserver()
    monkeypatch.setattr(cloud, "_CURRENT_WORKER_PHASES", observer)
    original_fit = fit_owner.fit_calibration_candidate
    entered = []

    def observed_fit(*args, **kwargs):
        assert observer.pending, "neural fit started outside its resource phase"
        observer.pending = False
        entered.append(len(args[0].y))
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(fit_owner, "fit_calibration_candidate", observed_fit)
    with cpu_thread_budget(1):
        if entry == "grouped":
            spec = programs()[-1]
            roster = EnsembleProgram.equal_weights("scratch", (spec.expert_id,))
            fitted = fit_owner.fit_calibration_ensemble(selections()[:3], (spec,), roster,
                                                       calibration_strategy="grouped_temperature")
            assert fitted.temperature_record["strategy"] == "grouped_temperature"
            assert entered == [12, 8, 8, 8]
        else:
            from research.calibration_selection import (
                CalibrationProgramCandidate,
                fit_nested_prefix_selection,
            )
            from tests.support.independent_prefix import independent_prefix_case

            *_, experts, views = independent_prefix_case(tmp_path)
            spec = replace(programs()[-1], profile=experts[1].profile)
            roster = EnsembleProgram.equal_weights("scratch", (spec.expert_id,))
            candidate = CalibrationProgramCandidate("baseline", (spec,), roster)
            selected = fit_nested_prefix_selection(views.calibration[200], (candidate,),
                baseline_name="baseline", prefix_compiler=views.prefix_compiler, minimum_selection_rows=3)
            assert selected.record["final_refit_scope"]["actual_K"] == 200
            assert entered == [196]
