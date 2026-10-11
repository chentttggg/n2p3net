"""PID/birth, zombie and thread boundaries of the stationary cloud adapter."""
from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from research import local_program_control as control
from research import stationary_panel as panel
from research import worker_diagnostics


def test_alive_checks_birth_status_and_keeps_live_child_of_zombie_launcher(monkeypatch):
    child = SimpleNamespace(pid=22, create_time=lambda: 20., is_running=lambda: True,
                            status=lambda: psutil.STATUS_RUNNING, children=lambda **_: [])
    zombie = SimpleNamespace(pid=11, create_time=lambda: 10., is_running=lambda: True,
                             status=lambda: psutil.STATUS_ZOMBIE, children=lambda **_: [child])
    processes = {11: zombie, 22: child}
    monkeypatch.setattr(control.psutil, "Process", processes.__getitem__)
    assert not control.actual_alive(11, 10.)
    assert not control.actual_alive(22, 21.)
    assert control.actual_alive(22, 20.)
    assert [process.pid for process in control.OwnedProcessTree(11, 10.)._alive()] == [22]


@pytest.mark.parametrize("native", [True, 0, -1, 1.5, "6"])
def test_stationary_native_budget_rejects_invalid_plan_values(native):
    with pytest.raises(ValueError, match="native thread budget"):
        panel.worker_native_threads({"resources": {"native_threads": native}})


@pytest.mark.parametrize("wrong_input", [False, True])
def test_actual_worker_consumes_native_plan_before_any_compile_or_score(tmp_path, monkeypatch, wrong_input):
    (tmp_path/"exports"/"A01").mkdir(parents=True)
    (tmp_path/"exports"/"A01"/"export.json").write_text("{}")
    monkeypatch.setattr(panel, "load_plan", lambda _, **__: {"resources": {"native_threads": 5}, "source_files": {},
        "participants": {"A01": {"original_files": {}, "input_manifest": {"path": str(tmp_path/"input.json")},
            "held": {"arrays": {}, "view": {}}, "samples_sha256": "a"*64 if wrong_input else panel.array_digest([]),
            "candidate_designs_sha256": panel.array_digest([]), "native": {"views": [{}, {}, {}, {}, {}, {}]}}}})
    monkeypatch.setattr(panel, "_export_state", lambda *_, **__: ({}, None))
    monkeypatch.setattr(panel, "_held_authority", lambda *_: {"view": {}})
    monkeypatch.setattr(panel, "PredictionDecisionView", SimpleNamespace(
        from_record=lambda *_, **__: SimpleNamespace(record=lambda: {})))
    # worker forbids these fits in its fresh process. Register their originals
    # with pytest so this in-process caller probe leaves other tests isolated.
    for owner, name in ((panel.noise_owner.FittedChannelNoise, "fit"),
                        (panel.noise_owner.FittedChannelNoise, "fit_response_given_frozen_noise"),
                        (panel.TemporalNoise, "fit")):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    class Inputs:
        def __init__(self, manifest):
            assert manifest.parent == tmp_path
        def arrays(self, *_):
            return [], []
    monkeypatch.setattr(panel, "BoundInput", Inputs)
    calls = []
    class NoScienceError(Exception):
        pass
    def limit(*, limits):
        calls.append(limits)
        raise NoScienceError
    monkeypatch.setattr(panel, "threadpool_limits", limit)
    if wrong_input:
        with pytest.raises(ValueError, match="original frozen scientific input"):
            panel.worker(tmp_path/"plan.json", "A01", None)
        assert calls == []
        return
    with pytest.raises(NoScienceError):
        panel.worker(tmp_path/"plan.json", "A01", None)
    assert calls == [5]


def test_cloud_checkpoint_marks_the_admitted_actor_without_host_cpu_pause(monkeypatch):
    facts, progress = [], []
    monkeypatch.setenv("N2P3_CLOUD_TASK_ID", "stationary:A01")
    monkeypatch.setattr(worker_diagnostics, "worker_diagnostic_phase", lambda phase, **kw: facts.append((phase, kw)))
    live = panel.LiveControl("stationary:A01", progress.append)
    live.checkpoint("candidate_diffuse_projection")
    assert progress == ["candidate_diffuse_projection"]
    assert facts[0][1]["stationary_task_id"] == "stationary:A01"
    monkeypatch.setenv("N2P3_CLOUD_TASK_ID", "stationary:A02")
    with pytest.raises(RuntimeError, match="outside its admitted"):
        live.checkpoint("stationary_root_compile")


@pytest.mark.skipif(sys.platform != "linux", reason="real unreaped fork zombie requires Linux")
def test_linux_real_zombie_releases_lease_but_live_recorded_child_blocks_resume(tmp_path):
    child = os.fork()
    if child == 0:
        os._exit(0)
    try:
        process = psutil.Process(child)
        created = process.create_time()
        deadline = time.monotonic()+5.
        while process.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert process.is_running()  # psutil's ordinary PID probe admits zombies.
        assert not control.actual_alive(child, created)
        owner = {"run_id": "zombie", "controller_pid": child, "controller_created": created}
        control.atomic_record(tmp_path/"control.json", owner | {"updated_time": time.time()})
        with pytest.raises(control.LocalProgramStoppedError, match="no longer exists"):
            control.read_control(tmp_path/"control.json", **owner)
        assert control.OwnedProcessTree(child, created).sample()["actual_processes"] == []
        control.atomic_record(tmp_path/"worker.A01.old.handle.json", {"worker_pid": child, "worker_created": created})
        control.assert_no_owned_live(tmp_path)
        # A live same-birth recorded orphan remains a blocker even though its
        # launcher is already a zombie. Never retire it merely by leader status.
        own = psutil.Process()
        control.atomic_record(tmp_path/"worker.A01.old.identity.json",
            {"actual_worker_pid": own.pid, "actual_worker_created": own.create_time()})
        with pytest.raises(control.LocalProgramStoppedError, match="descendant remains live"):
            control.assert_no_owned_live(tmp_path)
    finally:
        os.waitpid(child, 0)
