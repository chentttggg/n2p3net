"""Execution failures cannot be mistaken for scientific numerical failures."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from research.local_program_control import (
    LocalProgramStoppedError,
    LocalResourceBudget,
    OwnedProcessTree,
    assert_no_owned_live,
    atomic_record,
    read_control,
    score_checkpoints,
)


def test_control_requires_one_actual_controller_identity_and_freshness(tmp_path):
    current = psutil.Process()
    owner = {"run_id": "accepted", "controller_pid": os.getpid(), "controller_created": current.create_time()}
    path = tmp_path/"control.json"
    atomic_record(path, owner | {"updated_time": time.time()})
    assert read_control(path, **owner)["run_id"] == "accepted"
    # Individually allowed run IDs and PID/birth sets cannot be mixed into a
    # new identity. This is the prior controller's coherent tuple counterexample.
    atomic_record(path, owner | {"run_id": "parent", "updated_time": time.time()})
    with pytest.raises(LocalProgramStoppedError, match="another controller attempt"):
        read_control(path, **owner)
    atomic_record(path, owner | {"controller_created": current.create_time()+1., "updated_time": time.time()})
    with pytest.raises(LocalProgramStoppedError, match="another controller attempt"):
        read_control(path, **owner)
    atomic_record(path, owner | {"updated_time": time.time()-30.})
    with pytest.raises(LocalProgramStoppedError, match="stale"):
        read_control(path, **owner)
    for updated in (time.time()+3600., float("nan")):
        path.write_text(json.dumps(owner | {"updated_time": updated}), encoding="utf-8")
        with pytest.raises(LocalProgramStoppedError, match="timestamp"):
            read_control(path, **owner)


def test_resource_budget_uses_actual_growth_ram_pause_and_cpu_hysteresis():
    limits = {"max_workers": 2, "worker_rss_estimate_bytes": 100, "growth_safety_factor": 1.2,
        "memory_reserve_bytes": 20, "resume_cpu_threshold_percent": 60.,
        "worker_cpu_threshold_percent": 72., "launch_cpu_threshold_percent": 60., "required_quiet_samples": 3}
    budget = LocalResourceBudget(limits)
    assert budget.sample(50., 500., [])["pause"]
    assert budget.sample(50., 500., [])["pause"]
    assert budget.sample(50., 500., [])["launch"]
    grown = budget.sample(50., 250., [{"rss_bytes": 200}])
    assert grown["estimated_worker_peak_bytes"] == 240
    assert grown["existing_worker_growth_bytes"] == 40
    assert not grown["launch"]
    assert budget.sample(50., 50., [{"rss_bytes": 200}])["memory_pause"]
    assert budget.sample(75., 1000., [{"rss_bytes": 200}])["cpu_pause"]
    assert budget.sample(65., 1000., [{"rss_bytes": 200}])["cpu_pause"]
    assert budget.sample(50., 1000., [{"rss_bytes": 200}])["cpu_pause"]
    assert budget.sample(50., 1000., [{"rss_bytes": 200}])["cpu_pause"]
    assert budget.sample(50., 1000., [{"rss_bytes": 200}])["launch"]


def test_actual_launcher_descendant_rss_and_owned_cleanup_are_complete(tmp_path):
    ready = tmp_path/"ready.txt"
    child_source = "import time; from pathlib import Path; data=bytearray(24*1024**2); Path("+repr(str(ready))+").write_text('ready'); time.sleep(30)"
    launcher_source = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',"+repr(child_source)+"]); p.wait()"
    process = subprocess.Popen([sys.executable, "-c", launcher_source],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    tree = OwnedProcessTree(process.pid, psutil.Process(process.pid).create_time())
    try:
        deadline = time.monotonic()+10.
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.05)
        sample = tree.sample()
        assert len(sample["actual_process_pids"]) >= 2
        assert sample["rss_bytes"] > 24*1024**2
    finally:
        remaining = tree.stop()
        process.wait(timeout=3.)
    assert remaining["actual_processes"] == []


def test_dead_launcher_with_recorded_live_child_blocks_resume(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    created = psutil.Process(child.pid).create_time()
    try:
        atomic_record(tmp_path/"worker.A01.old.handle.json", {"worker_pid": child.pid, "worker_created": created-10.})
        atomic_record(tmp_path/"worker.A01.old.identity.json", {"actual_worker_pid": child.pid, "actual_worker_created": created})
        with pytest.raises(LocalProgramStoppedError, match="descendant remains live"):
            assert_no_owned_live(tmp_path)
    finally:
        tree = OwnedProcessTree(child.pid, created)
        assert tree.stop()["actual_processes"] == []
        child.wait(timeout=3.)
    assert_no_owned_live(tmp_path)


def test_checkpoint_failure_restores_original_owner_and_is_not_numeric_unavailable():
    phases, completed = [], []
    def projection(value):
        return value+1
    def resolution(value):
        return value+2
    owner = SimpleNamespace(_project_diffuse_arrays=projection, _conditional_score_resolution=resolution)
    class Control:
        def checkpoint(self, phase):
            phases.append(phase)
            if len(phases) > 1:
                raise LocalProgramStoppedError("interrupted")
    with pytest.raises(LocalProgramStoppedError, match="interrupted"), score_checkpoints(owner, Control(), completed.append):
        assert owner._project_diffuse_arrays(3) == 4
        owner._conditional_score_resolution(3)
    assert completed == []
    assert owner._project_diffuse_arrays is projection and owner._conditional_score_resolution is resolution
