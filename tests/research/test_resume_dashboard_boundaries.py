"""Narrow source-resume and dashboard startup controls; zero model fits."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.deep import DeepConfig
from tests import PROJECT_ROOT


def _module(name, relative):
    root = Path(os.environ.get("N2P3_BOUNDARY_CANDIDATE", PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resume(tmp_path, monkeypatch):
    owner = _module("resume_boundary_under_review", "src/research/source_resume.py")
    resume = owner.SourceChainResume(None)
    resume.root = tmp_path
    resume.record_path = tmp_path / "source_chain_record.json"
    resume.inventory = {"files": {}}
    resume._source_snapshots = {"c" * 64}
    config = DeepConfig(fixed_updates=1, compile_mode=None)
    ledger = {"training_participant_keys": ["person"]}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"original checkpoint image")
    resume.inventory["files"][checkpoint.name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    task = {"key": "original", "refit_key": ["outer", 7, 1], "checkpoint": str(checkpoint),
            "config": asdict(config), "fit_ledger": ledger, "source_cache_sha256": "a" * 64}
    resume.tasks = {"original": task}
    contract = SimpleNamespace(source_cache_sha256="a" * 64, source_snapshot_sha256="c" * 64,
                               training_participant_keys=["person"], architecture={"same": True})
    seen = []

    def load(source):
        seen.append(source.read() if hasattr(source, "read") else Path(source).read_bytes())
        return {"source_snapshot_sha256": "c" * 64}

    monkeypatch.setattr(owner, "load_checkpoint_payload", load)
    monkeypatch.setattr(owner, "checkpoint_training_contract", lambda payload: contract)
    return owner, resume, config, ledger, task, checkpoint, seen


def test_resume_rejects_same_size_preserved_mtime_rewrite(tmp_path, monkeypatch):
    _, resume, _, _, _, checkpoint, _ = _resume(tmp_path, monkeypatch)
    resume.verify(checkpoint)
    old = checkpoint.stat()
    content = checkpoint.read_bytes()
    checkpoint.write_bytes(b"X" + content[1:])
    os.utime(checkpoint, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) == (old.st_size, old.st_mtime_ns)
    with pytest.raises(ValueError, match="artifact changed"):
        resume.verify(checkpoint)


def test_refit_selects_the_matching_complete_request_independent_of_dict_order(tmp_path, monkeypatch):
    _, resume, config, ledger, task, _, seen = _resume(tmp_path, monkeypatch)
    wrong = {**task, "key": "wrong", "config": {**task["config"], "seed": 99}}
    resume.tasks = {"wrong": wrong, "original": task}
    result = resume.acquire("request", config, ledger, "a" * 64, refit_key=("outer", 7, 1))
    assert result["source_training_reuse"]["origin_task"] == "original"
    assert result["physical_optimizer_updates"] == 0 and len(seen) == 1


def test_refit_rejects_distinct_producers_and_allows_identical_byte_aliases(tmp_path, monkeypatch):
    _, resume, config, ledger, task, checkpoint, _ = _resume(tmp_path, monkeypatch)
    other = tmp_path / "other.pt"
    other.write_bytes(checkpoint.read_bytes() + b"different")
    resume.inventory["files"][other.name] = hashlib.sha256(other.read_bytes()).hexdigest()
    second = {**task, "key": "second", "checkpoint": str(other)}
    resume.tasks["second"] = second
    with pytest.raises(ValueError, match="multiple distinct checkpoint producers"):
        resume.acquire("request", config, ledger, "a" * 64, refit_key=("outer", 7, 1))
    other.write_bytes(checkpoint.read_bytes())
    resume.inventory["files"][other.name] = resume.inventory["files"][checkpoint.name]
    result = resume.acquire("request", config, ledger, "a" * 64, refit_key=("outer", 7, 1))
    assert result["source_training_reuse"]["origin_task"] == "original"


def test_resume_decodes_the_same_verified_checkpoint_image_once(tmp_path, monkeypatch):
    owner, resume, config, ledger, _, checkpoint, seen = _resume(tmp_path, monkeypatch)
    real = owner.read_verified_bytes
    reads = []

    def frozen_read(path, expected):
        reads.append(path)
        return real(path, expected)

    monkeypatch.setattr(owner, "read_verified_bytes", frozen_read)
    result = resume.acquire("request", config, ledger, "a" * 64, refit_key=("outer", 7, 1))
    assert reads == [checkpoint] and seen == [checkpoint.read_bytes()]
    assert result["source_training_reuse"]["origin_checkpoints"]["1"]["sha256"] == hashlib.sha256(seen[0]).hexdigest()


def test_dashboard_port_exhaustion_is_explicit(monkeypatch):
    owner = _module("dashboard_boundary_ports", "experiments/run_with_dashboard.py")

    class Busy:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            raise OSError("occupied")

    monkeypatch.setattr(owner.socket, "socket", lambda *args: Busy())
    with pytest.raises(OSError, match="all unavailable"):
        owner._free_port()


def test_dashboard_startup_exit_never_launches_experiment_or_publishes_url(tmp_path, monkeypatch, capsys):
    owner = _module("dashboard_boundary_start", "experiments/run_with_dashboard.py")
    server = SimpleNamespace(stdout=io.StringIO("startup error\n"), returncode=7, poll=lambda: 7,
                             wait=lambda **kwargs: 7, terminate=lambda: None, kill=lambda: None)
    calls = []
    monitored = []
    monkeypatch.setattr(owner, "ROOT", tmp_path)
    monkeypatch.setattr(owner, "EXPS", tmp_path / "experiments")
    monkeypatch.setattr(owner, "_free_port", lambda: 8812)
    monkeypatch.setattr(owner.subprocess, "Popen", lambda *args, **kwargs: server)
    monkeypatch.setattr(owner.subprocess, "call", lambda *args, **kwargs: calls.append(args) or 0)
    monkeypatch.setattr(owner, "_start_active_run_monitor", lambda stop: monitored.append(True) or SimpleNamespace(join=lambda **kwargs: None))
    monkeypatch.setattr(owner.sys, "argv", ["run_with_dashboard.py", "--", "fake-experiment"])
    with pytest.raises(SystemExit) as result:
        owner.main()
    assert result.value.code == 1 and calls == [] and monitored == []
    output = capsys.readouterr()
    assert "http://" not in output.out and "exited during startup" in output.err


def test_dashboard_readiness_requires_its_child_and_http_response(monkeypatch):
    owner = _module("dashboard_boundary_readiness", "experiments/run_with_dashboard.py")
    ready = threading.Event()
    ready.set()
    server = SimpleNamespace(poll=lambda: None)
    requests = []

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(owner, "urlopen", lambda request, **kwargs: requests.append(request) or Response())
    owner._wait_ready(server, ready, 8812, timeout=.1)
    assert len(requests) == 1 and requests[0].method == "HEAD"


def test_ready_dashboard_preserves_experiment_exit_and_cleans_up(tmp_path, monkeypatch, capsys):
    owner = _module("dashboard_boundary_success", "experiments/run_with_dashboard.py")
    state = {"code": None, "terminated": False}

    def terminate():
        state.update(code=0, terminated=True)

    server = SimpleNamespace(stdout=io.StringIO("[dashboard] http://127.0.0.1:8812/dashboard.html\n"),
        poll=lambda: state["code"], terminate=terminate, wait=lambda **kwargs: 0, kill=terminate)
    owner.ROOT, owner.EXPS = tmp_path, tmp_path / "experiments"
    calls = []
    monkeypatch.setattr(owner, "_free_port", lambda: 8812)
    monkeypatch.setattr(owner.subprocess, "Popen", lambda *args, **kwargs: server)
    monkeypatch.setattr(owner.subprocess, "call", lambda *args, **kwargs: calls.append(args) or 23)
    monkeypatch.setattr(owner, "_start_active_run_monitor", lambda stop: SimpleNamespace(join=lambda **kwargs: None))

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(owner, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(owner.sys, "argv", ["run_with_dashboard.py", "--", "fake-experiment"])
    with pytest.raises(SystemExit) as result:
        owner.main()
    assert result.value.code == 23 and len(calls) == 1 and state["terminated"]
    assert "http://localhost:8812/dashboard.html" in capsys.readouterr().out


def test_unresponsive_dashboard_has_a_bounded_startup_deadline(monkeypatch):
    owner = _module("dashboard_boundary_deadline", "experiments/run_with_dashboard.py")
    times = iter((0., 0., 1.))
    monkeypatch.setattr(owner.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(owner.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="startup deadline"):
        owner._wait_ready(SimpleNamespace(poll=lambda: None), threading.Event(), 8812, timeout=.1)
