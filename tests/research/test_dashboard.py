from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.dashboard_server import _list_runs, _run_dir
from tests import PROJECT_ROOT
from tests.support.io import write_json


def test_dashboard_guards_origin_host_and_static_scope(tmp_path):
    from experiments.dashboard_server import DashboardHandler

    directory = tmp_path / "experiments"
    run = directory / "runs" / "declared"
    run.mkdir(parents=True)
    (directory / "dashboard.html").write_text("dashboard", encoding="utf-8")
    (directory / "private.py").write_text("secret", encoding="utf-8")
    (run / "progress.jsonl").write_text('{}\n', encoding="utf-8")
    (run / "model.pt").write_bytes(b"private model")
    temp_run = tmp_path / "tmp" / "runs" / "temporary"
    temp_run.mkdir(parents=True)
    (temp_run / "record.json").write_text('{}', encoding="utf-8")
    (tmp_path / "tmp" / "secret.json").write_text('private', encoding="utf-8")
    claims = []

    class Handler(DashboardHandler):
        runs_root = directory / "runs"
        extra_run_roots = {"tmp/runs": tmp_path / "tmp" / "runs"}

        @classmethod
        def _claim_monitor(cls, selected):
            claims.append(selected)
            return "lease", {"state": "started"}

    server = ThreadingHTTPServer(("127.0.0.1", 0),
        lambda *args, **kwargs: Handler(*args, directory=str(directory), **kwargs))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    port = server.server_address[1]

    def request(method, path, *, host=None, origin=None, content_type="application/json", fetch_site=None):
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        headers = {"Host": host or f"127.0.0.1:{port}", "Content-Type": content_type}
        if origin is not None:
            headers["Origin"] = origin
        if fetch_site is not None:
            headers["Sec-Fetch-Site"] = fetch_site
        connection.request(method, path, body='{"run":"declared"}' if method == "POST" else None, headers=headers)
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    try:
        assert request("GET", "/dashboard.html") == (200, b"dashboard")
        assert request("GET", "/private.py")[0] == 404
        assert request("GET", "/runs/declared/model.pt")[0] == 404
        assert request("GET", "/runs/declared/")[0] == 404
        assert request("GET", "/runs/declared/progress.jsonl")[0] == 200
        assert request("GET", "/runs/tmp/runs/temporary/record.json")[0] == 200
        assert request("GET", "/runs/tmp/secret.json")[0] == 404
        assert request("GET", "/api/status", host="attacker.example")[0] == 403
        assert request("GET", "/api/status", host="127.0.0.1:0")[0] == 403
        assert request("POST", "/api/monitor/claim")[0] == 403
        assert request("POST", "/api/monitor/claim", origin="http://attacker.example")[0] == 403
        assert request("POST", "/api/monitor/claim", origin=f"http://127.0.0.1:{port}", content_type="text/plain")[0] == 403
        assert request("POST", "/api/monitor/claim", origin=f"http://127.0.0.1:{port}", fetch_site="cross-site")[0] == 403
        assert not claims
        assert request("POST", "/api/monitor/claim", origin=f"http://127.0.0.1:{port}")[0] == 200
        assert claims == ["declared"]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_dashboard_discovers_current_nested_run_and_done_state(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "dashboard-smoke" / "eegnet"
    run_dir.mkdir(parents=True)
    started = "2026-08-28T01:00:00+00:00"
    finished = "2026-08-28T01:01:00+00:00"
    manifest = {
        "type": "manifest",
        "run_name": "dashboard-smoke",
        "total_folds": 2,
        "started_utc": started,
    }
    (run_dir / "progress.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                manifest,
                {"type": "fold", "fold": 0, "n_folds_done": 1},
                {"type": "done", "ts": finished},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(run_dir / "record.json", {"finished_utc": finished})

    assert _run_dir(tmp_path / "runs", "dashboard-smoke/eegnet") == run_dir.resolve()
    runs = _list_runs(tmp_path / "runs")

    assert len(runs) == 1
    assert runs[0]["name"] == "dashboard-smoke/eegnet"
    assert runs[0]["done"] is True
    assert runs[0]["active"] is False


def test_dashboard_run_resolution_rejects_path_escape(tmp_path) -> None:
    run_root = tmp_path / "runs"
    (run_root / "valid").mkdir(parents=True)

    with pytest.raises(ValueError):
        _run_dir(run_root, "../valid")


def test_resource_liveness_is_read_only_for_live_and_exited_processes(monkeypatch):
    from experiments import watch_resources

    def forbidden_signal(*_args, **_kwargs):
        raise AssertionError("liveness checks must not send process signals")

    monkeypatch.setattr(watch_resources.os, "kill", forbidden_signal)
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        assert watch_resources._pid_alive(child.pid)
        assert child.poll() is None
        child.communicate(timeout=10)
        assert child.returncode == 0
        assert not watch_resources._pid_alive(child.pid)
        assert not watch_resources._pid_alive(0)
        assert not watch_resources._pid_alive(-1)
    finally:
        if child.poll() is None:
            child.communicate(timeout=10)


@pytest.mark.parametrize("arguments", [
    ["--run", "/tmp/a run"], ["--run=/tmp/a run"],
])
def test_existing_resource_monitor_is_matched_by_argv(monkeypatch, arguments):
    from experiments import dashboard_server as server

    process_directory = SimpleNamespace(glob=lambda _: [SimpleNamespace(name="12345")])
    monkeypatch.setattr(server, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(server, "Path", lambda value: process_directory if value == "/proc" else Path(value))
    monkeypatch.setattr(server, "_command_line", lambda _: ["python", "/tmp/tool path/watch_resources.py", *arguments])
    assert server._matching_pids("watch_resources.py", "--run", {"/tmp/a run"}) == [12345]
    assert server._matching_pids("watch_resources.py", "--run", {"/tmp/other"}) == []


def test_concurrent_monitor_leases_share_one_worker_until_last_release(tmp_path, monkeypatch):
    from experiments import dashboard_server as server

    run_root = tmp_path / "runs"
    (run_root / "run").mkdir(parents=True)
    handler = server.DashboardHandler
    monkeypatch.setattr(handler, "runs_root", run_root, raising=False)
    monkeypatch.setattr(handler, "monitor_clients", {})
    monkeypatch.setattr(handler, "monitor_workers", {})
    monkeypatch.setattr(handler, "monitor_lock", threading.Lock())
    monkeypatch.setattr(server, "_monitor_pid", lambda *_: None)
    monkeypatch.setattr(server, "_training_pid", lambda *_: 100)
    workers = []

    class Worker:
        def __init__(self, *_args, **_kwargs):
            self.pid = len(workers) + 200
            self.exit_code = None
            workers.append(self)

        def poll(self):
            return self.exit_code

        def terminate(self):
            self.exit_code = 0

        def wait(self, **_kwargs):
            return self.exit_code

    monkeypatch.setattr(server.subprocess, "Popen", Worker)
    with ThreadPoolExecutor(max_workers=4) as pool:
        leases = list(pool.map(lambda _: handler._claim_monitor("run"), range(4)))
    assert len(workers) == 1
    assert {status["pid"] for _, status in leases} == {workers[0].pid}
    for token, _ in leases[:-1]:
        handler._release_monitor("run", token)
        assert workers[0].poll() is None
    handler._release_monitor("run", leases[-1][0])
    assert workers[0].poll() == 0
    assert not handler.monitor_workers


def test_dashboard_numeric_conversion_preserves_missing_values():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard JavaScript regression")
    source = (PROJECT_ROOT / "experiments/dashboard.html").read_text(encoding="utf-8")
    function = re.search(r"^function finiteNumber\(value\) \{[\s\S]*?^\}", source, re.MULTILINE)
    assert function is not None
    memory_function = re.search(r"^function processMemoryGiB\(row, field\) \{[\s\S]*?^\}", source, re.MULTILINE)
    assert memory_function is not None
    script = function.group() + memory_function.group() + """
const missing = [null, undefined, '', '   ', false, true, [], {}, NaN, Infinity];
const present = [0, '0', -1, ' 2.5 ', 0.75];
const memory = [[], [null], [0], [1024, 512], [1024, null]].map(values =>
  processMemoryGiB({processes: values.map(value => ({rss_mib: value}))}, 'rss_mib'));
console.log(JSON.stringify({missing: missing.map(finiteNumber), present: present.map(finiteNumber), memory}));
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    values = json.loads(result.stdout)
    assert values["missing"] == [None] * 10
    assert values["present"] == [0, 0, -1, 2.5, 0.75]
    assert values["memory"] == [None, None, 0, 1.5, None]


def test_gpu_sample_aggregates_every_nvidia_smi_line():
    from experiments.watch_resources import _parse_gpu_query

    sample = _parse_gpu_query("50, 10, 1000, 24000\n70, 30, 3000, 24000\n")
    assert sample == {"gpu_util_pct": 60.0, "gpu_memory_util_pct": 20.0, "gpu_memory_used_mib": 4000.0,
                      "gpu_memory_total_mib": 48000.0, "gpu_count": 2}
    assert _parse_gpu_query("")["gpu_util_pct"] is None
