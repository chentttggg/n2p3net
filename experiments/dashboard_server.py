"""Serve the dashboard and a guarded endpoint for stopping the active run."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
MONITOR_LEASE_SECONDS = 20.0


def _run_dir(
    runs_root: Path,
    run: object,
    extra_roots: dict[str, Path] | None = None,
) -> Path:
    if not isinstance(run, str) or not RUN_NAME.fullmatch(run):
        raise ValueError("invalid run name")
    root = runs_root.resolve()
    relative = run
    for prefix, extra_root in sorted((extra_roots or {}).items(), key=lambda item: len(item[0]), reverse=True):
        if run == prefix or run.startswith(f"{prefix}/"):
            root = extra_root.resolve()
            relative = run[len(prefix):].lstrip("/")
            break
    if not relative or not RUN_NAME.fullmatch(relative):
        raise ValueError("invalid run name")
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents or not candidate.is_dir():
        raise ValueError("run directory not found")
    return candidate


def _latest_resource(run_dir: Path) -> dict[str, object]:
    latest = run_dir / "resources.latest.json"
    try:
        if latest.is_file():
            return json.loads(latest.read_text(encoding="utf-8"))
        history = run_dir / "resources.jsonl"
        line = history.read_text(encoding="utf-8").splitlines()[-1]
        return json.loads(line)
    except (OSError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError("no resource sample with a training PID") from exc


def _command_line(pid: int) -> list[str]:
    try:
        return [part.decode(errors="replace") for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
    except OSError as exc:
        raise ValueError("training process is no longer available") from exc


def _proc_parent(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return int(fields[3])
    except (OSError, IndexError, ValueError):
        return None


def _argument_value(command: list[str], option: str) -> str | None:
    for index, argument in enumerate(command):
        if argument == option and index + 1 < len(command):
            return command[index + 1]
        prefix = f"{option}="
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return None


def _matching_pids(script_name: str, option: str, values: set[str]) -> list[int]:
    if os.name != "posix":
        return []
    matches: list[int] = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
            command = " ".join(_command_line(pid))
        except (OSError, ValueError):
            continue
        if not any(script_name in part for part in command):
            continue
        if _argument_value(command, option) in values:
            matches.append(pid)
    return matches


def _path_matches(value: str, target: Path, project_root: Path) -> bool:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        return candidate.resolve() == target.resolve()
    except OSError:
        return False


def _process_belongs_to_run(command: list[str], run_dir: Path, project_root: Path) -> bool:
    run_name = _argument_value(command, "--run-name")
    run_root = _argument_value(command, "--run-dir")
    if not run_name:
        return False
    if run_root is None:
        return run_name in {run_dir.name, run_dir.parent.name}
    root = Path(run_root).expanduser()
    if not root.is_absolute():
        root = project_root / root
    expected = (root / run_name).resolve()
    selected = run_dir.resolve()
    # Binary EEG runs store one model below the logical run directory;
    # single-model GTN runs use the logical directory itself.
    return selected == expected or expected in selected.parents


def _training_pid(run_dir: Path, project_root: Path) -> int:
    candidates: list[int] = []
    if os.name == "posix":
        for entry in Path("/proc").glob("[0-9]*"):
            try:
                pid = int(entry.name)
                command = _command_line(pid)
            except (OSError, ValueError):
                continue
            if _process_belongs_to_run(command, run_dir, project_root):
                candidates.append(pid)
    if not candidates:
        raise ValueError("training process for the selected run is not available")
    candidate_set = set(candidates)
    roots = [pid for pid in candidates if _proc_parent(pid) not in candidate_set]
    return min(roots or candidates)


def _monitor_pid(run_dir: Path, project_root: Path) -> int | None:
    values = {str(run_dir), run_dir.relative_to(project_root).as_posix()}
    matches = _matching_pids("watch_resources.py", "--run", values)
    return min(matches) if matches else None


def _descendants(pid: int) -> list[int]:
    pending = [pid]
    found: list[int] = []
    while pending:
        current = pending.pop()
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
        except OSError:
            children = []
        for child in children:
            child_pid = int(child)
            if child_pid not in found:
                found.append(child_pid)
                pending.append(child_pid)
    return found


def _run_id(run_root: Path, relative: Path, primary_root: Path, prefix: str) -> str:
    name = relative.as_posix()
    return name if run_root.resolve() == primary_root.resolve() else f"{prefix}/{name}"


def _latest_run(runs_root: Path, extra_roots: dict[str, Path] | None = None) -> str | None:
    candidates: list[Path] = []
    roots = [(runs_root, "")] + [(root, prefix) for prefix, root in (extra_roots or {}).items()]
    for root, _prefix in roots:
        root_path = root if isinstance(root, Path) else Path(root)
        for name in ("progress.jsonl", "resources.jsonl", "resources.latest.json"):
            candidates.extend(path for path in root_path.rglob(name) if path.is_file())
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    roots = [(runs_root, "")] + [(root, prefix) for prefix, root in (extra_roots or {}).items()]
    for root, prefix in roots:
        root_path = root if isinstance(root, Path) else Path(root)
        try:
            return _run_id(root_path, newest.parent.relative_to(root_path), runs_root, prefix)
        except ValueError:
            continue
    return None


def _active_run(runs_root: Path, extra_roots: dict[str, Path] | None = None) -> str | None:
    """Prefer a run with a live N2P3-Net process over stale file mtimes."""

    candidates: list[tuple[str, Path]] = []
    roots = [(runs_root, "")] + [(root, prefix) for prefix, root in (extra_roots or {}).items()]
    for root, prefix in roots:
        root_path = root if isinstance(root, Path) else Path(root)
        for name in ("progress.jsonl", "resources.jsonl", "resources.latest.json"):
            for path in root_path.rglob(name):
                if path.is_file():
                    relative = path.parent.relative_to(root_path)
                    run_id = _run_id(root_path, relative, runs_root, prefix)
                    item = (run_id, path.parent)
                    if item not in candidates:
                        candidates.append(item)
    active: list[tuple[float, str]] = []
    project_root = runs_root.parent.resolve()
    for run, run_dir in candidates:
        try:
            _training_pid(run_dir, project_root)
            newest_mtime = max(
                (
                    path.stat().st_mtime
                    for path in (
                        run_dir / "progress.jsonl",
                        run_dir / "resources.jsonl",
                        run_dir / "resources.latest.json",
                    )
                    if path.is_file()
                ),
                default=0.0,
            )
        except (OSError, ValueError):
            continue
        active.append((newest_mtime, run))
    return max(active)[1] if active else None


def _run_candidates(runs_root: Path, extra_roots: dict[str, Path] | None = None) -> list[str]:
    candidates: set[str] = set()
    roots = [(runs_root, "")] + [(root, prefix) for prefix, root in (extra_roots or {}).items()]
    for root, prefix in roots:
        root_path = root if isinstance(root, Path) else Path(root)
        for name in ("progress.jsonl", "resources.jsonl", "resources.latest.json", "record.json"):
            for path in root_path.rglob(name):
                if not path.is_file():
                    continue
                try:
                    relative = path.parent.relative_to(root_path)
                    candidates.add(_run_id(root_path, relative, runs_root, prefix))
                except ValueError:
                    continue
    return sorted(candidates)


def _progress_done(run_dir: Path) -> bool:
    progress = run_dir / "progress.jsonl"
    try:
        tail = progress.read_bytes()[-8192:].splitlines()
        if tail:
            return json.loads(tail[-1]).get("type") == "done"
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    record = run_dir / "record.json"
    if not record.is_file():
        return False
    try:
        rows = [
            json.loads(line)
            for line in progress.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        latest_manifest = next(
            (row for row in reversed(rows) if row.get("type") == "manifest"),
            None,
        )
        record_data = json.loads(record.read_text(encoding="utf-8"))
        started = latest_manifest.get("started_utc") if latest_manifest else None
        finished = record_data.get("finished_utc")
        if not started or not finished:
            return False
        try:
            return datetime.fromisoformat(str(finished).replace("Z", "+00:00")) >= datetime.fromisoformat(
                str(started).replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            return str(finished) >= str(started)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return False


def _list_runs(runs_root: Path, extra_roots: dict[str, Path] | None = None) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    project_root = runs_root.parent.resolve()
    for name in _run_candidates(runs_root, extra_roots):
        try:
            run_dir = _run_dir(runs_root, name, extra_roots)
            marker_paths = [
                run_dir / marker
                for marker in ("progress.jsonl", "resources.jsonl", "resources.latest.json", "record.json")
            ]
            updated_at = max((path.stat().st_mtime for path in marker_paths if path.is_file()), default=0.0)
            try:
                pid: int | None = _training_pid(run_dir, project_root)
            except ValueError:
                pid = None
            runs.append({
                "name": name,
                "label": run_dir.relative_to(project_root).as_posix(),
                "active": pid is not None,
                "pid": pid,
                "done": _progress_done(run_dir),
                "updated_at": updated_at,
            })
        except (OSError, ValueError):
            continue
    return sorted(
        runs,
        key=lambda item: (bool(item["active"]), float(item["updated_at"]), str(item["name"])),
        reverse=True,
    )


class DashboardHandler(SimpleHTTPRequestHandler):
    runs_root: Path
    extra_run_roots: dict[str, Path] = {}
    monitor_lock = threading.Lock()
    monitor_clients: dict[str, dict[str, object]] = {}
    monitor_workers: dict[str, subprocess.Popen[bytes]] = {}

    def _resolve_run(self, run: object) -> Path:
        return _run_dir(self.runs_root, run, self.extra_run_roots)

    def translate_path(self, path: str) -> str:  # noqa: N802
        virtual = unquote(urlsplit(path).path).lstrip("/")
        if virtual.startswith("runs/"):
            relative = virtual[len("runs/"):]
            for prefix, root in self.extra_run_roots.items():
                marker = f"{prefix}/"
                if not relative.startswith(marker):
                    continue
                root = root.resolve()
                candidate = (root / relative[len(marker):]).resolve()
                if candidate != root and root in candidate.parents:
                    return str(candidate)
        return super().translate_path(path)

    def _json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _request_port(self) -> int:
        host = self.headers.get("Host", "")
        try:
            return int(host.rsplit(":", 1)[1]) if ":" in host else int(self.server.server_address[1])
        except (IndexError, TypeError, ValueError):
            return int(self.server.server_address[1])

    @classmethod
    def _reap_monitors(cls) -> None:
        now = time.monotonic()
        with cls.monitor_lock:
            stale = [
                token
                for token, session in cls.monitor_clients.items()
                if now - float(session["last_seen"]) > MONITOR_LEASE_SECONDS
            ]
            for token in stale:
                cls.monitor_clients.pop(token, None)
            active_runs = {
                str(session["run"])
                for session in cls.monitor_clients.values()
                if now - float(session["last_seen"]) <= MONITOR_LEASE_SECONDS
            }
            for run, process in list(cls.monitor_workers.items()):
                if process.poll() is not None:
                    cls.monitor_workers.pop(run, None)
                elif run not in active_runs:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    cls.monitor_workers.pop(run, None)

    @classmethod
    def _monitor_status(cls, run: str) -> dict[str, object]:
        cls._reap_monitors()
        run_dir = _run_dir(cls.runs_root, run, cls.extra_run_roots)
        latest = run_dir / "resources.latest.json"
        sample_age: float | None = None
        if latest.is_file():
            try:
                sample_age = max(0.0, time.time() - latest.stat().st_mtime)
            except OSError:
                pass
        with cls.monitor_lock:
            process = cls.monitor_workers.get(run)
            managed_pid = process.pid if process and process.poll() is None else None
        pid = managed_pid or _monitor_pid(run_dir, cls.runs_root.parent.resolve())
        if pid is None:
            state = "offline"
        elif sample_age is not None and sample_age > MONITOR_LEASE_SECONDS:
            state = "stale"
        else:
            state = "running"
        return {
            "state": state,
            "pid": pid,
            "managed": managed_pid is not None,
            "sample_age_sec": sample_age,
        }

    @classmethod
    def _claim_monitor(cls, run: str) -> tuple[str, dict[str, object]]:
        run_dir = _run_dir(cls.runs_root, run, cls.extra_run_roots)
        project_root = cls.runs_root.parent.resolve()
        cls._reap_monitors()
        existing_pid = _monitor_pid(run_dir, project_root)
        state = "running"
        if existing_pid is None:
            root_pid = _training_pid(run_dir, project_root)
            script = Path(__file__).with_name("watch_resources.py")
            process = subprocess.Popen(
                [sys.executable, str(script), "--run", str(run_dir), "--pid", str(root_pid)],
                cwd=str(project_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            with cls.monitor_lock:
                cls.monitor_workers[run] = process
            existing_pid = process.pid
            state = "started"
        token = secrets.token_urlsafe(24)
        with cls.monitor_lock:
            cls.monitor_clients[token] = {"run": run, "last_seen": time.monotonic()}
        return token, {
            "state": state,
            "pid": existing_pid,
            "run": run,
            "lease_seconds": MONITOR_LEASE_SECONDS,
            "sample": cls._monitor_status(run),
            "run_dir": str(run_dir),
        }

    @classmethod
    def _heartbeat_monitor(cls, run: str, token: str) -> dict[str, object]:
        cls._reap_monitors()
        with cls.monitor_lock:
            session = cls.monitor_clients.get(token)
            if not session or session.get("run") != run:
                raise ValueError("monitor lease is missing or expired")
            session["last_seen"] = time.monotonic()
        return cls._monitor_status(run)

    @classmethod
    def _release_monitor(cls, run: str, token: str) -> None:
        with cls.monitor_lock:
            session = cls.monitor_clients.get(token)
            if session and session.get("run") == run:
                cls.monitor_clients.pop(token, None)
        cls._reap_monitors()

    @classmethod
    def _stop_managed_monitors(cls) -> None:
        with cls.monitor_lock:
            workers = list(cls.monitor_workers.items())
            cls.monitor_workers.clear()
            cls.monitor_clients.clear()
        for _, process in workers:
            if process.poll() is None:
                process.terminate()

    def do_GET(self) -> None:  # noqa: N802
        self._reap_monitors()
        path = urlsplit(self.path).path
        if path == "/api/runs":
            runs = _list_runs(self.runs_root, self.extra_run_roots)
            self._json(
                200,
                {
                    "ok": True,
                    "runs": runs,
                    "active_run": next((run["name"] for run in runs if run["active"]), None),
                    "latest_run": runs[0]["name"] if runs else None,
                },
            )
            return
        if path == "/api/status":
            active = self.runs_root.parent / "active_run.txt"
            active_file_run = active.read_text(encoding="utf-8").strip() if active.is_file() else None
            detected_active_run = _active_run(self.runs_root, self.extra_run_roots)
            self._json(
                200,
                {
                    "ok": True,
                    "port": self._request_port(),
                    "active_run": detected_active_run or active_file_run or None,
                    "latest_run": detected_active_run or _latest_run(self.runs_root, self.extra_run_roots),
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path not in {"/api/kill", "/api/monitor/claim", "/api/monitor/heartbeat", "/api/monitor/release"}:
            self._json(404, {"ok": False, "error": "unknown endpoint"})
            return
        if path == "/api/monitor/claim":
            try:
                payload = self._read_payload()
                run = payload.get("run") if isinstance(payload, dict) else None
                token, status = self._claim_monitor(run)
                self._json(200, {"ok": True, "client_id": token, **status})
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json(409, {"ok": False, "error": str(exc)})
            return
        if path == "/api/monitor/heartbeat":
            try:
                payload = self._read_payload()
                run = payload.get("run") if isinstance(payload, dict) else None
                token = payload.get("client_id") if isinstance(payload, dict) else None
                status = self._heartbeat_monitor(run, token)
                self._json(200, {"ok": True, **status})
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json(409, {"ok": False, "error": str(exc)})
            return
        if path == "/api/monitor/release":
            try:
                payload = self._read_payload()
                run = payload.get("run") if isinstance(payload, dict) else None
                token = payload.get("client_id") if isinstance(payload, dict) else None
                self._release_monitor(run, token)
                self._json(200, {"ok": True})
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json(409, {"ok": False, "error": str(exc)})
            return
        if self.headers.get("X-Dashboard-Confirm") != "yes":
            self._json(403, {"ok": False, "error": "confirmation header required"})
            return
        try:
            payload = self._read_payload()
            run = payload.get("run") if isinstance(payload, dict) else None
            run_dir = self._resolve_run(run)
            resource = _latest_resource(run_dir)
            pid = int(resource["root_pid"])
            command = _command_line(pid)
            if not _process_belongs_to_run(command, run_dir, self.runs_root.parent.resolve()):
                raise ValueError("PID is not the selected N2P3-Net run")
            if pid == os.getpid():
                raise ValueError("refusing to stop dashboard server")
            targets = _descendants(pid) + [pid]
            signaled: list[int] = []
            for target in reversed(targets):
                try:
                    os.kill(target, signal.SIGTERM)
                    signaled.append(target)
                except ProcessLookupError:
                    pass
            self._json(200, {"ok": True, "run": run, "pid": pid, "signaled": signaled})
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json(409, {"ok": False, "error": str(exc)})

    def _read_payload(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 4096:
            raise ValueError("invalid request size")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8812)
    args = parser.parse_args()
    directory = args.directory.resolve()
    runs_root = directory / "runs"

    class Handler(DashboardHandler):
        pass

    Handler.runs_root = runs_root
    Handler.extra_run_roots = {"tmp": directory.parent / "tmp"}
    server = ThreadingHTTPServer((args.bind, args.port), lambda *a, **kw: Handler(*a, directory=str(directory), **kw))
    reaper_stop = threading.Event()

    def reap_loop() -> None:
        while not reaper_stop.wait(5.0):
            Handler._reap_monitors()

    reaper = threading.Thread(target=reap_loop, name="dashboard-monitor-reaper", daemon=True)
    reaper.start()
    print(f"[dashboard] http://{args.bind}:{args.port}/dashboard.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        reaper_stop.set()
        reaper.join(timeout=1)
        Handler._stop_managed_monitors()
        server.server_close()


if __name__ == "__main__":
    main()
