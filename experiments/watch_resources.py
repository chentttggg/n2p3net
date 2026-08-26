"""Write live process and GPU resource samples for the HTML dashboard."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "experiments" / "runs"


def _resolve_run(value: str) -> Path:
    candidate = Path(value).expanduser()
    options = [candidate]
    if not candidate.is_absolute():
        options.extend([ROOT / candidate, RUNS_ROOT / candidate])
    for path in options:
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(f"run directory not found: {value}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _process_rows(root_pid: int) -> list[dict[str, object]]:
    if os.name != "posix":
        return []
    try:
        raw = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,rss=,pcpu=,stat="],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return []
    rows: dict[int, dict[str, object]] = {}
    children: dict[int, list[int]] = {}
    for line in raw:
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            pid, ppid, rss_kib = int(fields[0]), int(fields[1]), int(fields[2])
            cpu_pct = float(fields[3])
        except ValueError:
            continue
        rows[pid] = {
            "pid": pid,
            "ppid": ppid,
            "rss_mib": round(rss_kib / 1024.0, 2),
            "cpu_pct": cpu_pct,
            "stat": fields[4],
        }
        rows[pid].update(_smaps_memory(pid))
        children.setdefault(ppid, []).append(pid)

    selected: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in selected:
            continue
        selected.add(pid)
        pending.extend(children.get(pid, ()))
    return [rows[pid] for pid in sorted(selected) if pid in rows]


def _smaps_memory(pid: int) -> dict[str, float | None]:
    """Read PSS/private memory so shared fork pages are not mistaken for leaks."""

    values: dict[str, float] = {}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {
                "Pss",
                "Private_Clean",
                "Private_Dirty",
                "Shared_Clean",
                "Shared_Dirty",
            }:
                values[key] = float(value.strip().split()[0]) / 1024.0
    except (OSError, ValueError):
        pass
    private = values.get("Private_Clean", 0.0) + values.get("Private_Dirty", 0.0)
    shared = values.get("Shared_Clean", 0.0) + values.get("Shared_Dirty", 0.0)
    return {
        "pss_mib": round(values["Pss"], 2) if "Pss" in values else None,
        "private_mib": round(private, 2),
        "shared_mib": round(shared, 2),
    }


def _gpu_row() -> dict[str, object]:
    empty = {
        "gpu_util_pct": None,
        "gpu_memory_util_pct": None,
        "gpu_memory_used_mib": None,
        "gpu_memory_total_mib": None,
    }
    if shutil.which("nvidia-smi") is None:
        return empty
    query = "utilization.gpu,utilization.memory,memory.used,memory.total"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        fields = [field.strip() for field in result.stdout.strip().split(",")]
        if len(fields) < 4:
            return empty
        values = [float(field) for field in fields[:4]]
        return {
            "gpu_util_pct": values[0],
            "gpu_memory_util_pct": values[1],
            "gpu_memory_used_mib": values[2],
            "gpu_memory_total_mib": values[3],
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        return empty


def _write_latest(path: Path, row: dict[str, object]) -> None:
    """Publish one complete sample for low-latency dashboard reads."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory or run name")
    parser.add_argument("--pid", required=True, type=int, help="parent training PID")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.pid < 1 or args.interval <= 0:
        parser.error("--pid and --interval must be positive")

    run_dir = _resolve_run(args.run)
    output = run_dir / "resources.jsonl"
    latest = run_dir / "resources.latest.json"
    while _pid_alive(args.pid):
        row = {
            "type": "resource",
            "schema": "resources.v1",
            "ts": datetime.now(UTC).isoformat(),
            "root_pid": args.pid,
            "interval_sec": args.interval,
            "processes": _process_rows(args.pid),
            **_gpu_row(),
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        _write_latest(latest, row)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
