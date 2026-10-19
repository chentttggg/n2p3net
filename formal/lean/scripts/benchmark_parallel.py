"""Measure two real proof checks serially/parallel with shared resident pages."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import psutil
from lean_process import managed_process
from run_lean import compile_slot
from verify import ROOT, digest

FILES = ("N2P3/Covariance.lean", "N2P3/FirstSlice.lean")


def measure(workers: int, output: Path, env: dict, lean: str) -> dict:
    with ExitStack() as stack:
        return _measure(workers, output, env, lean, stack)


def _measure(workers: int, output: Path, env: dict, lean: str, stack: ExitStack) -> dict:
    started = time.perf_counter()
    pending = list(FILES)
    active = []
    completed = []
    peak_rss = peak_private = peak_uss = 0
    before = psutil.virtual_memory().available
    minimum = before
    last_uss = -1.0
    while pending or active:
        while pending and len(active) < workers:
            filename = pending.pop(0)
            log = stack.enter_context(
                (output / f"workers_{workers}_{Path(filename).stem}.log").open(
                    "w", encoding="utf-8"
                )
            )
            p = stack.enter_context(
                managed_process(
                    [lean, "-j1", filename], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            )
            active.append(
                {"p": p, "ps": psutil.Process(p.pid), "file": filename, "log": log, "cpu": 0.0}
            )
        rss = private = 0
        do_uss = time.perf_counter() - last_uss >= 1
        uss = 0
        for task in list(active):
            try:
                memory = task["ps"].memory_info()
                rss += memory.rss
                private += getattr(memory, "private", 0)
                task["cpu"] = sum(task["ps"].cpu_times()[:2])
                if do_uss:
                    uss += getattr(task["ps"].memory_full_info(), "uss", 0)
            except psutil.Error:
                pass
            if task["p"].poll() is not None:
                task["log"].close()
                completed.append(
                    {
                        "file": task["file"],
                        "exit_code": task["p"].returncode,
                        "cpu_seconds": task["cpu"],
                    }
                )
                active.remove(task)
        peak_rss, peak_private = max(peak_rss, rss), max(peak_private, private)
        if do_uss:
            peak_uss = max(peak_uss, uss)
            last_uss = time.perf_counter()
        minimum = min(minimum, psutil.virtual_memory().available)
        if active:
            time.sleep(0.1)
    return {
        "workers": workers,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": sum(c["cpu_seconds"] for c in completed),
        "available_ram_before_bytes": before,
        "minimum_host_available_ram_bytes": minimum,
        "sampled_sum_rss_peak_bytes": peak_rss,
        "sampled_sum_private_commit_peak_bytes": peak_private,
        "sampled_sum_uss_peak_bytes": peak_uss,
        "uss_sample_interval_seconds": 1,
        "checks": completed,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads(args.environment_report.read_text(encoding="utf-8"))
    if not info["passed"]:
        raise ValueError("Launcher benchmark did not pass")
    env = dict(os.environ)
    binary = Path("D:/BCI/lean-tools/lean-4.24.0-windows/bin/lean.exe")
    env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
    env["LEAN_NUM_THREADS"] = "1"
    for k, v in info["resolved_lean_paths"].items():
        if v is not None:
            env[k] = v
        else:
            env.pop(k, None)
    hashes = {f: digest(ROOT / f) for f in FILES}
    report = {
        "schema": "n2p3_lean_parallel_benchmark/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_sha256": hashes,
        "runs": [],
    }
    with compile_slot():
        for workers in (1, 2):
            result = measure(workers, args.output, env, str(binary))
            report["runs"].append(result)
            print(json.dumps(result), flush=True)
    report["sources_unchanged"] = all(digest(ROOT / f) == h for f, h in hashes.items())
    report["passed"] = report["sources_unchanged"] and all(
        c["exit_code"] == 0 for r in report["runs"] for c in r["checks"]
    )
    report["speedup"] = report["runs"][0]["wall_seconds"] / report["runs"][1]["wall_seconds"]
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
