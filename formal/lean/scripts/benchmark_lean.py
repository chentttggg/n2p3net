"""Compare actual same-source Lean checks, launcher overhead and thread counts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
from run_lean import compile_slot
from verify import ROOT, digest, run


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    binaries = Path("D:/BCI/lean-tools/lean-4.24.0-windows/bin")
    lake, lean = str(binaries / "lake.exe"), str(binaries / "lean.exe")
    env = dict(os.environ)
    env["PATH"] = str(binaries) + os.pathsep + env.get("PATH", "")
    env["LEAN_NUM_THREADS"] = "1"
    source = ROOT / "N2P3/Covariance.lean"
    before = digest(source)
    report = {
        "schema": "n2p3_lean_launcher_benchmark/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "source": "N2P3/Covariance.lean",
        "source_sha256": before,
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_bytes": psutil.virtual_memory().total,
        "checks": [],
    }
    with compile_slot():
        started = time.perf_counter()
        script = "import os,json;print(json.dumps({k:os.environ.get(k) for k in ['LEAN_PATH','LEAN_SRC_PATH','LEAN_SYSROOT']}))"
        paths = json.loads(
            subprocess.check_output(
                [lake, "env", sys.executable, "-c", script], cwd=ROOT, env=env, text=True
            )
        )
        report["resolve_environment_seconds"] = time.perf_counter() - started
        direct_env = dict(env)
        for key, value in paths.items():
            if value is not None:
                direct_env[key] = value
            else:
                direct_env.pop(key, None)
        report["resolved_lean_paths"] = paths
        commands = [
            ("lake_1_a", [lake, "env", "lean", "-j1", "N2P3/Covariance.lean"], env),
            ("direct_1_a", [lean, "-j1", "N2P3/Covariance.lean"], direct_env),
            ("direct_2", [lean, "-j2", "N2P3/Covariance.lean"], direct_env),
            ("direct_1_b", [lean, "-j1", "N2P3/Covariance.lean"], direct_env),
            ("lake_1_b", [lake, "env", "lean", "-j1", "N2P3/Covariance.lean"], env),
        ]
        for name, command, current_env in commands:
            mem = psutil.virtual_memory()
            check = run(command, output, name, current_env)
            check["available_ram_before_bytes"] = mem.available
            check["name"] = name
            check["diagnostics_empty"] = (
                not (output / check["log"]).read_text(encoding="utf-8").strip()
            )
            report["checks"].append(check)
            print(json.dumps(check), flush=True)
            if check["exit_code"]:
                break
    report["source_unchanged"] = digest(source) == before
    report["passed"] = (
        len(report["checks"]) == 5
        and all(c["exit_code"] == 0 and c["diagnostics_empty"] for c in report["checks"])
        and report["source_unchanged"]
    )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
