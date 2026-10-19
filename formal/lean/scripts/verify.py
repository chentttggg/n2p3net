"""Check the pilot, audit proof assumptions, and measure actual local execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
from lean_process import managed_process

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCES = (
    "src/transfer/noise_model.py",
    "src/transfer/exact_covariance.py",
    "src/models/candidate_evidence.py",
    "src/transfer/binary.py",
    "src/data/preprocess.py",
)
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], output: Path, name: str, env: dict) -> dict:
    """Sample the process tree; report sampled RSS separately from Windows peaks."""
    log_path = output / f"{name}.log"
    started = time.perf_counter()
    cpu_by_pid: dict[int, float] = {}
    windows_peak_by_pid: dict[int, int] = {}
    sampled_tree_peak = 0
    samples = 0
    with (
        log_path.open("w", encoding="utf-8") as log,
        managed_process(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT) as process,
    ):
        parent = psutil.Process(process.pid)
        while True:
            processes = []
            try:
                processes = [parent, *parent.children(recursive=True)]
            except psutil.Error:
                pass
            rss = 0
            for child in processes:
                try:
                    memory = child.memory_info()
                    rss += memory.rss
                    cpu = child.cpu_times()
                    cpu_by_pid[child.pid] = cpu.user + cpu.system
                    windows_peak_by_pid[child.pid] = max(
                        windows_peak_by_pid.get(child.pid, 0),
                        getattr(memory, "peak_wset", 0),
                    )
                except psutil.Error:
                    pass
            samples += 1
            sampled_tree_peak = max(sampled_tree_peak, rss)
            if process.poll() is not None:
                break
            time.sleep(0.05)
    return {
        "command": command,
        "exit_code": process.returncode,
        "wall_seconds": round(time.perf_counter() - started, 4),
        "sampled_tree_cpu_seconds": round(sum(cpu_by_pid.values()), 4),
        "sampled_tree_peak_rss_bytes": sampled_tree_peak,
        "largest_observed_windows_process_peak_wset_bytes": max(
            windows_peak_by_pid.values(), default=0
        ),
        "sample_interval_seconds": 0.05,
        "samples": samples,
        "log": log_path.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lake = str(args.lake.resolve())
    env = dict(os.environ)
    env["PATH"] = str(args.lake.resolve().parent) + os.pathsep + env.get("PATH", "")
    # Limit each Lean process, independently of other applications on this host.
    env["LEAN_NUM_THREADS"] = "1"
    theorem_text = (ROOT / "N2P3Pilot.lean").read_text(encoding="utf-8")
    theorem_names = re.findall(r"^theorem (\w+)", theorem_text, re.MULTILINE)
    if re.search(r"\b(sorry|admit|axiom|native_decide)\b", theorem_text):
        raise RuntimeError("Unfinished or extra-trust construct in pilot source")
    proof_files = [
        ROOT / "N2P3Pilot.lean",
        ROOT / "Audit.lean",
        ROOT / "lakefile.toml",
        ROOT / "lake-manifest.json",
        ROOT / "lean-toolchain",
        *sorted((ROOT / "negative").glob("*.lean")),
        *sorted((ROOT / "scripts").glob("*.py")),
    ]
    report = {
        "schema": "n2p3_lean_pilot/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "selected real-arithmetic specifications; no Python/IEEE proof",
        "lean_threads": 1,
        "host": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(),
            "total_ram_bytes": psutil.virtual_memory().total,
            "available_ram_before_bytes": psutil.virtual_memory().available,
        },
        "versions": subprocess.check_output(
            [lake, "env", "lean", "--version"], cwd=ROOT, env=env, text=True
        ).strip(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_sha256": {name: digest(REPO / name) for name in SOURCES},
        "proof_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in proof_files},
        "theorems": theorem_names,
        "checks": [],
    }
    successful = True
    commands = [
        ("build", [lake, "build", "N2P3Pilot"], False),
        ("recheck", [lake, "env", "lean", "-j1", "N2P3Pilot.lean"], False),
        ("audit", [lake, "env", "lean", "-j1", "Audit.lean"], False),
        ("reject_false_offset", [lake, "env", "lean", "-j1", "negative/FalseOffset.lean"], True),
        (
            "reject_false_pd",
            [lake, "env", "lean", "-j1", "negative/FalsePositiveDefinite.lean"],
            True,
        ),
    ]
    for name, command, expect_rejection in commands:
        result = run(command, output, name, env)
        text = (output / result["log"]).read_text(encoding="utf-8")
        if expect_rejection:
            # Import/path/setup failures must never count as a soundness control.
            result["passed"] = (
                result["exit_code"] != 0
                and "unsolved goals" in text
                and "⊢ False" in text
                and "unknown module" not in text
            )
        else:
            result["passed"] = result["exit_code"] == 0
        if name == "audit" and result["passed"]:
            axioms = re.findall(r"depends on axioms: \[([^\]]*)\]", text)
            found = {a.strip() for group in axioms for a in group.split(",") if a.strip()}
            audited = re.findall(r"'N2P3Pilot\.(\w+)'", text)
            result["observed_axioms"] = sorted(found)
            result["audited_theorems"] = audited
            result["passed"] = (
                found <= ALLOWED_AXIOMS
                and set(audited) == set(theorem_names)
                and len(audited) == len(theorem_names)
            )
        report["checks"].append(result)
        successful &= result["passed"]
        print(json.dumps({"check": name, **result}), flush=True)
        if not result["passed"] and not expect_rejection:
            break
    report["source_unchanged_during_checks"] = all(
        digest(REPO / name) == expected for name, expected in report["source_sha256"].items()
    )
    report["proof_source_unchanged_during_checks"] = all(
        digest(ROOT / name) == expected for name, expected in report["proof_sha256"].items()
    )
    report["passed"] = (
        successful
        and len(report["checks"]) == len(commands)
        and report["source_unchanged_during_checks"]
        and report["proof_source_unchanged_during_checks"]
    )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
