"""Test safe namespace-disjoint Lean import-path ordering on actual proof checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_lean import compile_slot
from verify import ROOT, digest, run


def reordered_paths(value: str, core: Path) -> tuple[str, dict]:
    original = [Path(p).resolve() for p in value.split(os.pathsep) if p]
    if core.resolve() not in original:
        original.append(core.resolve())
    skipped = [str(p) for p in original if not p.is_dir()]
    original = [p for p in original if p.is_dir()]
    ownership: dict[str, set[str]] = {}
    counts = {}
    for p in original:
        for child in p.iterdir():
            prefix = (
                child.name if child.is_dir() else child.stem if child.suffix == ".olean" else None
            )
            if prefix:
                ownership.setdefault(prefix, set()).add(str(p))
        counts[str(p)] = len(list(p.rglob("*.olean")))
    overlaps = {k: sorted(v) for k, v in ownership.items() if len(v) > 1}
    if overlaps:
        raise ValueError(f"Namespace prefixes overlap; preserve original resolution: {overlaps}")
    ordered = sorted(original, key=lambda p: counts[str(p)], reverse=True)
    return os.pathsep.join(map(str, ordered)), {
        "original": list(map(str, original)),
        "reordered": list(map(str, ordered)),
        "compiled_module_counts": counts,
        "duplicate_top_level_namespaces": overlaps,
        "same_module_resolution": True,
        "skipped_nonexistent_search_paths": skipped,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.environment_report.read_text(encoding="utf-8"))
    if not original["passed"]:
        raise ValueError("Prior benchmark incomplete")
    binary = Path("D:/BCI/lean-tools/lean-4.24.0-windows/bin/lean.exe")
    env = dict(os.environ)
    env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
    env["LEAN_NUM_THREADS"] = "1"
    for k, v in original["resolved_lean_paths"].items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    old_path = env["LEAN_PATH"]
    new_path, ordering = reordered_paths(old_path, binary.parent.parent / "lib/lean")
    env["LEAN_PATH"] = new_path
    source = ROOT / "N2P3/Covariance.lean"
    h = digest(source)
    report = {
        "schema": "n2p3_lean_path_benchmark/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_sha256": h,
        "paths": ordering,
        "resolved_lean_paths": {
            k: env.get(k) for k in ["LEAN_PATH", "LEAN_SRC_PATH", "LEAN_SYSROOT"]
        },
        "checks": [],
    }
    with compile_slot():
        cases = ("original", "ordered", "original", "ordered") if args.paired else ("ordered",) * 2
        for i, case in enumerate(cases):
            env["LEAN_PATH"] = old_path if case == "original" else new_path
            result = run(
                [str(binary), "-j1", "N2P3/Covariance.lean"], args.output, f"{case}_{i}", env
            )
            result["path_order"] = case
            result["diagnostics_empty"] = (
                not (args.output / result["log"]).read_text(encoding="utf-8").strip()
            )
            report["checks"].append(result)
            print(json.dumps(result), flush=True)
    report["source_unchanged"] = digest(source) == h
    report["passed"] = report["source_unchanged"] and all(
        c["exit_code"] == 0 and c["diagnostics_empty"] for c in report["checks"]
    )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
