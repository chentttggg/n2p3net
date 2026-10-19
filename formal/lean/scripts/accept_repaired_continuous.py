"""Reuse immutable successful checks and strictly recheck the repaired negative."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from run_lean import compile_slot, lean_environment, revalidate_environment
from verify import REPO, ROOT, digest, run
from verify_continuous import MODULES, NEGATIVES
from verify_stage import MODULES as BASE_MODULES
from verify_stage import lean_code, theorem_inventory

NEGATIVE = "negative/FalseProjection.lean"
CHECK = "reject_FalseProjection"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lake", type=Path, default=Path("D:/BCI/lean-tools/lean-4.24.0-windows/bin/lake.exe")
    )
    args = parser.parse_args()
    original = args.original.resolve()
    original_bytes = original.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    report = json.loads(original_bytes)
    repair_script = Path(__file__).resolve()
    expected = {"build", "audit", *(f"recheck_{m.rsplit('.', 1)[-1]}" for m in MODULES)}
    expected |= {f"reject_{Path(n).stem}" for n in NEGATIVES}
    checks = report["checks"]
    if len(checks) != len(expected) or {c["name"] for c in checks} != expected:
        raise ValueError("Original round must contain all expected checks exactly once")
    if [c["name"] for c in checks if not c["passed"]] != [CHECK]:
        raise ValueError("Only the identified projection negative may require repair")
    if not report["sources_and_base_unchanged"] or not report["proof_source_unchanged"]:
        raise ValueError("Original check inputs were not immutable")
    base_path = REPO / report["base_acceptance"]
    base_bytes = base_path.read_bytes()
    if hashlib.sha256(base_bytes).hexdigest() != report["base_acceptance_sha256"]:
        raise ValueError("Accepted base receipt changed")
    base = json.loads(base_bytes)
    if not base["passed"] or report["source_sha256"] != base["source_sha256"]:
        raise ValueError("Source coverage must exactly equal the accepted base")
    expected_base = {
        *(m.replace(".", "/") + ".lean" for m in BASE_MODULES),
        "lean-toolchain",
        "lake-manifest.json",
        "N2P3Program.lean",
    }
    if report["reused_base_proofs_sha256"] != {n: base["proof_sha256"][n] for n in expected_base}:
        raise ValueError("Accepted base proof coverage changed")
    expected_proofs = {
        *(m.replace(".", "/") + ".lean" for m in MODULES),
        "N2P3Continuous.lean",
        "lakefile.toml",
        "AuditContinuous.lean",
        *NEGATIVES,
        *(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "scripts").glob("*.py")
            if p.resolve() != repair_script
        ),
    }
    if set(report["proof_sha256"]) != expected_proofs:
        raise ValueError("Required proof, runner and configuration coverage is incomplete")
    reused_logs = {
        c["name"]: {
            "path": original.parent / c["log"],
            "sha256": digest(original.parent / c["log"]),
        }
        for c in checks
        if c["name"] != CHECK
    }
    for name, h in report["source_sha256"].items():
        if digest(REPO / name) != h:
            raise ValueError(f"Production source changed: {name}")
    for mapping in (report["reused_base_proofs_sha256"], report["proof_sha256"]):
        for name, h in mapping.items():
            if name != NEGATIVE and digest(ROOT / name) != h:
                raise ValueError(f"Previously checked input changed: {name}")
    if {m: theorem_inventory(m) for m in MODULES} != report["new_theorems"]:
        raise ValueError("Theorem inventory changed")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = digest(ROOT / NEGATIVE)
    script_hash = digest(repair_script)
    env = lean_environment(args.lake.resolve())
    lean = args.lake.resolve().with_name("lean.exe")
    queued = time.perf_counter()
    with compile_slot(shared=True):
        queue_seconds = time.perf_counter() - queued
        check = run([str(lean), "-j1", NEGATIVE], args.output, CHECK, revalidate_environment(env))
    log = (args.output / check["log"]).read_text(encoding="utf-8")
    count = len(
        re.findall(
            r"^example\s*[:({]", lean_code((ROOT / NEGATIVE).read_text(encoding="utf-8")), re.M
        )
    )
    errors = re.findall(r"^.*?:\d+:\d+: error(?:\([^)]*\))?: (.*)$", log, re.M)
    check.update(
        name=CHECK,
        compile_queue_seconds=round(queue_seconds, 4),
        expected_false_examples=count,
        observed_error_diagnostics=errors,
        passed=(
            check["exit_code"] != 0
            and count == 2
            and errors == ["unsolved goals"] * count
            and log.count("⊢ False") == count
        ),
    )
    provenance = {"path": original.relative_to(REPO).as_posix(), "sha256": original_hash}
    for old in checks:
        if old["name"] != CHECK:
            old["producer_report"] = provenance
            old["producer_log_sha256"] = reused_logs[old["name"]]["sha256"]
            old["producer_log_path"] = (original.parent / old["log"]).relative_to(REPO).as_posix()
    report["checks"] = [check if old["name"] == CHECK else old for old in checks]
    report["initial_round"] = provenance
    report["initial_round_checks_wall_seconds"] = report.pop("checks_wall_seconds")
    report["repair_scope"] = (
        "only equivalent finite negative matrix representation changed; successful kernel, axiom and other negative checks retain their original producer identity"
    )
    report["proof_sha256"][NEGATIVE] = source_hash
    report["proof_sha256"][repair_script.relative_to(ROOT).as_posix()] = script_hash
    report["sources_and_base_unchanged"] = (
        all(digest(REPO / n) == h for n, h in report["source_sha256"].items())
        and all(digest(ROOT / n) == h for n, h in report["reused_base_proofs_sha256"].items())
        and (digest(base_path) == report["base_acceptance_sha256"])
    )
    report["reused_producer_inputs_unchanged"] = digest(original) == original_hash and all(
        digest(v["path"]) == v["sha256"] for v in reused_logs.values()
    )
    report["proof_source_unchanged"] = all(
        digest(ROOT / n) == h for n, h in report["proof_sha256"].items()
    )
    report["passed"] = (
        all(c["passed"] for c in report["checks"])
        and report["sources_and_base_unchanged"]
        and report["proof_source_unchanged"]
        and report["reused_producer_inputs_unchanged"]
    )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "repair": check}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
