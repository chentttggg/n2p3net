"""Verify Bayes/support-measure/noise algebra against the accepted probability base."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from run_lean import compile_slot, lean_environment, revalidate_environment
from verify import ALLOWED_AXIOMS, REPO, ROOT, digest, run
from verify_stage import lean_code, theorem_inventory

MODULES = (
    "N2P3.BayesDensity",
    "N2P3.RetainedMeasure",
    "N2P3.RetainedMatrixMeasure",
    "N2P3.GaussianPredictive",
)
NEGATIVES = ("negative/FalseBayesDensity.lean", "negative/FalsePredictive.lean")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lake", type=Path, default=Path("D:/BCI/lean-tools/lean-4.24.0-windows/bin/lake.exe")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    base_path = REPO / "doc/assets/lean_probability_20261016/acceptance.json"
    manifest_path = REPO / "frozen/lean_probability_20261016/manifest.json"
    base_hash = digest(base_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not base["passed"] or manifest["producer_receipt_sha256"] != base_hash:
        raise ValueError("The required accepted algebra base is not intact")
    base_proofs = {**base["reused_base_proofs_sha256"], **base["proof_sha256"]}
    if base_proofs != manifest["members_sha256"]:
        raise ValueError("Accepted proof coverage disagrees with its frozen manifest")
    for name, h in base_proofs.items():
        if digest(ROOT / name) != h:
            raise ValueError(f"Accepted proof input changed: {name}")
    for name, h in base["source_sha256"].items():
        if digest(REPO / name) != h:
            raise ValueError(f"Production source changed: {name}")
    module_hashes = {
        m.replace(".", "/") + ".lean": digest(ROOT / (m.replace(".", "/") + ".lean"))
        for m in MODULES
    }
    inventory = {m: theorem_inventory(m) for m in MODULES}
    if any(digest(ROOT / n) != h for n, h in module_hashes.items()):
        raise ValueError("New proof changed during inventory")
    names = [n for group in inventory.values() for n in group]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate theorem names")
    audit = ROOT / "AuditPosterior.lean"
    audit.write_text(
        "\n".join([*(f"import {m}" for m in MODULES), "", *(f"#print axioms {n}" for n in names)])
        + "\n",
        encoding="utf-8",
    )
    proof_files = [
        *(ROOT / n for n in module_hashes),
        ROOT / "N2P3/PosteriorSlice.lean",
        audit,
        *(ROOT / n for n in NEGATIVES),
        *sorted((ROOT / "scripts").glob("*.py")),
    ]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in proof_files}
    if any(hashes[n] != h for n, h in module_hashes.items()):
        raise ValueError("New proof changed before recording identity")
    env = lean_environment(args.lake.resolve())
    env["LEAN_NUM_THREADS"] = "1"
    lake = str(args.lake.resolve())
    lean = str(args.lake.resolve().with_name("lean.exe" if os.name == "nt" else "lean"))
    version_check = run([lean, "--version"], output, "version", env)
    version = (output / version_check["log"]).read_text(encoding="utf-8").strip()
    if version_check["exit_code"] != 0 or version != base["versions"]:
        raise ValueError("Lean version differs from the accepted algebra base")
    report = {
        "schema": "n2p3_lean_posterior_slice/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "proper-prior/likelihood Bayes density, basis-independent retained support volume, multichannel lifting and correlated-noise covariance algebra; joint disintegration, predictive marginalization and actual noise-block assumption binding remain open",
        "base_acceptance": base_path.relative_to(REPO).as_posix(),
        "base_acceptance_sha256": base_hash,
        "base_manifest": manifest_path.relative_to(REPO).as_posix(),
        "base_manifest_sha256": manifest_hash,
        "reused_base_proofs_sha256": base_proofs,
        "reused_base_theorems": {**base["reused_base_theorems"], **base["new_theorems"]},
        "source_sha256": base["source_sha256"],
        "versions": version,
        "version_check": version_check,
        "tool_executables_sha256": {lake: digest(Path(lake)), lean: digest(Path(lean))},
        "proof_sha256": hashes,
        "new_theorems": inventory,
        "maximum_parallel_checks": 2,
        "lean_threads_per_check": 1,
        "checks": [],
    }
    commands = [("build", [lake, "build", "N2P3.PosteriorSlice"])]
    commands += [
        (f"recheck_{m.rsplit('.', 1)[-1]}", [lean, "-j1", m.replace(".", "/") + ".lean"])
        for m in MODULES
    ]
    commands += [("audit", [lean, "-j1", "AuditPosterior.lean"])]
    commands += [(f"reject_{Path(n).stem}", [lean, "-j1", n]) for n in NEGATIVES]

    def check_command(name: str, command: list[str]) -> dict:
        queued = time.perf_counter()
        with compile_slot(shared=name != "build"):
            queue_seconds = time.perf_counter() - queued
            check = run(command, output, name, revalidate_environment(env))
        text = (output / check["log"]).read_text(encoding="utf-8")
        check.update(
            name=name,
            compile_queue_seconds=round(queue_seconds, 4),
            passed=check["exit_code"] == 0 and "declaration uses 'sorry'" not in text,
        )
        if name.startswith("recheck_"):
            check["diagnostics_empty"] = not text.strip()
            check["passed"] &= check["diagnostics_empty"]
        if name.startswith("reject_"):
            code = lean_code((ROOT / command[-1]).read_text(encoding="utf-8"))
            count = len(re.findall(r"^example\s*[:({]", code, re.M))
            errors = re.findall(r"^.*?:\d+:\d+: error(?:\([^)]*\))?: (.*)$", text, re.M)
            check.update(
                expected_false_examples=count,
                observed_error_diagnostics=errors,
                passed=check["exit_code"] != 0
                and count > 0
                and errors == ["unsolved goals"] * count
                and text.count("⊢ False") == count,
            )
        if name == "audit" and check["passed"]:
            printed = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", text)
            audited = [n for n, _ in printed] + re.findall(
                r"'([^']+)' does not depend on any axioms", text
            )
            axioms = {a.strip() for _, group in printed for a in group.split(",") if a.strip()}
            check.update(
                audited_theorems=audited,
                observed_axioms=sorted(axioms),
                passed=sorted(audited) == sorted(names) and axioms <= ALLOWED_AXIOMS,
            )
        return check

    def record(check: dict) -> None:
        report["checks"].append(check)
        print(json.dumps({k: v for k, v in check.items() if k != "audited_theorems"}), flush=True)

    started = time.perf_counter()
    build = check_command(*commands[0])
    record(build)
    if build["passed"]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(check_command, *command) for command in commands[1:]]
            for future in as_completed(futures):
                record(future.result())
    report["checks_wall_seconds"] = round(time.perf_counter() - started, 4)
    report["sources_and_base_unchanged"] = (
        all(digest(REPO / n) == h for n, h in report["source_sha256"].items())
        and all(digest(ROOT / n) == h for n, h in base_proofs.items())
        and digest(base_path) == base_hash
        and digest(manifest_path) == manifest_hash
    )
    report["proof_source_unchanged"] = all(digest(ROOT / n) == h for n, h in hashes.items())
    report["tool_executables_unchanged"] = all(
        digest(Path(n)) == h for n, h in report["tool_executables_sha256"].items()
    )
    report["passed"] = (
        len(report["checks"]) == len(commands)
        and all(c["passed"] for c in report["checks"])
        and report["sources_and_base_unchanged"]
        and report["proof_source_unchanged"]
        and report["tool_executables_unchanged"]
    )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
