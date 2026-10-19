"""Verify literal event accumulation, finite history closure and constructor state ledgers."""

from __future__ import annotations

import argparse
import hashlib
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
from verify_stage import DECLARATION, FORBIDDEN, lean_code

MODULES = (
    "N2P3.EventDesign",
    "N2P3.EventHistory",
    "N2P3.StateLedger",
)
NEGATIVES = (
    "negative/FalseEventDesign.lean",
    "negative/FalseEventHistory.lean",
    "negative/FalseStateLedger.lean",
)
EXPECTED_BASE_MEMBER_COUNT = 84
EXPECTED_RUNTIME_SOURCE_COUNT = 189
EXPECTED_FALSE_EXAMPLES = dict(zip(NEGATIVES, (3, 3, 5), strict=True))


def theorem_inventory(module: str) -> list[str]:
    """Inventory explicit proofs, including mutually recursive value lemmas.

    Mutual blocks have no namespace component but do consume an `end`.
    The already accepted inventory and all inherited identities stay frozen.
    """
    path = ROOT / (module.replace(".", "/") + ".lean")
    code = lean_code(path.read_text(encoding="utf-8"))
    if FORBIDDEN.search(code):
        raise ValueError(f"Unfinished proof or extra-trust construct: {path}")
    scopes: list[tuple[str, str]] = []
    names = []
    for line in code.splitlines():
        stripped = line.strip()
        if match := re.fullmatch(r"namespace\s+([\w.]+)", stripped):
            scopes.append(("namespace", match.group(1)))
        elif re.fullmatch(r"(?:noncomputable\s+)?section(?:\s+[\w.]+)?", stripped):
            scopes.append(("section", ""))
        elif stripped == "mutual":
            scopes.append(("mutual", ""))
        elif re.fullmatch(r"end(?:\s+[\w.]+)?", stripped):
            if not scopes:
                raise ValueError(f"Unbalanced scope in {path}")
            scopes.pop()
        elif match := DECLARATION.match(line):
            prefix = ".".join(name for kind, name in scopes if kind == "namespace")
            names.append(f"{prefix}.{match.group(1)}" if prefix else match.group(1))
    if scopes or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid theorem inventory for {path}")
    return names


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lake", type=Path, default=Path("D:/BCI/lean-tools/lean-4.24.0-windows/bin/lake.exe")
    )
    args = parser.parse_args()
    if any(type(count) is not int or count < 1 for count in EXPECTED_FALSE_EXAMPLES.values()):
        raise ValueError("Expected negative-control denominators must be positive integers")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    base_path = REPO / "doc/assets/lean_joint_state_20261016/acceptance.json"
    manifest_path = REPO / "frozen/lean_joint_state_20261016/manifest.json"
    base_bytes = base_path.read_bytes()
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    base = json.loads(base_bytes)
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if not base["passed"] or manifest["producer_receipt_sha256"] != base_hash:
        raise ValueError("The required accepted algebra base is not intact")
    base_proofs = {**base["reused_base_proofs_sha256"], **base["proof_sha256"]}
    if base_proofs != manifest["members_sha256"]:
        raise ValueError("Accepted proof coverage disagrees with its frozen manifest")
    if len(base_proofs) != EXPECTED_BASE_MEMBER_COUNT:
        raise ValueError("Accepted proof/config/runner denominator differs from this slice")
    if len(base["source_sha256"]) != EXPECTED_RUNTIME_SOURCE_COUNT:
        raise ValueError("Production source denominator differs from this slice")
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
    audit = ROOT / "AuditEventLedger.lean"
    audit.write_text(
        "\n".join([*(f"import {m}" for m in MODULES), "", *(f"#print axioms {n}" for n in names)])
        + "\n",
        encoding="utf-8",
    )
    proof_files = [
        *(ROOT / n for n in module_hashes),
        ROOT / "N2P3/EventLedgerSlice.lean",
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
        "schema": "n2p3_lean_event_ledger_slice/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "literal per-event additive slice loop, affine label blocks and censored row restriction; finite-response event-history closure from validated repetition schedules and observed-set accounting; source-shaped constructor provenance ledgers over decoded finite semantic values; NumPy/MAT/JSON refinement and numerical/physical filter-support boundaries remain separate",
        "base_acceptance": base_path.relative_to(REPO).as_posix(),
        "base_acceptance_sha256": base_hash,
        "base_manifest": manifest_path.relative_to(REPO).as_posix(),
        "base_manifest_sha256": manifest_hash,
        "reused_base_proofs_sha256": base_proofs,
        "reused_base_theorems": {**base["reused_base_theorems"], **base["new_theorems"]},
        "source_sha256": base["source_sha256"],
        "inherited_member_count": len(base_proofs),
        "runtime_source_count": len(base["source_sha256"]),
        "versions": version,
        "version_check": version_check,
        "tool_executables_sha256": {lake: digest(Path(lake)), lean: digest(Path(lean))},
        "proof_sha256": hashes,
        "new_theorems": inventory,
        "maximum_parallel_checks": 2,
        "lean_threads_per_check": 1,
        "checks": [],
    }
    commands = [("build", [lake, "build", "N2P3.EventLedgerSlice"])]
    commands += [
        (f"recheck_{m.rsplit('.', 1)[-1]}", [lean, "-j1", m.replace(".", "/") + ".lean"])
        for m in MODULES
    ]
    commands += [("audit", [lean, "-j1", "AuditEventLedger.lean"])]
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
            expected = EXPECTED_FALSE_EXAMPLES[command[-1]]
            errors = re.findall(r"^.*?:\d+:\d+: error(?:\([^)]*\))?: (.*)$", text, re.M)
            check.update(
                expected_false_examples=expected,
                source_false_examples=count,
                observed_error_diagnostics=errors,
                passed=check["exit_code"] == 1
                and count == expected
                and not FORBIDDEN.search(code)
                and errors == ["unsolved goals"] * expected
                and text.count("⊢ False") == expected
                and "warning" not in text.lower()
                and "sorry" not in text.lower(),
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
