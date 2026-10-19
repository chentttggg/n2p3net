"""Verify explicit ERP features, temporal contrast Grams, tapering and state replay."""

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
    "N2P3.ERPFeatures",
    "N2P3.ERPTemporalBasis",
    "N2P3.ERPToeplitz",
    "N2P3.ERPState",
)
NEGATIVES = (
    "negative/FalseERPFeatures.lean",
    "negative/FalseERPTemporalBasis.lean",
    "negative/FalseERPToeplitz.lean",
    "negative/FalseERPState.lean",
)
EXPECTED_BASE_MEMBER_COUNT = 114
EXPECTED_RUNTIME_SOURCE_COUNT = 189
EXPECTED_COMMAND_COUNT = 10
EXPECTED_BASE_THEOREM_COUNT = 1150
EXPECTED_BASE_FALSE_EXAMPLES = 110
EXPECTED_MANIFEST_FALSE_EXAMPLES = 101
EXPECTED_BASE_RECEIPT_SHA256 = "662eaa5e74cd19f8c4d593388a150f89aa6511774e6018790a298cf4cd785b11"
EXPECTED_BASE_MANIFEST_SHA256 = "f10144f57f32c5f71ae1832a8b93d2b20c7d0ec667e545512c289422ad0219eb"
EXPECTED_BASE_EVIDENCE_SHA256 = "f7c07ba8aa74f5bc10c3d6e5c8c1691624b9a6f2b77f6aa19f8235f50048ad12"
# Denominators are fixed by each module's independent strict negative check.
EXPECTED_FALSE_EXAMPLES = dict(zip(NEGATIVES, (6, 4, 4, 8), strict=True))
# Counts identify the final source inventory; they are not proof quotas.
EXPECTED_NEW_THEOREM_COUNTS = dict(zip(MODULES, (46, 40, 35, 51), strict=True))
EXPECTED_PROOF_SHA256 = {
    "N2P3/ERPFeatures.lean": "cbdabe62820725dea0592ca570cf86266b6d905caf6fc0deef638a278bb1c825",
    "N2P3/ERPTemporalBasis.lean": "ce71990c640f6b7c784c0ad1acea3b12cca780ff56757476fcd82a4833547f19",
    "N2P3/ERPToeplitz.lean": "5dfcfdb6972d1f3e5ed56e52dcdfcda19d627f0caf9cdd719c130a75ebfb5680",
    "N2P3/ERPState.lean": "584d18a7cfe9682b033ba9e6d2c056d04cf055aedc9e88a95a52265b61f1121e",
    "negative/FalseERPFeatures.lean": "9e27c3f862a50b779047304d6aebd1667121d1f8066b7a96b3281cc79e2bece1",
    "negative/FalseERPTemporalBasis.lean": "de99b2f8ff6fdcfd6fe6ef7c26b355faaa41ea59ecaf53296007405df9100dab",
    "negative/FalseERPToeplitz.lean": "2f189f99d51a73b328fc39e14f9fc6a3c5aaeb127544798302d14be75a58975a",
    "negative/FalseERPState.lean": "7e66857285d69008558f996ee960ebd88ca3afcb351852b189c2218b947774eb",
}


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
    if set(EXPECTED_FALSE_EXAMPLES) != set(NEGATIVES) or any(
        type(count) is not int or count < 1 for count in EXPECTED_FALSE_EXAMPLES.values()
    ):
        raise ValueError("Expected negative-control denominators must be positive integers")
    if set(EXPECTED_NEW_THEOREM_COUNTS) != set(MODULES) or any(
        type(count) is not int or count < 1 for count in EXPECTED_NEW_THEOREM_COUNTS.values()
    ):
        raise ValueError("Final independently reviewed ERP theorem denominators are not frozen")
    expected_proof_paths = {m.replace(".", "/") + ".lean" for m in MODULES} | set(NEGATIVES)
    if set(EXPECTED_PROOF_SHA256) != expected_proof_paths or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in EXPECTED_PROOF_SHA256.values()
    ):
        raise ValueError("Final independently reviewed ERP proof identities are not frozen")
    if ALLOWED_AXIOMS != {"propext", "Classical.choice", "Quot.sound"}:
        raise ValueError("Accepted standard axiom policy changed")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    base_path = REPO / "doc/assets/lean_temporal_operator_20261016/acceptance.json"
    manifest_path = REPO / "frozen/lean_temporal_operator_20261018/manifest.json"
    evidence_path = REPO / "doc/assets/lean_temporal_operator_20261016/evidence_manifest.json"
    base_bytes = base_path.read_bytes()
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    base = json.loads(base_bytes)
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    evidence_bytes = evidence_path.read_bytes()
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()
    evidence = json.loads(evidence_bytes)
    if (
        base_hash != EXPECTED_BASE_RECEIPT_SHA256
        or manifest_hash != EXPECTED_BASE_MANIFEST_SHA256
        or evidence_hash != EXPECTED_BASE_EVIDENCE_SHA256
        or not base["passed"]
        or manifest["producer_receipt_sha256"] != base_hash
        or evidence["acceptance_sha256"] != base_hash
    ):
        raise ValueError("The required accepted temporal/operator base is not intact")
    if (
        evidence["cumulative_theorems"] != EXPECTED_BASE_THEOREM_COUNT
        or evidence["cumulative_false_examples"] != EXPECTED_BASE_FALSE_EXAMPLES
    ):
        raise ValueError("Historical accepted denominators differ from the prior evidence receipt")
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
    reused_theorems = {**base["reused_base_theorems"], **base["new_theorems"]}
    reused_names = [name for group in reused_theorems.values() for name in group]
    if len(reused_names) != EXPECTED_BASE_THEOREM_COUNT or len(reused_names) != len(set(reused_names)):
        raise ValueError("The accepted 1150-theorem base is not reused exactly")
    manifest_false_examples = {
        name: len(re.findall(r"^example\s*[:({]", lean_code((ROOT / name).read_text(encoding="utf-8")), re.M))
        for name in base_proofs
        if name.startswith("negative/") and name.endswith(".lean")
    }
    if sum(manifest_false_examples.values()) != EXPECTED_MANIFEST_FALSE_EXAMPLES:
        raise ValueError("Current frozen-manifest negative source denominator changed")
    for name, expected_hash in EXPECTED_PROOF_SHA256.items():
        if digest(ROOT / name) != expected_hash:
            raise ValueError(f"ERP proof differs from its independently reviewed final identity: {name}")
    module_hashes = {
        m.replace(".", "/") + ".lean": digest(ROOT / (m.replace(".", "/") + ".lean"))
        for m in MODULES
    }
    inventory = {m: theorem_inventory(m) for m in MODULES}
    if {module: len(group) for module, group in inventory.items()} != EXPECTED_NEW_THEOREM_COUNTS:
        raise ValueError("ERP theorem source inventory differs from its final reviewed denominator")
    if any(digest(ROOT / n) != h for n, h in module_hashes.items()):
        raise ValueError("New proof changed during inventory")
    names = [n for group in inventory.values() for n in group]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate theorem names")
    if set(names) & set(reused_names):
        raise ValueError("A new ERP theorem duplicates a reused base declaration")
    expected_slice = "\n".join(["import N2P3.TemporalOperatorSlice", *(f"import {m}" for m in MODULES)]) + "\n"
    if (ROOT / "N2P3/ERPSlice.lean").read_text(encoding="utf-8") != expected_slice:
        raise ValueError("ERP aggregation module must only import its accepted base and four new modules")
    audit = ROOT / "AuditERP.lean"
    audit.write_text(
        "\n".join([*(f"import {m}" for m in MODULES), "", *(f"#print axioms {n}" for n in names)])
        + "\n",
        encoding="utf-8",
    )
    proof_files = [
        *(ROOT / n for n in module_hashes),
        ROOT / "N2P3/ERPSlice.lean",
        audit,
        *(ROOT / n for n in NEGATIVES),
        *sorted((ROOT / "scripts").glob("*.py")),
    ]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in proof_files}
    if any(hashes[n] != h for n, h in module_hashes.items()):
        raise ValueError("New proof changed before recording identity")
    if any(hashes[n] != h for n, h in EXPECTED_PROOF_SHA256.items()):
        raise ValueError("New proof identity changed before recording the acceptance inputs")
    env = lean_environment(args.lake.resolve())
    env["LEAN_NUM_THREADS"] = "1"
    lake = str(args.lake.resolve())
    lean = str(args.lake.resolve().with_name("lean.exe" if os.name == "nt" else "lean"))
    tool_hashes = {lake: digest(Path(lake)), lean: digest(Path(lean))}
    if set(tool_hashes.values()) != set(base["tool_executables_sha256"].values()):
        raise ValueError("Lean/Lake executables differ from the accepted base tools")
    version_check = run([lean, "--version"], output, "version", env)
    version = (output / version_check["log"]).read_text(encoding="utf-8").strip()
    if version_check["exit_code"] != 0 or version != base["versions"]:
        raise ValueError("Lean version differs from the accepted temporal/operator base")
    report = {
        "schema": "n2p3_lean_erp_slice/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "literal complete-bin means, channel-prime features, temporal projection and affine score; explicit selected class means, equal-participant trace-normalized ERP contrast Gram; actual lag-write Toeplitz loop and finite zero-extension congruences preserving PSD/PD on stated domains; family-specific ERP numeric state guards and canonical reader/writer/personal-prefix prediction bridges; exact-real and decoded semantic layer only, with numerical solvers, full IEEE/NumPy refinement, xDAWN/OAS/Riemann forward, fit authority and physical provenance kept separate",
        "base_acceptance": base_path.relative_to(REPO).as_posix(),
        "base_acceptance_sha256": base_hash,
        "base_manifest": manifest_path.relative_to(REPO).as_posix(),
        "base_manifest_sha256": manifest_hash,
        "base_evidence_manifest": evidence_path.relative_to(REPO).as_posix(),
        "base_evidence_manifest_sha256": evidence_hash,
        "reused_base_proofs_sha256": base_proofs,
        "reused_base_theorems": reused_theorems,
        "reused_base_theorem_count": len(reused_names),
        "historical_base_false_examples": EXPECTED_BASE_FALSE_EXAMPLES,
        "manifest_negative_source_inventory": manifest_false_examples,
        "manifest_negative_source_count": sum(manifest_false_examples.values()),
        "negative_inheritance_boundary": "historical accepted110 comes from the bound prior evidence receipt; latest114-member manifest includes101 negative examples, while the initial9 retain their original producer/archive identity and are not rechecked here",
        "source_sha256": base["source_sha256"],
        "inherited_member_count": len(base_proofs),
        "runtime_source_count": len(base["source_sha256"]),
        "versions": version,
        "version_check": version_check,
        "tool_executables_sha256": tool_hashes,
        "proof_sha256": hashes,
        "new_theorems": inventory,
        "new_theorem_count": len(names),
        "expected_new_theorem_counts": EXPECTED_NEW_THEOREM_COUNTS,
        "expected_reviewed_proof_sha256": EXPECTED_PROOF_SHA256,
        "new_false_example_count": sum(EXPECTED_FALSE_EXAMPLES.values()),
        "software_source_probe_is_separate_evidence": True,
        "source_probe_counts_are_not_Lean_acceptance": True,
        "maximum_parallel_checks": 2,
        "lean_threads_per_check": 1,
        "expected_command_count": EXPECTED_COMMAND_COUNT,
        "expected_false_examples": EXPECTED_FALSE_EXAMPLES,
        "checks": [],
    }
    commands = [("build", [lake, "build", "N2P3.ERPSlice"])]
    commands += [
        (f"recheck_{m.rsplit('.', 1)[-1]}", [lean, "-j1", m.replace(".", "/") + ".lean"])
        for m in MODULES
    ]
    commands += [("audit", [lean, "-j1", "AuditERP.lean"])]
    commands += [(f"reject_{Path(n).stem}", [lean, "-j1", n]) for n in NEGATIVES]
    if len(commands) != EXPECTED_COMMAND_COUNT:
        raise ValueError("ERP verification command denominator changed")

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
        and digest(evidence_path) == evidence_hash
    )
    report["proof_source_unchanged"] = all(digest(ROOT / n) == h for n, h in hashes.items())
    report["tool_executables_unchanged"] = all(
        digest(Path(n)) == h for n, h in report["tool_executables_sha256"].items()
    )
    report["passed"] = (
        len(report["checks"]) == len(commands) == EXPECTED_COMMAND_COUNT
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
