"""Execute a reviewed component inventory without claiming full P01-P12 completion."""

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
from verify_erp import theorem_inventory
from verify_stage import FORBIDDEN, lean_code


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lake", type=Path, default=Path("D:/BCI/lean-tools/lean-4.24.0-windows/bin/lake.exe")
    )
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan_hash = digest(plan_path)
    plan = json.loads(plan_path.read_bytes())
    if plan["schema"] != "n2p3_personal_erp_component_check_plan/1":
        raise ValueError("Unexpected reviewed execution plan schema")
    if ALLOWED_AXIOMS != {"propext", "Classical.choice", "Quot.sound"}:
        raise ValueError("Standard axiom policy changed")
    contract_path = REPO / plan["contract"]
    if digest(contract_path) != plan["contract_sha256"]:
        raise ValueError("Contract identity changed")
    contract = json.loads(contract_path.read_bytes())
    base_path, manifest_path = REPO / contract["base_acceptance"], REPO / contract["base_manifest"]
    base, manifest = json.loads(base_path.read_bytes()), json.loads(manifest_path.read_bytes())
    if digest(base_path) != contract["base_acceptance_sha256"] or not base["passed"]:
        raise ValueError("Historical accepted base changed")
    base_hash, manifest_hash = digest(base_path), digest(manifest_path)
    inherited = {**base["reused_base_proofs_sha256"], **base["proof_sha256"]}
    if inherited != manifest["members_sha256"] or manifest["producer_receipt_sha256"] != base_hash:
        raise ValueError("Historical manifest and receipt differ")
    if len(inherited) != plan["inherited_member_count"]:
        raise ValueError("Historical proof inventory changed")
    source_hashes, proof_hashes = contract["source_sha256"], plan["proof_sha256"]
    runtime_inventory = {
        path.relative_to(REPO).as_posix() for path in (REPO / "src").rglob("*.py")
    } | {path.relative_to(REPO).as_posix() for path in (REPO / "experiments").glob("*.py")}
    expected_runtime = {name for name in source_hashes if name.startswith(("src/", "experiments/"))}
    if runtime_inventory != expected_runtime or set(inherited) & set(proof_hashes):
        raise ValueError("Runtime inventory changed or a new proof would replace an inherited member")
    for name, expected in source_hashes.items():
        if digest(REPO / name) != expected:
            raise ValueError(f"Current source changed: {name}")
    for name, expected in {**inherited, **proof_hashes}.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Proof/runner identity changed: {name}")
    required = {module.replace(".", "/") + ".lean" for module in plan["modules"]}
    required |= set(plan["negative_examples"]) | {plan["aggregate"], plan["audit"]}
    required.add(Path(__file__).resolve().relative_to(ROOT).as_posix())
    if required != set(proof_hashes):
        raise ValueError("Execution plan does not cover every new proof, audit and runner")
    inventory = {module: theorem_inventory(module) for module in plan["modules"]}
    if inventory != plan["theorems"]:
        raise ValueError("Reviewed declaration inventory changed")
    names = [name for declarations in inventory.values() for name in declarations]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate declaration in reviewed inventory")
    for path, count in plan["negative_examples"].items():
        code = lean_code((ROOT / path).read_text(encoding="utf-8"))
        if type(count) is not int or count < 1 or FORBIDDEN.search(code):
            raise ValueError(f"Invalid negative-control inventory: {path}")
        if len(re.findall(r"^example\s*[:({]", code, re.M)) != count:
            raise ValueError(f"Reviewed negative denominator changed: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    lake, lean = str(args.lake.resolve()), str(args.lake.resolve().with_name("lean.exe"))
    env = dict(os.environ)
    env.update(LEAN_NUM_THREADS="1", MATHLIB_CACHE_DIR="D:/BCI/lean-tools/mathlib-cache")
    env = lean_environment(args.lake, env)
    tool_hashes = {name: digest(Path(name)) for name in (lake, lean)}
    if tool_hashes != base["tool_executables_sha256"]:
        raise ValueError("Pinned Lean/Lake binary identities differ from the accepted base")
    commands = [("build", [lake, "build", plan["aggregate"].removesuffix(".lean").replace("/", ".")])]
    commands += [(f"recheck_{module.rsplit('.', 1)[-1]}", [lean, "-j1", module.replace(".", "/") + ".lean"])
                 for module in plan["modules"]]
    commands += [("audit", [lean, "-j1", plan["audit"]])]
    commands += [(f"reject_{Path(path).stem}", [lean, "-j1", path]) for path in plan["negative_examples"]]
    report = {
        "schema": "n2p3_personal_erp_component_execution/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": plan["scope"], "uncompleted_obligations": plan["uncompleted_obligations"],
        "whole_contract_or_project_complete": False,
        "plan": str(plan_path), "plan_sha256": plan_hash,
        "contract": plan["contract"], "contract_sha256": digest(contract_path),
        "base_acceptance": str(base_path), "base_acceptance_sha256": base_hash,
        "base_manifest": str(manifest_path), "base_manifest_sha256": manifest_hash,
        "historical_source_boundary": "Inherited proof identities are preserved; old source acceptance is not reclassified as current-source refinement.",
        "source_sha256": source_hashes, "proof_sha256": proof_hashes,
        "reused_base_proofs_sha256": inherited, "new_theorems": inventory,
        "new_theorem_count": len(names), "new_false_example_count": sum(plan["negative_examples"].values()),
        "tool_executables_sha256": tool_hashes,
        "versions": base["versions"], "pinned_binaries_match_accepted_base": True,
        "maximum_parallel_checks": 2, "lean_threads_per_check": 1,
        "expected_command_count": len(commands), "checks": [],
    }

    def check_command(name: str, command: list[str]) -> dict:
        queued = time.perf_counter()
        with compile_slot(shared=name != "build"):
            result = run(command, output, name, revalidate_environment(env))
        text = (output / result["log"]).read_text(encoding="utf-8")
        result.update(name=name, compile_queue_and_check_seconds=round(time.perf_counter() - queued, 4))
        result["passed"] = result["exit_code"] == 0 and "sorry" not in text.lower() and "warning" not in text.lower()
        if name.startswith("recheck_"):
            result["diagnostics_empty"] = not text.strip()
            result["passed"] &= result["diagnostics_empty"]
        elif name.startswith("reject_"):
            count = plan["negative_examples"][command[-1]]
            errors = re.findall(r"^.*?:\d+:\d+: error(?:\([^)]*\))?: (.*)$", text, re.M)
            result.update(expected_false_examples=count, observed_error_diagnostics=errors)
            result["passed"] = (result["exit_code"] == 1 and errors == ["unsolved goals"] * count
                                and text.count("⊢ False") == count
                                and "warning" not in text.lower() and "sorry" not in text.lower())
        elif name == "audit" and result["passed"]:
            printed = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", text)
            audited = [name for name, _ in printed] + re.findall(r"'([^']+)' does not depend on any axioms", text)
            axioms = {axiom.strip() for _, group in printed for axiom in group.split(",") if axiom.strip()}
            result.update(audited_theorems=audited, observed_axioms=sorted(axioms))
            result["passed"] &= sorted(audited) == sorted(names) and axioms <= ALLOWED_AXIOMS
        return result

    def record(result: dict) -> None:
        report["checks"].append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "audited_theorems"}), flush=True)

    started = time.perf_counter()
    build = check_command(*commands[0])
    record(build)
    if build["passed"]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(check_command, *command) for command in commands[1:]]):
                record(future.result())
    report["checks_wall_seconds"] = round(time.perf_counter() - started, 4)
    report["sources_and_base_unchanged"] = (
        all(digest(REPO / name) == expected for name, expected in source_hashes.items())
        and all(digest(ROOT / name) == expected for name, expected in inherited.items())
        and digest(base_path) == base_hash and digest(manifest_path) == manifest_hash
        and digest(contract_path) == plan["contract_sha256"]
    )
    runtime_after = {
        path.relative_to(REPO).as_posix() for path in (REPO / "src").rglob("*.py")
    } | {path.relative_to(REPO).as_posix() for path in (REPO / "experiments").glob("*.py")}
    report["runtime_inventory_unchanged"] = runtime_after == expected_runtime
    report["proof_and_plan_unchanged"] = (digest(plan_path) == plan_hash
        and all(digest(ROOT / name) == expected for name, expected in proof_hashes.items()))
    report["tools_unchanged"] = all(digest(Path(name)) == expected for name, expected in tool_hashes.items())
    report["passed"] = (len(report["checks"]) == len(commands) and all(check["passed"] for check in report["checks"])
                        and report["sources_and_base_unchanged"] and report["proof_and_plan_unchanged"]
                        and report["tools_unchanged"] and report["runtime_inventory_unchanged"])
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
