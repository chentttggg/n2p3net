"""Verify explicit proof modules, all their theorem axioms and falsity controls."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import psutil
from run_lean import compile_slot
from verify import ALLOWED_AXIOMS, REPO, ROOT, digest, run

MODULES = (
    "N2P3Pilot",
    "N2P3.RawScope",
    "N2P3.FiniteSOS",
    "N2P3.FiniteAdjoint",
    "N2P3.Covariance",
    "N2P3.FirstSlice",
)
NEGATIVES = (
    "negative/FalseOffset.lean",
    "negative/FalsePositiveDefinite.lean",
    "negative/FalseRawScope.lean",
    "negative/FalseSOS.lean",
    "negative/FalseCovarianceBound.lean",
)
FORBIDDEN = re.compile(
    r"\b(?:sorry|admit|axiom|native_decide|unsafe|extern|implemented_by|run_tac|run_elab)\b"
    r"|debug\.skipKernelTC"
)
DECLARATION = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)*(?:protected\s+)?(?:theorem|lemma)\s+([\w'.]+)")


def lean_code(text: str) -> str:
    """Remove nested comments and strings, retaining line structure for inventory."""
    result = []
    i, depth = 0, 0
    in_string = False
    while i < len(text):
        pair = text[i : i + 2]
        if depth:
            if pair == "/-":
                depth += 1
                result.extend("  ")
                i += 2
            elif pair == "-/":
                depth -= 1
                result.extend("  ")
                i += 2
            else:
                result.append("\n" if text[i] == "\n" else " ")
                i += 1
        elif in_string:
            if text[i] == "\\":
                result.extend("  ")
                i += 2
            elif text[i] == '"':
                in_string = False
                result.append(" ")
                i += 1
            else:
                result.append("\n" if text[i] == "\n" else " ")
                i += 1
        elif pair == "/-":
            depth = 1
            result.extend("  ")
            i += 2
        elif pair == "--":
            end = text.find("\n", i)
            end = len(text) if end < 0 else end
            result.extend(" " * (end - i))
            i = end
        elif text[i] == '"':
            in_string = True
            result.append(" ")
            i += 1
        else:
            result.append(text[i])
            i += 1
    if depth or in_string:
        raise ValueError("Unterminated Lean comment/string")
    return "".join(result)


def theorem_inventory(module: str) -> list[str]:
    path = ROOT / (module.replace(".", "/") + ".lean")
    code = lean_code(path.read_text(encoding="utf-8"))
    if FORBIDDEN.search(code):
        raise ValueError(f"Unfinished proof or extra-trust construct: {path}")
    scopes: list[tuple[str, str]] = []
    names = []
    for line in code.splitlines():
        stripped = line.strip()
        match = re.match(r"namespace\s+([\w.]+)\s*$", stripped)
        if match:
            scopes.append(("namespace", match.group(1)))
            continue
        if re.fullmatch(r"(?:noncomputable\s+)?section(?:\s+[\w.]+)?", stripped):
            scopes.append(("section", ""))
            continue
        if re.fullmatch(r"end(?:\s+[\w.]+)?", stripped):
            if not scopes:
                raise ValueError(f"Unbalanced scope in {path}")
            scopes.pop()
            continue
        if match := DECLARATION.match(line):
            prefix = ".".join(name for kind, name in scopes if kind == "namespace")
            names.append(f"{prefix}.{match.group(1)}" if prefix else match.group(1))
    if scopes or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid theorem inventory for {path}")
    return names


def call(command: list[str], output: Path, name: str, env: dict) -> dict:
    with compile_slot():
        result = run(command, output, name, env)
    log = (output / result["log"]).read_text(encoding="utf-8")
    result["passed"] = result["exit_code"] == 0 and "declaration uses 'sorry'" not in log
    return result


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lake", type=Path, default=Path("D:/BCI/lean-tools/lean-4.24.0-windows/bin/lake.exe")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    env = dict(os.environ)
    env["PATH"] = str(args.lake.resolve().parent) + os.pathsep + env.get("PATH", "")
    env["LEAN_NUM_THREADS"] = "1"
    lake = str(args.lake.resolve())
    inventory = {module: theorem_inventory(module) for module in MODULES}
    names = [name for group in inventory.values() for name in group]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate theorem names")
    audit = ROOT / "AuditStage.lean"
    audit_text = (
        "\n".join(
            [
                *(f"import {module}" for module in MODULES),
                "",
                *(f"#print axioms {name}" for name in names),
            ]
        )
        + "\n"
    )
    audit.write_text(audit_text, encoding="utf-8")
    runtime_paths = subprocess.check_output(
        ["rg", "--files", "src", "experiments", "-g", "*.py"], cwd=REPO, text=True
    ).splitlines()
    source_hashes = {Path(p).as_posix(): digest(REPO / p) for p in sorted(runtime_paths)}
    proof_paths = [
        *(ROOT / (m.replace(".", "/") + ".lean") for m in MODULES),
        ROOT / "N2P3Program.lean",
        audit,
        ROOT / "lakefile.toml",
        ROOT / "lake-manifest.json",
        ROOT / "lean-toolchain",
        *(ROOT / p for p in NEGATIVES),
        *sorted((ROOT / "scripts").glob("*.py")),
    ]
    proof_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in proof_paths}
    result = {
        "schema": "n2p3_lean_stage_acceptance/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "first approved real-mathematics slice; Python/IEEE refinement separately tracked",
        "lean_threads": 1,
        "host": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(),
            "total_ram_bytes": psutil.virtual_memory().total,
            "available_ram_before_bytes": psutil.virtual_memory().available,
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "versions": subprocess.check_output(
            [lake, "env", "lean", "--version"], cwd=ROOT, env=env, text=True
        ).strip(),
        "theorems": inventory,
        "source_sha256": source_hashes,
        "proof_sha256": proof_hashes,
        "checks": [],
    }
    commands = [("build", [lake, "build", "N2P3Program"])]
    commands += [
        (
            f"recheck_{module.replace('.', '_')}",
            [lake, "env", "lean", "-j1", module.replace(".", "/") + ".lean"],
        )
        for module in MODULES
    ]
    commands.append(("audit", [lake, "env", "lean", "-j1", "AuditStage.lean"]))
    commands += [(f"reject_{Path(p).stem}", [lake, "env", "lean", "-j1", p]) for p in NEGATIVES]
    for name, command in commands:
        check = call(command, output, name, env)
        log = (output / check["log"]).read_text(encoding="utf-8")
        if name.startswith("reject_"):
            negative_path = ROOT / command[-1]
            negative_code = lean_code(negative_path.read_text(encoding="utf-8"))
            expected_examples = len(re.findall(r"^example\s*[:({]", negative_code, re.MULTILINE))
            diagnostics = re.findall(r"^.*?:\d+:\d+: error(?:\([^)]*\))?: (.*)$", log, re.MULTILINE)
            check["passed"] = (
                check["exit_code"] != 0
                and expected_examples > 0
                and len(diagnostics) == expected_examples
                and all(diagnostic == "unsolved goals" for diagnostic in diagnostics)
                and log.count("⊢ False") == expected_examples
            )
            check["expected_false_examples"] = expected_examples
            check["observed_error_diagnostics"] = diagnostics
        if name == "audit" and check["passed"]:
            printed = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", log)
            no_axioms = re.findall(r"'([^']+)' does not depend on any axioms", log)
            audited = [n for n, _ in printed] + no_axioms
            axioms = {a.strip() for _, group in printed for a in group.split(",") if a.strip()}
            check["observed_axioms"] = sorted(axioms)
            check["audited_theorems"] = audited
            check["passed"] = axioms <= ALLOWED_AXIOMS and sorted(audited) == sorted(names)
        result["checks"].append(check)
        print(json.dumps({"check": name, **check}), flush=True)
        if not check["passed"]:
            break
    result["source_unchanged_during_checks"] = all(
        digest(REPO / name) == h for name, h in source_hashes.items()
    )
    result["proof_unchanged_during_checks"] = all(
        digest(ROOT / name) == h for name, h in proof_hashes.items()
    )
    result["passed"] = (
        len(result["checks"]) == len(commands)
        and all(c["passed"] for c in result["checks"])
        and result["source_unchanged_during_checks"]
        and result["proof_unchanged_during_checks"]
    )
    result["finished_utc"] = datetime.now(UTC).isoformat()
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
