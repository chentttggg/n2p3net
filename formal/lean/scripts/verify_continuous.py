"""Check a new continuous-model algebra layer against its frozen accepted base."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from run_lean import compile_slot, lean_environment, revalidate_environment
from verify import ALLOWED_AXIOMS, REPO, ROOT, digest, run
from verify_stage import MODULES as BASE_MODULES
from verify_stage import lean_code, theorem_inventory

MODULES = (
    "N2P3.GaussianLowRank",
    "N2P3.ScaleGauge",
    "N2P3.NuisanceProjection",
    "N2P3.GaussianConditioning",
)
NEGATIVES = (
    "negative/FalseGaussianLowRank.lean",
    "negative/FalseScaleGauge.lean",
    "negative/FalseProjection.lean",
)


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
    base_path = REPO / "doc/assets/lean_first_slice_20261016/acceptance.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if not base["passed"]:
        raise ValueError("The earlier proof base was not accepted")
    base_proofs = {
        m.replace(".", "/") + ".lean": base["proof_sha256"][m.replace(".", "/") + ".lean"]
        for m in BASE_MODULES
    }
    base_proofs.update(
        {
            p: base["proof_sha256"][p]
            for p in ["lean-toolchain", "lake-manifest.json", "N2P3Program.lean"]
        }
    )
    for name, h in base_proofs.items():
        if digest(ROOT / name) != h:
            raise ValueError(f"Accepted proof base changed: {name}")
    for name, h in base["source_sha256"].items():
        if digest(REPO / name) != h:
            raise ValueError(f"Production source changed: {name}")
    module_hashes = {
        m.replace(".", "/") + ".lean": digest(ROOT / (m.replace(".", "/") + ".lean"))
        for m in MODULES
    }
    inventory = {m: theorem_inventory(m) for m in MODULES}
    if any(digest(ROOT / n) != h for n, h in module_hashes.items()):
        raise ValueError("Proof source changed while collecting theorem inventory")
    names = [n for ns in inventory.values() for n in ns]
    audit = ROOT / "AuditContinuous.lean"
    audit.write_text(
        "\n".join([*(f"import {m}" for m in MODULES), "", *(f"#print axioms {n}" for n in names)])
        + "\n",
        encoding="utf-8",
    )
    proofs = [
        *(ROOT / (m.replace(".", "/") + ".lean") for m in MODULES),
        ROOT / "N2P3Continuous.lean",
        ROOT / "lakefile.toml",
        audit,
        *(ROOT / n for n in NEGATIVES),
        *sorted((ROOT / "scripts").glob("*.py")),
    ]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in proofs}
    if any(hashes[n] != h for n, h in module_hashes.items()):
        raise ValueError("Proof source changed before recording verification identity")
    env = lean_environment(args.lake.resolve())
    env["LEAN_NUM_THREADS"] = "1"
    lake = str(args.lake.resolve())
    lean = str(args.lake.resolve().with_name("lean.exe" if os.name == "nt" else "lean"))
    report = {
        "schema": "n2p3_lean_continuous_algebra/1",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "low-rank/scale/support/AR1 whitening and Gaussian kernel algebra; full V03 normalized probability, numerical QR and IEEE obligations remain open",
        "base_acceptance": base_path.relative_to(REPO).as_posix(),
        "base_acceptance_sha256": digest(base_path),
        "reused_base_proofs_sha256": base_proofs,
        "reused_base_theorems": base["theorems"],
        "source_sha256": base["source_sha256"],
        "proof_sha256": hashes,
        "new_theorems": inventory,
        "lean_threads": 1,
        "maximum_parallel_checks": 2,
        "versions": subprocess.check_output(
            [lean, "--version"], cwd=ROOT, env=env, text=True
        ).strip(),
        "checks": [],
    }
    commands = [("build", [lake, "build", "N2P3Continuous"])]
    commands += [
        (
            f"recheck_{m.rsplit('.', 1)[-1]}",
            [lean, "-j1", m.replace(".", "/") + ".lean"],
        )
        for m in MODULES
    ]
    commands.append(("audit", [lean, "-j1", "AuditContinuous.lean"]))
    commands += [(f"reject_{Path(n).stem}", [lean, "-j1", n]) for n in NEGATIVES]

    def check_command(name: str, command: list[str]) -> dict:
        queued = time.perf_counter()
        with compile_slot(shared=name != "build"):
            queue_seconds = time.perf_counter() - queued
            check = run(command, output, name, revalidate_environment(env))
        check["compile_queue_seconds"] = round(queue_seconds, 4)
        text = (output / check["log"]).read_text(encoding="utf-8")
        check["passed"] = check["exit_code"] == 0 and "declaration uses 'sorry'" not in text
        if name.startswith("recheck_"):
            check["diagnostics_empty"] = not text.strip()
            check["passed"] &= check["diagnostics_empty"]
        if name.startswith("reject_"):
            code = lean_code((ROOT / command[-1]).read_text(encoding="utf-8"))
            count = len(re.findall(r"^example\s*[:({]", code, re.MULTILINE))
            diagnostics = re.findall(
                r"^.*?:\d+:\d+: error(?:\([^)]*\))?: (.*)$", text, re.MULTILINE
            )
            check["expected_false_examples"] = count
            check["observed_error_diagnostics"] = diagnostics
            check["passed"] = (
                check["exit_code"] != 0
                and count > 0
                and len(diagnostics) == count
                and all(d == "unsolved goals" for d in diagnostics)
                and text.count("⊢ False") == count
            )
        if name == "audit" and check["passed"]:
            printed = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", text)
            audited = [n for n, _ in printed] + re.findall(
                r"'([^']+)' does not depend on any axioms", text
            )
            axioms = {a.strip() for _, group in printed for a in group.split(",") if a.strip()}
            check["observed_axioms"] = sorted(axioms)
            check["audited_theorems"] = audited
            check["passed"] = sorted(audited) == sorted(names) and axioms <= ALLOWED_AXIOMS
        return check

    def record(name: str, check: dict) -> None:
        check["name"] = name
        report["checks"].append(check)
        print(
            json.dumps(
                {"check": name, **{k: v for k, v in check.items() if k != "audited_theorems"}}
            ),
            flush=True,
        )

    started = time.perf_counter()
    build = check_command(*commands[0])
    record("build", build)
    if build["passed"]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(check_command, *item): item[0] for item in commands[1:]}
            for future in as_completed(futures):
                record(futures[future], future.result())
    report["checks_wall_seconds"] = round(time.perf_counter() - started, 4)
    report["sources_and_base_unchanged"] = all(
        digest(REPO / n) == h for n, h in report["source_sha256"].items()
    ) and all(digest(ROOT / n) == h for n, h in base_proofs.items())
    report["proof_source_unchanged"] = all(digest(ROOT / n) == h for n, h in hashes.items())
    report["passed"] = (
        len(report["checks"]) == len(commands)
        and all(c["passed"] for c in report["checks"])
        and report["sources_and_base_unchanged"]
        and report["proof_source_unchanged"]
    )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
