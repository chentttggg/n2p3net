"""Inspect current source/proof identity while preserving historical verifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def inspect(repository: Path, *, historical_repository: Path | None = None) -> dict:
    repository = repository.resolve()
    historical = repository if historical_repository is None else historical_repository.resolve()
    checkpoint_path = historical / "doc/assets/lean_gaussian_20261016/active_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_bytes())
    selected = checkpoint["active_source_contract"]
    contract_path = historical / selected["path"]
    contract = json.loads(contract_path.read_bytes())
    if sha(contract_path) != selected["sha256"]:
        raise ValueError("Active source contract SHA-256 mismatch")
    manifest_path = historical / checkpoint["frozen_manifest"]
    manifest = json.loads(manifest_path.read_bytes())
    accepted_path = historical / checkpoint["current_accepted_receipt"]
    accepted = json.loads(accepted_path.read_bytes())
    if (not accepted["passed"] or accepted["frozen_manifest"] != checkpoint["frozen_manifest"]
            or accepted["frozen_manifest_sha256"] != sha(manifest_path)):
        raise ValueError("Accepted proof manifest differs from its original producer receipt")
    source_drift = []
    proof_drift = []
    for name, expected in contract["source_sha256"].items():
        actual = sha(repository / name)
        if actual != expected:
            source_drift.append({"path": name, "expected_sha256": expected, "actual_sha256": actual})
    for name, expected in manifest["members_sha256"].items():
        actual = sha(repository / "formal/lean" / name)
        if actual != expected:
            proof_drift.append({"path": name, "expected_sha256": expected, "actual_sha256": actual})
    baseline = {p.relative_to(repository).as_posix() for p in (repository / "src").rglob("*.py")} | {
        p.relative_to(repository).as_posix() for p in (repository / "experiments").glob("*.py")}
    expected = set(contract["baseline_runtime_source_paths"])
    matches = not (source_drift or proof_drift or baseline != expected)
    return {"schema": "n2p3_lean_workspace_identity/1", "repository": str(repository),
            "historical_evidence_repository": str(historical),
            "status": "original_verified_source" if matches else "source_changed_not_revalidated",
            "original_verified_source": selected,
            "checkpoint_sha256": sha(checkpoint_path), "source_contract": selected,
            "source_files": len(contract["source_sha256"]),
            "proof_manifest": checkpoint["frozen_manifest"], "proof_manifest_sha256": sha(manifest_path),
            "accepted_receipt": checkpoint["current_accepted_receipt"], "accepted_receipt_sha256": sha(accepted_path),
            "proof_files": len(manifest["members_sha256"]), "source_drift": source_drift,
            "proof_drift": proof_drift, "new_runtime_paths": sorted(baseline - expected),
            "missing_runtime_paths": sorted(expected - baseline),
            "identity_matches": matches,
            "proof_execution_performed": False, "historical_receipts_modified": False}


def resolve_lake(explicit: Path | None) -> Path:
    candidate = explicit or os.environ.get("N2P3_LAKE") or shutil.which("lake")
    if candidate is None:
        raise FileNotFoundError("Supply --lake, set N2P3_LAKE, or install the pinned toolchain on PATH")
    path = Path(candidate).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def current_program_identity(repository: Path) -> dict:
    """Bind this compile to actual current runtime and proof inputs."""
    repository = repository.resolve()
    paths = set((repository / "src").rglob("*.py")) | set((repository / "experiments").glob("*.py"))
    for directory, directories, files in os.walk(repository / "formal/lean"):
        directories[:] = sorted(n for n in directories if n not in {".lake", "artifacts", "__pycache__"})
        for filename in sorted(files):
            path = Path(directory) / filename
            if path.suffix in {".lean", ".py", ".toml", ".json"} or filename == "lean-toolchain":
                paths.add(path)
    pins = {path.relative_to(repository).as_posix(): sha(path) for path in sorted(paths)}
    return {"files_sha256": pins, "file_count": len(pins),
            "scope": "all src Python, top-level experiment Python, and formal sources/scripts/configuration; excludes generated Lean caches and outputs",
            "inventory_sha256": hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--historical-repo", type=Path,
                        help="Read the unchanged original checkpoint/receipts here when inspecting another checkout")
    parser.add_argument("--lake", type=Path)
    parser.add_argument("--show-command", metavar="FILE",
                        help="Print the pinned targeted check command after inspecting identity; do not execute")
    parser.add_argument("--check-file", metavar="FILE",
                        help="Execute one targeted Lean file under the original source identity and compile slot")
    parser.add_argument("--compile-current-target", metavar="FILE",
                        help="Compile actual current inputs, allowing source changes without reusing old acceptance")
    parser.add_argument("--output", type=Path, help="New output directory required with an execution mode")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.show_command, args.check_file, args.compile_current_target)) > 1:
        parser.error("choose one target mode")
    execute_target = args.check_file or args.compile_current_target
    if bool(execute_target) != bool(args.output):
        parser.error("an execution mode and --output must be supplied together")
    result = inspect(args.repo, historical_repository=args.historical_repo)
    target = args.show_command or execute_target
    if target:
        if re.fullmatch(r"[A-Za-z_]\w*(?:/[A-Za-z_]\w*)*\.lean", target) is None:
            parser.error("target must be a repository-local Lean module file")
        lake = resolve_lake(args.lake)
        result["targeted_check"] = [str(lake), "env", "lean", "-j1", target]
        result["targeted_check_cwd"] = str(args.repo.resolve() / "formal/lean")
    result["original_acceptance_applicable"] = result["identity_matches"] and not bool(args.compile_current_target)
    if execute_target and (result["identity_matches"] or args.compile_current_target):
        lean_root = args.repo.resolve() / "formal/lean"
        sys.path.insert(0, str(lean_root / "scripts"))
        from run_lean import compile_slot
        from verify import ROOT, run

        if ROOT.resolve() != lean_root:
            raise ValueError("Loaded Lean runner belongs to another checkout")
        args.output.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ)
        env["LEAN_NUM_THREADS"] = "1"
        with compile_slot(shared=True, root=lean_root) as admission:
            current_before = current_program_identity(args.repo)
            measurement = run(result["targeted_check"], args.output, "lean", env)
            current_after = current_program_identity(args.repo)
        after = inspect(args.repo, historical_repository=args.historical_repo)
        current_unchanged = current_before == current_after
        passed = not measurement["exit_code"] and current_unchanged and (
            bool(args.compile_current_target) or after["identity_matches"])
        result.update({"proof_execution_performed": True, "before": result.copy(), "after": after,
                       "measurement": measurement, "admission": admission,
                       "lake_sha256": sha(lake), "driver_sha256": sha(Path(__file__)),
                       "not_component_acceptance": True, "passed": passed,
                       "mode": "compile_current_target" if args.compile_current_target else "check_original_source_file",
                       "current_program_before": current_before, "current_program_after": current_after,
                       "current_program_unchanged": current_unchanged,
                       "proof_exit_zero": measurement["exit_code"] == 0})
        (args.output / "workspace_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8", newline="\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if passed else (measurement["exit_code"] or 2)
    if execute_target:
        result["execution_status"] = "not_run_source_identity_mismatch"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["identity_matches"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
